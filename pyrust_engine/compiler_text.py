"""Code generation for strings, growable lists, dicts and sets (0.7.2).

Mixed into Compiler. Kinds: 'strv' (a string value), 'char', 'str' (a str argument),
'flist' / 'ilist' (lists of numbers), 'slist' (list of str), 'dict(K,V)', 'set(K)'.

Rules that keep results identical to Python:
- containers are never aliased: b = a for a list/dict/set runs as Python; copies with
  list(a) / dict(a) / set(a) are compiled,
- iterating a set is not compiled (Python's set order is not reproducible), but sets can be
  built, tested with `in`, combined with & | - ^, sorted, measured and returned,
- dicts keep insertion order, and changing a dict's size while looping over it replays in
  Python, which raises the same RuntimeError,
- lists, dicts and sets passed in as arguments are read-only (they are copied in); build a
  new one and return it.
"""

import ast

from .kinds import (
    ARRAY_INFO,
    ELEMENT,
    LIST_KINDS,
    NUMPY_KINDS,
    PLACEHOLDERS,
    TEXT,
    dict_parts,
    is_container,
    is_dict,
    is_packed,
    is_set,
    set_part,
    zero,
)

MUTATING_METHODS = {
    'append',
    'extend',
    'insert',
    'pop',
    'remove',
    'clear',
    'sort',
    'reverse',
    'add',
    'discard',
    'update',
    'setdefault',
}
STR_TESTS = {'isdigit': 'c_isdigit', 'isalpha': 'c_isalpha', 'isalnum': 'c_isalnum', 'isspace': 'c_isspace'}
GROWABLE = ('flist', 'ilist', 'slist', 'list?')


def _rename(node, mapping):
    """Copy of an AST with names renamed (comprehension variables get their own scope)."""

    class Renamer(ast.NodeTransformer):
        def visit_Name(self, n):
            if n.id in mapping:
                return ast.copy_location(ast.Name(id=mapping[n.id], ctx=n.ctx), n)
            return n

    return Renamer().visit(ast.parse(ast.unparse(node), mode='eval').body if not isinstance(node, ast.stmt) else node)


class TextMixin:
    # ------------------------------------------------------------------ kinds
    def kind_of(self, node):
        if isinstance(node, ast.Name) and node.id in self.assigned:
            return self.types.get(node.id)
        return None

    def refine(self, name, kind, node):
        """Fix the element/key/value types of an empty list, dict or set on first use."""
        current = self.types.get(name)
        if current == kind:
            return
        if current in PLACEHOLDERS:
            self.types[name] = kind
            self.refined[name] = kind
            return
        self.fail(
            node,
            f'{name!r} holds {current} values, not {kind}',
            'Keep one type per list/dict/set, for example only floats (write 1.0, not 1).',
        )

    def container_name(self, node):
        """(name, kind) for a local or argument list/dict/set used with a method."""
        if isinstance(node, ast.Name) and node.id in self.assigned:
            kind = self.types.get(node.id)
            if is_container(kind) or kind in LIST_KINDS:
                return node.id, kind
        return None, None

    def cref(self, name):
        """Immutable reference to a container (local value or argument reference)."""
        return f'(&*{self.name(name)})' if name in self.params else f'(&{self.name(name)})'

    def cmut(self, name, node):
        """Mutable access to a container; arguments that were copied in are read-only."""
        if self.parallel is not None:
            self.fail(node, 'lists, dicts and sets cannot be changed inside prange', 'Use a normal range loop.')
        if name in self.params:
            if self.is_top and (is_packed(self.types[name]) or self.types[name] in ('series', 'iseries')):
                self.fail(
                    node,
                    f'{name!r} was passed in and is read-only in compiled code',
                    'Build a new list/dict/set inside the function and return it.',
                )
            self.written.add(name)
            return f'(&mut *{self.name(name)})'
        return f'(&mut {self.name(name)})'

    # ---------------------------------------------------------------- strings
    def text(self, node):
        """(Rust &[u32] expression, kind) for a str argument, a string value or a character."""
        if isinstance(node, ast.Name) and node.id in self.assigned:
            kind = self.types.get(node.id)
            if kind == 'strv':
                return f'(&{self.name(node.id)}[..])', kind
            if kind == 'str':
                return f'sl({self.aref(node.id)})', kind
        value, kind = self.expr(node)
        return self.as_text(value, kind, node), kind

    def as_text(self, value, kind, node):
        if kind == 'strv':
            return f'(&({value})[..])'
        if kind == 'char':
            return f'std::slice::from_ref(&({value}))'
        if kind == 'str':
            return f'sl({value})'
        self.fail(node, 'expected a string')

    def to_s(self, value, kind, node):
        if kind == 'strv':
            return value
        if kind == 'char':
            return f'sv(&[{value}])'
        if kind == 'str':
            return f'sv(sl({value}))'
        self.fail(node, 'expected a string')

    def string_literal(self, text):
        if not text:
            return 'S::default()'
        return 'sv(&[' + ','.join(f'{ord(c)}u32' for c in text) + '])'

    def item(self, node, want):
        """Value for a list element / dict key or value / set element of kind `want`."""
        value, kind = self.expr(node)
        if want == 'strv' and kind in TEXT:
            return self.to_s(value, kind, node)
        if kind == want:
            return value
        hints = {
            'f64': 'This container holds floats; Python would store the int as an int. Write float(x).',
            'i64': 'This container holds ints. Convert with int(x) first.',
        }
        self.fail(node, f'expected a {want} value here, got {kind}', hints.get(want))

    @staticmethod
    def element_kind(kind):
        """Container element kind for a value kind (characters become strings)."""
        if kind in TEXT:
            return 'strv'
        return kind

    # ------------------------------------------------------------ expressions
    def ext_expr(self, n):
        if isinstance(n, ast.Constant) and type(n.value) is str:
            return self.string_literal(n.value), 'strv'
        if isinstance(n, ast.JoinedStr):
            return self.fstring(n)
        if isinstance(n, ast.List):
            return self.list_literal(n)
        if isinstance(n, ast.ListComp):
            return self.comprehension(n)
        if isinstance(n, ast.Dict):
            return self.dict_literal(n)
        if isinstance(n, ast.Set):
            return self.set_literal(n)
        if isinstance(n, ast.Subscript):
            return self.ext_subscript(n)
        return None

    def fstring(self, n):
        t = self.temp('f')
        parts = [f'let mut {t} = S::default();']
        for value in n.values:
            if isinstance(value, ast.Constant):
                if value.value:
                    parts.append(f's_push(&mut {t},&{self.string_literal(value.value)}[..]);')
                continue
            if value.conversion != -1 or value.format_spec is not None:
                self.fail(value, 'f-string conversions and format specs are not compiled', 'Write f"{x}" only.')
            piece, kind = self.expr(value.value)
            if kind == 'i64':
                piece = f'(&s_int({piece})[..])'
            elif kind == 'bool':
                piece = f'(&s_bool({piece})[..])'
            elif kind in TEXT:
                piece = self.as_text(piece, kind, value.value)
            else:
                self.fail(
                    value,
                    'only ints, bools and strings are formatted in compiled f-strings',
                    "Python's float formatting is not reproduced; format floats outside the function.",
                )
            parts.append(f's_push(&mut {t},{piece});')
        return '{ ' + ' '.join(parts) + f' {t} }}', 'strv'

    def list_literal(self, n):
        if not n.elts:
            return zero('list?'), 'list?'
        if any(isinstance(e, ast.Starred) for e in n.elts):
            self.fail(n, '*unpacking in list literals is not compiled')
        first, kind = self.expr(n.elts[0])
        kind = self.element_kind(kind)
        if kind == 'strv':
            return 'vec![' + ','.join(self.item(e, 'strv') for e in n.elts) + ']', 'slist'
        if kind in ('i64', 'f64'):
            values = ','.join(self.item(e, kind) for e in n.elts)
            return f'arr_from(vec![{values}],{zero(kind)}{kind})?', 'flist' if kind == 'f64' else 'ilist'
        self.fail(n, 'lists of ints, floats or strings are compiled')

    def comprehension(self, n):
        if len(n.generators) != 1 or n.generators[0].is_async:
            self.fail(n, 'only comprehensions with one `for` are compiled', 'Write two nested loops with append().')
        gen = n.generators[0]
        self.comp_count += 1
        target = f'_comp{self.comp_count}'
        names = {x.id for x in ast.walk(gen.target) if isinstance(x, ast.Name)}
        mapping = {name: f'_comp{self.comp_count}_{name}' for name in names}
        self.local_names |= {target, *mapping.values()}
        loop_target = _rename(gen.target, mapping) if mapping else gen.target
        for x in ast.walk(loop_target):
            if isinstance(x, ast.Name):
                x.ctx = ast.Store()
        element = _rename(n.elt, mapping)
        conditions = [_rename(c, mapping) for c in gen.ifs]
        append = ast.Expr(
            value=ast.Call(
                func=ast.Attribute(value=ast.Name(id=target, ctx=ast.Load()), attr='append', ctx=ast.Load()),
                args=[element],
                keywords=[],
            )
        )
        body = [append]
        if conditions:
            test = conditions[0] if len(conditions) == 1 else ast.BoolOp(op=ast.And(), values=conditions)
            body = [ast.If(test=test, body=[append], orelse=[])]
        statements = [
            ast.Assign(targets=[ast.Name(id=target, ctx=ast.Store())], value=ast.List(elts=[], ctx=ast.Load())),
            ast.For(target=loop_target, iter=gen.iter, body=body, orelse=[]),
        ]
        for s in statements:
            ast.copy_location(s, n)
            ast.fix_missing_locations(s)
        exits = getattr(self, 'exits', False)
        code = self.block(statements)
        self.exits = exits
        return f'{{ {code}\nstd::mem::take(&mut {self.name(target)}) }}', self.types[target]

    def dict_literal(self, n):
        if not n.keys:
            return zero('dict?'), 'dict?'
        if any(k is None for k in n.keys):
            self.fail(n, '**unpacking in dict literals is not compiled')
        _, kk = self.expr(n.keys[0])
        _, vk = self.expr(n.values[0])
        kk, vk = self.element_kind(kk), self.element_kind(vk)
        self.check_key(n, kk)
        t = self.temp('d')
        sets = ' '.join(f'{t}.set({self.item(k, kk)},{self.item(v, vk)});' for k, v in zip(n.keys, n.values))
        return f'{{ let mut {t} = OMap::new(); {sets} {t} }}', f'dict({kk},{vk})'

    def set_literal(self, n):
        _, kk = self.expr(n.elts[0])
        kk = self.element_kind(kk)
        self.check_key(n, kk)
        t = self.temp('s')
        adds = ' '.join(f'{t}.insert({self.item(e, kk)});' for e in n.elts)
        return f'{{ let mut {t} = HashSet::new(); {adds} {t} }}', f'set({kk})'

    def check_key(self, node, kind):
        if kind not in ('i64', 'strv'):
            self.fail(
                node,
                f'dict keys and set elements must be ints or strings here (got {kind})',
                'Python treats 1 and 1.0 as the same key; floats and bools are not compiled as keys.',
            )

    def bound(self, node):
        if node is None or (isinstance(node, ast.Constant) and node.value is None):
            return 'None'
        return f'Some({self.expect(node, "i64")})'

    def ext_subscript(self, n):
        base = n.value
        if isinstance(base, ast.Attribute):
            return None  # a.shape[0], s.iloc[i]
        kind = self.kind_of(base)
        if kind is None:
            if isinstance(base, ast.Name):
                return None
            value, kind = self.expr(base)
            ref = f'(&{value})'
        else:
            ref = self.cref(base.id)
        if isinstance(n.slice, ast.Slice):
            sl = n.slice
            if sl.step is not None and not (isinstance(sl.step, ast.Constant) and sl.step.value in (1, None)):
                self.fail(n, 'slices with a step are not compiled', 'Loop with range(start, stop, step) instead.')
            if kind in NUMPY_KINDS:
                self.fail(
                    n,
                    'NumPy slices are views that share memory, so they are not compiled',
                    'Loop over indices instead: for i in range(start, stop): ... a[i] ...',
                )
            lo, hi = self.bound(sl.lower), self.bound(sl.upper)
            if kind in ('str', 'strv', 'char'):
                text = self.as_text(value, kind, base) if not isinstance(base, ast.Name) else self.text(base)[0]
                return f's_slice({text},{lo},{hi})', 'strv'
            if kind in LIST_KINDS:
                return f'a_slice({ref},{lo},{hi},{zero(ELEMENT[kind])}{ELEMENT[kind]})?', kind
            if kind == 'slist':
                return f'v_slice({ref},{lo},{hi})', 'slist'
            self.fail(n, f'slicing {kind} is not compiled')
        if kind == 'strv':
            text = self.as_text(value, kind, base) if not isinstance(base, ast.Name) else self.text(base)[0]
            return f's_at({text},{self.expect(n.slice, "i64")})?', 'char'
        if kind == 'slist':
            return f'v_at({ref},{self.expect(n.slice, "i64")})?', 'strv'
        if is_dict(kind):
            k, v = dict_parts(kind)
            return f'{ref}.need(&{self.item(n.slice, k)})?', v
        if kind == 'dict?':
            self.fail(n, 'reading from a dict before anything was stored in it', 'This raises KeyError in Python.')
        if kind == 'list?':
            self.fail(n, 'reading from an empty list raises IndexError in Python')
        if not isinstance(base, ast.Name):
            self.fail(n, f'indexing {kind} is not compiled here')
        return None

    # --------------------------------------------------------------- binops
    def ext_binop(self, n, op, a, ta, left, b, tb, right):
        if ta in TEXT and tb in TEXT and isinstance(op, ast.Add):
            return f's_cat({self.as_text(a, ta, left)},{self.as_text(b, tb, right)})', 'strv'
        if isinstance(op, ast.Mult) and {ta, tb} & set(TEXT) and {ta, tb} & {'i64'}:
            text, count = (a, b) if ta in TEXT else (b, a)
            kind = ta if ta in TEXT else tb
            return f's_mul({self.as_text(text, kind, n)},{count})?', 'strv'
        if ta == tb and ta in LIST_KINDS + ('slist',) and isinstance(op, ast.Add):
            refs = [self.cref(x.id) if isinstance(x, ast.Name) else f'(&{v})' for x, v in ((left, a), (right, b))]
            if ta == 'slist':
                return f'[{refs[0]}.as_slice(),{refs[1]}.as_slice()].concat()', 'slist'
            el = ELEMENT[ta]
            return f'arr_from([av({refs[0]}),av({refs[1]})].concat(),{zero(el)}{el})?', ta
        if is_set(ta) and ta == tb:
            helper = {ast.BitAnd: 'set_and', ast.BitOr: 'set_or', ast.Sub: 'set_sub', ast.BitXor: 'set_xor'}.get(
                type(op)
            )
            if helper:
                refs = [self.cref(x.id) if isinstance(x, ast.Name) else f'(&{v})' for x, v in ((left, a), (right, b))]
                return f'{helper}({refs[0]},{refs[1]})', ta
        if any(k in ('strv', 'str', 'char') or is_container(k) for k in (ta, tb)):
            self.fail(
                n,
                f'only numbers can be used in arithmetic here (got {ta} and {tb})',
                'Strings support + (joining) and * (repeating); lists support +; sets support & | - ^.',
            )
        return None

    def text_compare(self, n, op, a, ta, left, b, tb, right):
        if ta not in TEXT or tb not in TEXT:
            self.fail(n, 'strings can only be compared with strings')
        symbol = {ast.Lt: '<', ast.LtE: '<=', ast.Gt: '>', ast.GtE: '>=', ast.Eq: '==', ast.NotEq: '!='}[type(op)]
        # Code-point order, like Python's str comparison.
        return f'(({self.as_text(a, ta, left)}) {symbol} ({self.as_text(b, tb, right)}))'

    # ------------------------------------------------------------------ calls
    def ext_call(self, n):
        func = n.func
        if isinstance(func, ast.Name) and func.id not in self.local_names:
            if func.id in ('sorted', 'str', 'list', 'set', 'dict', 'len', 'min', 'max', 'int', 'float'):
                if self.is_builtin(func.id):
                    return self.ext_builtin(n, func.id)
            return None
        if not isinstance(func, ast.Attribute):
            return None
        base = func.value
        if isinstance(base, ast.Name) and base.id not in self.assigned:
            return None  # module functions: math.sqrt, np.zeros, random.random, ...
        kind = self.kind_of(base)
        if kind is None:
            value, kind = self.expr(base)
        else:
            value = None
        if kind == 'char':
            return None  # one-character methods are handled by the scalar compiler
        if kind in ('str', 'strv'):
            return self.string_method(n, func.attr, base)
        if kind in GROWABLE or kind in LIST_KINDS:
            if func.attr == 'pop':
                return self.list_pop(n, base)
            if func.attr in ('index', 'count'):
                return self.list_search(n, base, kind, func.attr)
        if is_dict(kind) or kind == 'dict?':
            return self.dict_method(n, base, kind, func.attr)
        if func.attr in MUTATING_METHODS:
            self.fail(n, f'.{func.attr}() returns None in Python; use it as a statement on its own line')
        if value is not None or kind is not None:
            self.fail(n, f'.{func.attr}() is not compiled for {kind}')
        return None

    def string_method(self, n, attr, base):
        text, _ = self.text(base)
        args = n.args
        if n.keywords:
            self.fail(n, f'keyword arguments for str.{attr}() are not compiled')
        if attr in ('find', 'rfind', 'index', 'count') and 1 <= len(args) <= 3:
            sub, _ = self.text(args[0])
            start = self.bound(args[1]) if len(args) > 1 else 'None'
            end = self.bound(args[2]) if len(args) > 2 else 'None'
            if attr == 'index':
                t = self.temp('x')
                return (
                    f'({{ let {t} = s_find({text},{sub},{start},{end}); if {t} < 0 {{ return Err(6) }} {t} }})',
                    'i64',
                )
            helper = {'find': 's_find', 'rfind': 's_rfind', 'count': 's_count'}[attr]
            return f'{helper}({text},{sub},{start},{end})', 'i64'
        if attr in ('startswith', 'endswith') and len(args) == 1:
            if isinstance(args[0], ast.Tuple):
                self.fail(n, f'{attr}() with a tuple is not compiled', f'Write s.{attr}(a) or s.{attr}(b).')
            sub, _ = self.text(args[0])
            return f'{"s_starts" if attr == "startswith" else "s_ends"}({text},{sub})', 'bool'
        if attr == 'split' and not args:
            return f's_split_ws({text})', 'slist'
        if attr == 'split' and len(args) == 1:
            sep, _ = self.text(args[0])
            return f's_split({text},{sep})?', 'slist'
        if attr in ('strip', 'lstrip', 'rstrip') and len(args) <= 1:
            chars = f'Some({self.text(args[0])[0]})' if args else 'None'
            left, right = attr != 'rstrip', attr != 'lstrip'
            return f's_strip({text},{chars},{str(left).lower()},{str(right).lower()})', 'strv'
        if attr in ('lower', 'upper') and not args:
            return f's_{attr}({text})?', 'strv'
        if attr == 'replace' and len(args) == 2:
            old, _ = self.text(args[0])
            new, _ = self.text(args[1])
            return f's_replace({text},{old},{new})?', 'strv'
        if attr == 'join' and len(args) == 1:
            items, kind = self.sequence_of_strings(args[0])
            return f's_join({text},{items})', 'strv'
        if attr in STR_TESTS and not args:
            return f's_all({text},{STR_TESTS[attr]})?', 'bool'
        self.fail(
            n,
            f'str.{attr}() is not compiled with these arguments',
            'Compiled: find rfind index count startswith endswith split strip lstrip rstrip lower upper '
            'replace join isdigit isalpha isalnum isspace.',
        )

    def sequence_of_strings(self, node):
        """Rust &[S] for a list of strings (or the characters of a string)."""
        kind = self.kind_of(node)
        if kind == 'slist':
            return f'({self.cref(node.id)}.as_slice())', kind
        value, kind = self.expr(node)
        if kind == 'slist':
            return f'(&({value})[..])', kind
        if kind in TEXT:
            return f'(&s_chars({self.as_text(value, kind, node)})[..])', kind
        self.fail(node, 'expected a list of strings')

    def list_pop(self, n, base):
        name, kind = self.container_name(base)
        if name is None or n.keywords or len(n.args) > 1:
            self.fail(n, 'pop() is compiled on a list variable, as lst.pop() or lst.pop(i)')
        if name in self.params and kind in LIST_KINDS:
            self.fail(n, 'lists passed in cannot grow or shrink in compiled code', 'Build a new list and return it.')
        index = self.expect(n.args[0], 'i64') if n.args else '-1i64'
        if kind == 'slist':
            return f'v_pop({self.cmut(name, n)},{index})?', 'strv'
        if kind in PLACEHOLDERS:
            self.fail(n, 'pop() from an empty list raises IndexError in Python')
        return f'{self.cmut(name, n)}.pop_at({index})?', ELEMENT[kind]

    def list_search(self, n, base, kind, attr):
        if len(n.args) != 1 or n.keywords:
            self.fail(n, f'list.{attr}(x) takes one argument here')
        name, _ = self.container_name(base)
        ref = self.cref(name)
        t = self.temp('q')
        if kind == 'slist':
            needle, _ = self.text(n.args[0])
            test = f'&e[..]=={needle}'
            items = f'{ref}.iter()'
        else:
            needle, nk = self.expr(n.args[0])
            el = ELEMENT[kind]
            if nk != el:
                self.fail(n, f'list.{attr}() needs a value of the same type as the list')
            if el == 'f64':
                needle = f'({{ let {t}v = {needle}; if {t}v.is_nan() {{ return Err(6) }} {t}v }})'
            test = f'*e=={needle}'
            items = f'av({ref}).iter()'
        if attr == 'count':
            return f'({items}.filter(|e| {test}).count() as i64)', 'i64'
        return f'({items}.position(|e| {test}).map(|p| p as i64).ok_or(6)?)', 'i64'

    def dict_method(self, n, base, kind, attr):
        name, _ = self.container_name(base)
        if name is None:
            self.fail(n, 'dict methods are compiled on dict variables')
        args = n.args
        if attr in ('get', 'pop', 'setdefault') and 1 <= len(args) <= 2 and not n.keywords:
            if len(args) == 1 and attr in ('get', 'setdefault'):
                self.fail(
                    n,
                    f'd.{attr}(key) without a default can return None, which is not compiled',
                    f'Pass a default of the right type: d.{attr}(key, 0).',
                )
            key_value, key_kind = self.expr(args[0])
            key_kind = self.element_kind(key_kind)
            if kind == 'dict?':
                if len(args) < 2:
                    self.fail(n, 'pop() from an empty dict raises KeyError in Python')
                _, default_kind = self.expr(args[1])
                self.check_key(n, key_kind)
                self.refine(name, f'dict({key_kind},{self.element_kind(default_kind)})', n)
                kind = self.types[name]
            k, v = dict_parts(kind)
            key = self.item(args[0], k)
            default = self.item(args[1], v) if len(args) == 2 else None
            t = self.temp('g')
            if attr == 'get':
                return (
                    f'({{ let {t}k = {key}; let {t}d = {default}; {self.cref(name)}.get(&{t}k).unwrap_or({t}d) }})',
                    v,
                )
            if attr == 'pop':
                target = self.cmut(name, n)
                if default is None:
                    return f'{target}.del(&{key})?', v
                return (
                    f'({{ let {t}k = {key}; let {t}d = {default}; '
                    f'if {target}.has(&{t}k) {{ {target}.del(&{t}k)? }} else {{ {t}d }} }})'
                ), v
            target = self.cmut(name, n)
            return (
                f'({{ let {t}k = {key}; let {t}d = {default}; '
                f'if !{target}.has(&{t}k) {{ {target}.set({t}k.clone(),{t}d); }} {target}.need(&{t}k)? }})'
            ), v
        self.fail(
            n,
            f'dict.{attr}() is not compiled here',
            'Compiled: d[k], d.get(k, default), d.pop(k[, default]), d.setdefault(k, v), k in d, len(d), '
            'and for loops over d, d.keys(), d.values(), d.items().',
        )

    def ext_builtin(self, n, name):
        args = n.args
        keywords = {kw.arg: kw.value for kw in n.keywords}
        if name == 'len' and len(args) == 1 and not keywords:
            kind = self.kind_of(args[0])
            if kind is None:
                value, kind = self.expr(args[0])
                if kind == 'strv':
                    return f'((&({value})[..]).len() as i64)', 'i64'
                if kind == 'slist' or is_set(kind):
                    return f'(({value}).len() as i64)', 'i64'
                if is_dict(kind):
                    return f'({value}).len()', 'i64'
                return None
            if kind == 'strv':
                return f'({self.name(args[0].id)}.len() as i64)', 'i64'
            if kind == 'slist' or is_set(kind) or kind in ('list?', 'set?'):
                return f'({self.cref(args[0].id)}.len() as i64)', 'i64'
            if is_dict(kind) or kind == 'dict?':
                return f'{self.cref(args[0].id)}.len()', 'i64'
            return None
        if name == 'str' and len(args) == 1 and not keywords:
            value, kind = self.expr(args[0])
            self.builtin('str')
            if kind == 'i64':
                return f's_int({value})', 'strv'
            if kind == 'bool':
                return f's_bool({value})', 'strv'
            if kind in TEXT:
                return self.to_s(value, kind, args[0]), 'strv'
            self.fail(n, f'str() of {kind} is not compiled', "Python's float formatting is not reproduced.")
        if name in ('int', 'float') and len(args) == 1 and not keywords:
            value, kind = self.expr(args[0])
            if kind not in ('str', 'strv'):
                return None
            self.builtin(name)
            text = self.as_text(value, kind, args[0])
            return (f's_parse_int({text})?', 'i64') if name == 'int' else (f's_parse_float({text})?', 'f64')
        if name == 'sorted' and len(args) == 1 and set(keywords) <= {'reverse'}:
            self.builtin('sorted')
            return self.sorted_call(n, args[0], keywords.get('reverse'))
        if name in ('list', 'set', 'dict') and not keywords and len(args) <= 1:
            self.builtin(name)
            if not args:
                kind = {'list': 'list?', 'set': 'set?', 'dict': 'dict?'}[name]
                return zero(kind), kind
            return self.convert_container(n, name, args[0])
        if name in ('min', 'max') and len(args) == 1:
            if keywords and set(keywords) == {'key'}:
                return self.arg_best(n, name, args[0], keywords['key'])
            if keywords:
                return None
            self.builtin(name)
            return self.sequence_best(n, name, args[0])
        return None

    def materialize(self, node):
        """(Rust expression of an owned/borrowed container, kind) for any expression."""
        kind = self.kind_of(node)
        if kind is not None:
            if kind in ('str',):
                return f'sl({self.aref(node.id)})', kind
            if kind == 'strv':
                return f'(&{self.name(node.id)}[..])', kind
            return self.cref(node.id), kind
        value, kind = self.expr(node)
        if kind in TEXT:
            return self.as_text(value, kind, node), kind
        return f'(&{value})', kind

    def sorted_call(self, n, arg, reverse):
        rev = 'false'
        if reverse is not None:
            if not (isinstance(reverse, ast.Constant) and type(reverse.value) is bool):
                self.fail(n, 'sorted(..., reverse=...) needs True or False written in the code')
            rev = 'true' if reverse.value else 'false'
        ref, kind = self.materialize(arg)
        if kind in ('array', 'flist', 'series'):
            return f'({{ let mut t = av({ref}).to_vec(); sort_f(&mut t,{rev})?; arr_from(t,0.0f64)? }})', 'flist'
        if kind in ('iarray', 'ilist', 'iseries'):
            return f'({{ let mut t = av({ref}).to_vec(); sort_o(&mut t,{rev}); arr_from(t,0i64)? }})', 'ilist'
        if kind == 'slist':
            return f'({{ let mut t = {ref}.clone(); sort_o(&mut t,{rev}); t }})', 'slist'
        if kind in TEXT:
            return f'({{ let mut t = s_chars({ref}); sort_o(&mut t,{rev}); t }})', 'slist'
        if is_dict(kind) or is_set(kind):
            key = dict_parts(kind)[0] if is_dict(kind) else set_part(kind)
            items = f'{ref}.keys()' if is_dict(kind) else f'{ref}.iter().cloned().collect::<Vec<_>>()'
            if key == 'strv':
                return f'({{ let mut t = {items}; sort_o(&mut t,{rev}); t }})', 'slist'
            return f'({{ let mut t = {items}; sort_o(&mut t,{rev}); arr_from(t,0i64)? }})', 'ilist'
        self.fail(n, f'sorted() of {kind} is not compiled')

    def convert_container(self, n, name, arg):
        ref, kind = self.materialize(arg)
        if name == 'list':
            if kind == 'slist':
                return f'{ref}.clone()', 'slist'
            if kind in TEXT:
                return f's_chars({ref})', 'slist'
            if kind in ('flist', 'ilist', 'array', 'iarray'):
                el = ELEMENT[kind]
                return f'arr_from(av({ref}).to_vec(),{zero(el)}{el})?', 'flist' if el == 'f64' else 'ilist'
            if is_dict(kind):
                key = dict_parts(kind)[0]
                if key == 'strv':
                    return f'{ref}.keys()', 'slist'
                return f'arr_from({ref}.keys(),0i64)?', 'ilist'
            self.fail(n, f'list() of {kind} is not compiled', 'Iterating a set has no reproducible order.')
        if name == 'set':
            if kind == 'slist':
                return f'{ref}.iter().cloned().collect::<HashSet<S>>()', 'set(strv)'
            if kind in TEXT:
                return f's_chars({ref}).into_iter().collect::<HashSet<S>>()', 'set(strv)'
            if kind in ('ilist', 'iarray'):
                return f'av({ref}).iter().cloned().collect::<HashSet<i64>>()', 'set(i64)'
            if is_set(kind):
                return f'{ref}.clone()', kind
            if is_dict(kind):
                key = dict_parts(kind)[0]
                return f'{ref}.keys().into_iter().collect::<HashSet<_>>()', f'set({key})'
            self.fail(n, f'set() of {kind} is not compiled')
        if is_dict(kind):
            return (
                f'({{ let mut t = OMap::new(); let mut i = 0usize; while let Some((k,v)) = {ref}.next(&mut i) {{ t.set(k,v); }} t }})',
                kind,
            )
        self.fail(n, f'dict() of {kind} is not compiled', 'Build the dict with d[key] = value in a loop.')

    def sequence_best(self, n, name, arg):
        ref, kind = self.materialize(arg)
        low = 'true' if name == 'min' else 'false'
        if kind in ('array', 'iarray', 'flist', 'ilist', 'series', 'iseries'):
            return f'seq_max(av({ref}),{low})?', ELEMENT[kind]
        if kind == 'slist':
            return f'seq_max({ref}.as_slice(),{low})?', 'strv'
        if kind in TEXT:
            return f'seq_max({ref},{low})?', 'char'
        self.fail(n, f'{name}() of {kind} is not compiled')

    def arg_best(self, n, name, arg, key):
        """max(d, key=d.get) / min(d, key=d.get): the first key with the best value."""
        dname, kind = self.container_name(arg)
        if not (
            dname
            and is_dict(kind)
            and isinstance(key, ast.Attribute)
            and key.attr == 'get'
            and isinstance(key.value, ast.Name)
            and key.value.id == dname
        ):
            self.fail(n, f'{name}(..., key=...) is compiled only as {name}(d, key=d.get)')
        self.builtin(name)
        k, v = dict_parts(kind)
        if v not in ('i64', 'f64'):
            self.fail(n, f'{name}(d, key=d.get) needs number values')
        better = '<' if name == 'min' else '>'
        ref = self.cref(dname)
        return (
            f'({{ let mut i = 0usize; let mut best: Option<({_rt(k)},{_rt(v)})> = None; '
            f'while let Some((k,v)) = {ref}.next(&mut i) {{ match &best {{ None => best = Some((k,v)), '
            f'Some((_,b)) => if v {better} *b {{ best = Some((k,v)) }} }} }} best.ok_or(6)?.0 }})'
        ), k

    # ------------------------------------------------------------- statements
    def ext_statement_peek(self, n):
        """True when a call statement is a method call on a list/dict/set variable."""
        func = n.value.func
        return isinstance(func, ast.Attribute) and self.container_name(func.value)[0] is not None

    def ext_statement(self, n):
        """Method calls used as statements (lst.append(x), s.add(x), ...) and `del`."""
        if isinstance(n, ast.Delete):
            lines = []
            for target in n.targets:
                if not isinstance(target, ast.Subscript) or isinstance(target.slice, ast.Slice):
                    self.fail(n, 'only del d[key] and del lst[i] are compiled')
                name, kind = self.container_name(target.value)
                if name is None:
                    self.fail(n, 'del is compiled for dict and list variables')
                if is_dict(kind):
                    k, _ = dict_parts(kind)
                    lines.append(f'{self.cmut(name, n)}.del(&{self.item(target.slice, k)})?;')
                elif kind == 'slist':
                    lines.append(f'v_pop({self.cmut(name, n)},{self.expect(target.slice, "i64")})?;')
                elif kind in LIST_KINDS and name not in self.params:
                    lines.append(f'{self.cmut(name, n)}.pop_at({self.expect(target.slice, "i64")})?;')
                else:
                    self.fail(n, f'del is not compiled for {kind}')
            return ' '.join(lines)
        if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)):
            return None
        call = n.value
        attr = call.func.attr
        name, kind = self.container_name(call.func.value)
        if name is None:
            return None
        args = call.args
        if kind in GROWABLE or kind in LIST_KINDS:
            if attr == 'append' and len(args) == 1 and not call.keywords:
                if name in self.params:
                    self.fail(n, 'lists passed in cannot grow in compiled code', 'Build a new list and return it.')
                _, vk = self.expr(args[0])
                el = self.element_kind(vk)
                if kind == 'list?':
                    if el not in ('i64', 'f64', 'strv'):
                        self.fail(n, f'lists of {vk} values are not compiled', 'Use ints, floats or strings.')
                    self.refine(name, {'i64': 'ilist', 'f64': 'flist', 'strv': 'slist'}[el], n)
                    kind = self.types[name]
                want = 'strv' if kind == 'slist' else ELEMENT[kind]
                return f'{self.cmut(name, n)}.push({self.item(args[0], want)});'
            if attr == 'extend' and len(args) == 1 and not call.keywords and kind != 'list?':
                if name in self.params:
                    self.fail(n, 'lists passed in cannot grow in compiled code')
                ref, other = self.materialize(args[0])
                if other != kind:
                    self.fail(n, 'extend() needs a list of the same type')
                if kind == 'slist':
                    return f'{{ let t = {ref}.clone(); {self.cmut(name, n)}.extend(t); }}'
                return f'{{ let t = av({ref}).to_vec(); for x in t {{ {self.cmut(name, n)}.push(x); }} }}'
            if attr == 'sort' and not args and set(kw.arg for kw in call.keywords) <= {'reverse'}:
                reverse = next((kw.value for kw in call.keywords), None)
                rev = 'false'
                if reverse is not None:
                    if not (isinstance(reverse, ast.Constant) and type(reverse.value) is bool):
                        self.fail(n, 'sort(reverse=...) needs True or False written in the code')
                    rev = 'true' if reverse.value else 'false'
                target = self.cmut(name, n)
                if kind == 'slist':
                    return f'sort_o({target}.as_mut_slice(),{rev});'
                if kind == 'flist':
                    return f'sort_f(avm({target}),{rev})?;'
                if kind == 'ilist':
                    return f'sort_o(avm({target}),{rev});'
            if attr == 'reverse' and not args:
                target = self.cmut(name, n)
                return f'{target}.reverse();' if kind == 'slist' else f'avm({target}).reverse();'
            if attr == 'clear' and not args:
                if name in self.params and kind in LIST_KINDS:
                    self.fail(n, 'lists passed in cannot shrink in compiled code')
                target = self.cmut(name, n)
                return f'{target}.clear();' if kind == 'slist' else f'*{target} = Arr::empty();'
            if attr == 'pop':
                return f'let _ = {self.list_pop(call, call.func.value)[0]};'
        if is_set(kind) or kind == 'set?':
            if attr in ('add', 'discard', 'remove') and len(args) == 1 and not call.keywords:
                if kind == 'set?':
                    _, vk = self.expr(args[0])
                    el = self.element_kind(vk)
                    self.check_key(n, el)
                    self.refine(name, f'set({el})', n)
                    kind = self.types[name]
                element = self.item(args[0], set_part(kind))
                target = self.cmut(name, n)
                if attr == 'add':
                    return f'{target}.insert({element});'
                if attr == 'discard':
                    return f'{target}.remove(&{element});'
                return f'set_remove({target},&{element})?;'
            if attr == 'clear' and not args:
                return f'{self.cmut(name, n)}.clear();'
            if attr == 'update' and len(args) == 1 and kind != 'set?':
                ref, other = self.materialize(args[0])
                el = set_part(kind)
                if other == 'slist' and el == 'strv':
                    return f'{{ let t = {ref}.clone(); {self.cmut(name, n)}.extend(t); }}'
                if is_set(other) and other == kind:
                    return f'{{ let t = {ref}.clone(); {self.cmut(name, n)}.extend(t); }}'
                self.fail(n, 'set.update() needs a list of strings or a set of the same type')
        if is_dict(kind) or kind == 'dict?':
            if attr == 'clear' and not args:
                return f'{self.cmut(name, n)}.clear();'
            if attr in ('pop', 'setdefault'):
                return f'let _ = {self.dict_method(call, call.func.value, kind, attr)[0]};'
        if attr in MUTATING_METHODS or attr in ('pop',):
            self.fail(n, f'{kind}.{attr}() is not compiled with these arguments')
        return None

    def ext_store(self, node, target, value_node, op):
        """d[k] = v, d[k] += v, names[i] = s."""
        name, kind = self.container_name(target.value)
        if name is None or isinstance(target.slice, ast.Slice):
            if name is not None:
                self.fail(node, 'assigning to a slice is not compiled')
            return None
        if kind == 'dict?':
            _, kk = self.expr(target.slice)
            _, vk = self.expr(value_node)
            kk, vk = self.element_kind(kk), self.element_kind(vk)
            self.check_key(node, kk)
            if vk not in ('i64', 'f64', 'bool', 'strv'):
                self.fail(node, f'dict values of {vk} are not compiled')
            self.refine(name, f'dict({kk},{vk})', node)
            kind = self.types[name]
        if is_dict(kind):
            k, v = dict_parts(kind)
            t = self.temp('w')
            key = self.item(target.slice, k)
            if op is None:
                value = self.item(value_node, v)
                return f'{{ let {t}k = {key}; let {t}v = {value}; {self.cmut(name, node)}.set({t}k,{t}v); }}'
            current = f'__dictitem{t}'
            self.types[current] = v
            self.names[current] = f'{t}e'
            self.assigned.add(current)
            value, vk = self.binop(node, ast.Name(id=current, ctx=ast.Load()), op, value_node)
            self.assigned.discard(current)
            del self.types[current]
            if vk != v:
                self.fail(node, f'this dict holds {v} values; the result is {vk}', 'Start values at 0.0 for floats.')
            target_ref = self.cmut(name, node)
            return f'{{ let {t}k = {key}; let {t}e = {target_ref}.need(&{t}k)?; let {t}v = {value}; {target_ref}.set({t}k,{t}v); }}'
        if kind == 'slist':
            if op is not None:
                if not isinstance(op, ast.Add):
                    self.fail(node, 'only += is compiled for strings in a list')
                index = self.expect(target.slice, 'i64')
                piece, _ = self.text(value_node)
                t = self.temp('w')
                ref = self.cmut(name, node)
                return f'{{ let {t}i = {index}; let mut {t}s = v_at({ref},{t}i)?; s_push(&mut {t}s,{piece}); v_put({ref},{t}i,{t}s)?; }}'
            t = self.temp('w')
            index = self.expect(target.slice, 'i64')
            value = self.item(value_node, 'strv')
            return f'{{ let {t}i = {index}; let {t}v = {value}; v_put({self.cmut(name, node)},{t}i,{t}v)?; }}'
        return None

    # ------------------------------------------------------------------ loops
    def ext_for(self, n, j):
        """Header for loops over strings, lists of strings and dicts, or None."""
        it = n.iter
        target = n.target
        # for k, v in d.items() / for k in d.keys() / for v in d.values()
        if (
            isinstance(it, ast.Call)
            and isinstance(it.func, ast.Attribute)
            and it.func.attr in ('items', 'keys', 'values')
            and not it.args
        ):
            name, kind = self.container_name(it.func.value)
            if name is not None and is_dict(kind):
                return self.dict_loop(n, name, kind, it.func.attr, j)
        if (
            isinstance(it, ast.Call)
            and isinstance(it.func, ast.Name)
            and it.func.id == 'zip'
            and self.is_builtin('zip')
        ):
            return self.zip_loop(n, j)
        if (
            isinstance(it, ast.Call)
            and isinstance(it.func, ast.Name)
            and it.func.id == 'reversed'
            and self.is_builtin('reversed')
            and len(it.args) == 1
            and isinstance(it.args[0], ast.Call)
            and isinstance(it.args[0].func, ast.Name)
            and it.args[0].func.id == 'range'
            and 1 <= len(it.args[0].args) <= 2
        ):
            self.builtin('reversed')
            r = it.args[0].args
            start, stop = (ast.Constant(value=0), r[0]) if len(r) == 1 else (r[0], r[1])
            one = ast.Constant(value=1)
            n.iter = ast.Call(
                func=it.args[0].func,
                args=[
                    ast.BinOp(left=stop, op=ast.Sub(), right=one),
                    ast.BinOp(left=start, op=ast.Sub(), right=one),
                    ast.UnaryOp(op=ast.USub(), operand=one),
                ],
                keywords=[],
            )
            ast.fix_missing_locations(ast.copy_location(n.iter, it))
            return 'range'
        name, kind = self.container_name(it) if isinstance(it, ast.Name) else (None, None)
        if name is not None and is_dict(kind):
            return self.dict_loop(n, name, kind, 'keys', j)
        if name is not None and (is_set(kind) or kind == 'set?'):
            self.fail(
                n.iter,
                "looping over a set is not compiled: Python's set order cannot be reproduced",
                'Loop over sorted(s) instead.',
            )
        enumerate_call = (
            isinstance(it, ast.Call)
            and isinstance(it.func, ast.Name)
            and it.func.id == 'enumerate'
            and self.is_builtin('enumerate')
            and 1 <= len(it.args) <= 2
        )
        source = it.args[0] if enumerate_call else it
        skind = self.kind_of(source)
        if skind is None and not (
            isinstance(source, ast.Name)
            or (
                isinstance(source, ast.Call)
                and source.func
                and isinstance(source.func, ast.Name)
                and source.func.id in ('range', 'prange', 'enumerate')
            )
        ):
            if isinstance(source, ast.Call) and self.is_prange(source.func):
                return None
            value, skind = self.expr(source)
            if is_set(skind):
                self.fail(
                    n.iter,
                    "looping over a set is not compiled: Python's set order cannot be reproduced",
                    'Loop over sorted(s) instead.',
                )
            if skind not in ('strv', 'slist', 'flist', 'ilist'):
                return None
            holder = f'let it{j} = {value};'
            length = f'it{j}.len()' if skind != 'strv' else f'(&it{j}[..]).len()'
            if skind in ('flist', 'ilist'):
                length = f'it{j}.len'
                get = f'at(&it{j},idx{j} as i64)?'
            else:
                get = f'it{j}[idx{j}].clone()'
        elif skind == 'strv':
            holder = f'let it{j} = {self.name(source.id)}.clone();'
            length = f'it{j}.len()'
            get = f'it{j}[idx{j}]'
        elif skind == 'slist':
            ref = self.cref(source.id)
            holder = ''
            length = f'{ref}.len()'
            get = f'{ref}[idx{j}].clone()'
        else:
            return None
        item_kind = {'strv': 'char', 'slist': 'strv', 'flist': 'f64', 'ilist': 'i64'}[skind]
        if enumerate_call:
            self.builtin('enumerate')
            if not (
                isinstance(target, ast.Tuple)
                and len(target.elts) == 2
                and all(isinstance(e, ast.Name) for e in target.elts)
            ):
                self.fail(target, 'use for i, x in enumerate(items)')
            start = self.expect(it.args[1], 'i64') if len(it.args) == 2 else '0i64'
            index_target = self.assign(n, target.elts[0].id, f'(start{j}).checked_add(idx{j} as i64).ok_or(1)?', 'i64')
            item_target = self.assign(n, target.elts[1].id, get, item_kind)
            holder += f' let start{j}: i64 = {start};'
            targets = f'{index_target}\n{item_target}'
        else:
            if not isinstance(target, ast.Name):
                self.fail(target, 'use one loop variable here')
            targets = self.assign(n, target.id, get, item_kind)
        return f"""{{ {holder} let mut k{j}: usize = 0;
while k{j} < {length} {{
tick(fuel)?;
let idx{j}=k{j}; k{j}+=1;
{targets}"""

    def dict_loop(self, n, name, kind, what, j):
        k, v = dict_parts(kind)
        target = n.target
        ref = self.cref(name)
        if what == 'items':
            if not (
                isinstance(target, ast.Tuple)
                and len(target.elts) == 2
                and all(isinstance(e, ast.Name) for e in target.elts)
            ):
                self.fail(target, 'use for key, value in d.items()')
            targets = (
                self.assign(n, target.elts[0].id, f'e{j}.0.clone()', k)
                + '\n'
                + self.assign(n, target.elts[1].id, f'e{j}.1.clone()', v)
            )
        else:
            if not isinstance(target, ast.Name):
                self.fail(target, f'use one loop variable for d.{what}()')
            targets = self.assign(
                n, target.id, f'e{j}.{0 if what == "keys" else 1}.clone()', k if what == 'keys' else v
            )
        # Python checks for size changes on every step, including the last one.
        return f"""{{ let ver{j} = {ref}.ver; let mut k{j}: usize = 0;
loop {{
if {ref}.ver != ver{j} {{ return Err(6) }}
let e{j} = match {ref}.next(&mut k{j}) {{ Some(e) => e, None => break }};
tick(fuel)?;
{targets}"""

    def zip_loop(self, n, j):
        it = n.iter
        target = n.target
        if it.keywords or len(it.args) < 2 or not (isinstance(target, ast.Tuple) and len(target.elts) == len(it.args)):
            self.fail(it, 'use for a, b in zip(xs, ys) with one name per sequence')
        self.builtin('zip')
        holders, lengths, assigns = [], [], []
        for k, (source, tname) in enumerate(zip(it.args, target.elts)):
            if not isinstance(tname, ast.Name):
                self.fail(target, 'zip loop variables must be plain names')
            kind = self.kind_of(source)
            if kind is None:
                value, kind = self.expr(source)
                holders.append(f'let z{j}_{k} = {value};')
                ref = f'(&z{j}_{k})'
            elif kind == 'strv':
                holders.append(f'let z{j}_{k} = {self.name(source.id)}.clone();')
                ref = f'(&z{j}_{k})'
            elif kind in ARRAY_INFO:
                ref = self.aref(source.id)
            else:
                ref = self.cref(source.id)
            if kind in ARRAY_INFO:
                if ARRAY_INFO[kind][1] != 1:
                    self.fail(source, 'zip over 2-D arrays is not compiled')
                lengths.append(f'({ref}).len')
                value, vk = self.read(f'at({ref},idx{j} as i64)?', kind)
            elif kind == 'strv':
                lengths.append(f'{ref}.len()')
                value, vk = f'{ref}[idx{j}]', 'char'
            elif kind == 'slist':
                lengths.append(f'{ref}.len()')
                value, vk = f'{ref}[idx{j}].clone()', 'strv'
            else:
                self.fail(source, f'zip over {kind} is not compiled')
            assigns.append((tname.id, value, vk))
        targets = '\n'.join(self.assign(n, name, value, vk) for name, value, vk in assigns)
        condition = ' && '.join(f'k{j} < {length}' for length in lengths)
        return f"""{{ {' '.join(holders)} let mut k{j}: usize = 0;
while {condition} {{
tick(fuel)?;
let idx{j}=k{j}; k{j}+=1;
{targets}"""

    # ------------------------------------------------------------- conditions
    def ext_membership(self, n, negate):
        left, right = n.left, n.comparators[0]
        rkind = self.kind_of(right)
        if rkind is None and not isinstance(right, ast.Constant):
            _, rkind = self.expr(right)
            if not (is_container(rkind) or rkind in ('strv',)):
                return None
            ref = None
        else:
            ref = self.cref(right.id) if isinstance(right, ast.Name) and rkind not in ('str', 'strv') else None
        if rkind in ('str', 'strv') or (isinstance(right, ast.Constant) and type(right.value) is str):
            hay, _ = self.text(right)
            if self.expr(left)[1] not in TEXT:
                self.fail(
                    n,
                    '`in` on a string needs a string or one character on the left, like ch in "aeiou"',
                    'For numbers, compare with == in a loop.',
                )
            needle, _ = self.text(left)
            test = f's_has({hay},{needle})'
        elif rkind == 'slist':
            needle, _ = self.text(left)
            source = ref or f'(&{self.expr(right)[0]})'
            test = f'{source}.iter().any(|e| &e[..] == {needle})'
        elif is_dict(rkind):
            k, _ = dict_parts(rkind)
            source = ref or f'(&{self.expr(right)[0]})'
            test = f'{source}.has(&{self.item(left, k)})'
        elif is_set(rkind):
            source = ref or f'(&{self.expr(right)[0]})'
            test = f'{source}.contains(&{self.item(left, set_part(rkind))})'
        elif rkind in ('dict?', 'set?', 'list?'):
            test = 'false'
        elif rkind in LIST_KINDS:
            value, vk = self.expr(left)
            el = ELEMENT[rkind]
            if vk != el:
                self.fail(n, '`in` on a list of numbers needs a value of the same type')
            t = self.temp('m')
            guard = f'if {t}.is_nan() {{ return Err(6) }}' if el == 'f64' else ''
            test = f'({{ let {t} = {value}; {guard} av({self.cref(right.id)}).contains(&{t}) }})'
        else:
            return None
        return f'(!({test}))' if negate else f'({test})'

    def ext_truth(self, value, kind, node):
        """Truth value of strings and containers: non-empty."""
        if kind == 'strv':
            return f'(!(&({value})[..]).is_empty())'
        if kind in ('slist',) or is_set(kind) or kind in PLACEHOLDERS:
            return f'(({value}).len() != 0)'
        if is_dict(kind):
            return f'(({value}).len() != 0)'
        if kind in LIST_KINDS:
            return f'(({value}).len != 0)'
        return None


def _rt(kind):
    from .kinds import RUST_TYPE

    return RUST_TYPE[kind]
