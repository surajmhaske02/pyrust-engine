"""Conservative typed Python AST to Rust. Unsupported semantics fall back.

Every native operation either produces exactly CPython's result or returns a
non-zero status, in which case the runtime replays the call in Python.
"""
import ast
import builtins
import inspect
import math
import textwrap
import types

class Unsupported(ValueError):
    pass

class F64Array:
    """Annotation for a contiguous, aligned, native-endian 1-D NumPy float64 array.

    Read elements with float(array[index]); write them with array[index] = value.
    Writes go to a private copy that is published only when the native call succeeds,
    so a replay in Python always starts from the original data.
    """

class I64Array:
    """Annotation for a contiguous, aligned, native-endian 1-D NumPy int64 array.

    Read elements with int(array[index]) or float(array[index]); write int values with
    array[index] = value.
    """

ARRAY_KINDS = ('array', 'iarray')
ELEMENT = {'array': 'f64', 'iarray': 'i64'}
MAX_DEPTH = 300  # native call depth; deeper recursion replays in Python (keeps thread stacks safe)

# Status codes shared with runtime._STATUS.
#   1 integer overflow          2 zero divisor / zero range step
#   3 array index out of bounds 4 iteration budget exceeded
#   5 no scalar return          6 value outside the native fast path (math domain,
#                                  float overflow, negative exponent, ...)
HELPERS = r'''
fn floor_div(a:i64,b:i64)->Result<i64,i32>{
    if b==0 {return Err(2)}
    let a=a as i128; let b=b as i128;
    let q=a/b; let r=a%b;
    let q=if r!=0 && ((r<0)!=(b<0)) {q-1} else {q};
    i64::try_from(q).map_err(|_|1)
}
fn modulo(a:i64,b:i64)->Result<i64,i32>{
    if b==0 {return Err(2)}
    let a=a as i128; let b=b as i128; let r=a%b;
    Ok((if r!=0 && ((r<0)!=(b<0)) {r+b} else {r}) as i64)
}
fn divide(a:f64,b:f64)->Result<f64,i32>{if b==0.0 {Err(2)} else {Ok(a/b)}}
fn int_true_div(a:i64,b:i64)->Result<f64,i32>{
    // CPython divides exactly-representable ints as floats; larger ones need
    // its correctly rounded long division, so those calls replay.
    if b==0 {return Err(2)}
    const LIM:i64=1i64<<53;
    if a< -LIM || a>LIM || b< -LIM || b>LIM {return Err(6)}
    Ok((a as f64)/(b as f64))
}
fn float_floor_div(a:f64,b:f64)->Result<f64,i32>{
    // CPython _float_div_mod
    if b==0.0 {return Err(2)}
    let m=a%b;
    let mut d=(a-m)/b;
    if m!=0.0 && ((b<0.0)!=(m<0.0)) {d-=1.0;}
    if d!=0.0 {
        let f=d.floor();
        Ok(if d-f>0.5 {f+1.0} else {f})
    } else {
        Ok((0.0f64).copysign(a/b))
    }
}
fn float_mod(a:f64,b:f64)->Result<f64,i32>{
    // CPython float_rem
    if b==0.0 {return Err(2)}
    let mut m=a%b;
    if m!=0.0 {
        if (b<0.0)!=(m<0.0) {m+=b;}
    } else {
        m=(0.0f64).copysign(b);
    }
    Ok(m)
}
fn int_pow(a:i64,b:i64)->Result<i64,i32>{
    if b<0 {return Err(6)}
    if b>u32::MAX as i64 {
        return match a {0=>Ok(0),1=>Ok(1),-1=>Ok(if b%2==0 {1} else {-1}),_=>Err(1)};
    }
    a.checked_pow(b as u32).ok_or(1)
}
fn float_pow(a:f64,b:f64)->Result<f64,i32>{
    // CPython float_pow: special cases first, then C pow with error checks.
    if b==0.0 {return Ok(1.0)}
    if a.is_nan() {return Ok(a)}
    if b.is_nan() {return Ok(if a==1.0 {1.0} else {b})}
    if a.is_infinite() || b.is_infinite() {return Ok(a.powf(b))}
    if a==0.0 && b<0.0 {return Err(2)}
    if a<0.0 && b!=b.floor() {return Err(6)}
    let r=a.powf(b);
    if r.is_infinite() {return Err(6)}
    Ok(r)
}
fn shift_left(a:i64,b:i64)->Result<i64,i32>{
    if b<0 {return Err(6)}
    if a==0 {return Ok(0)}
    if b>=64 {return Err(1)}
    i64::try_from((a as i128)<<b).map_err(|_|1)
}
fn shift_right(a:i64,b:i64)->Result<i64,i32>{
    if b<0 {return Err(6)}
    Ok(if b>=64 {if a<0 {-1} else {0}} else {a>>b})
}
fn max_i(a:i64,b:i64)->i64{if b>a {b} else {a}}
fn min_i(a:i64,b:i64)->i64{if b<a {b} else {a}}
fn max_f(a:f64,b:f64)->f64{if b>a {b} else {a}}
fn min_f(a:f64,b:f64)->f64{if b<a {b} else {a}}
fn to_int(x:f64)->Result<i64,i32>{
    if !x.is_finite() {return Err(6)}
    let t=x.trunc();
    if t>=-9223372036854775808.0 && t<9223372036854775808.0 {Ok(t as i64)} else {Err(1)}
}
fn math1(x:f64,r:f64)->Result<f64,i32>{
    // CPython math_1: NaN from a non-NaN input is a domain error, and an
    // infinite result from a finite input is a domain or overflow error.
    if r.is_nan() && !x.is_nan() {return Err(6)}
    if r.is_infinite() && x.is_finite() {return Err(6)}
    Ok(r)
}
fn index(len:usize,i:i64)->Result<usize,i32>{
    let j=if i<0 {(len as i128)+(i as i128)} else {i as i128};
    if j<0 || j>=len as i128 {Err(3)} else {Ok(j as usize)}
}
fn at<T:Copy>(a:&[T],i:i64)->Result<T,i32>{Ok(a[index(a.len(),i)?])}
fn put<T:Copy>(a:&mut [T],i:i64,v:T)->Result<(),i32>{let j=index(a.len(),i)?; a[j]=v; Ok(())}
fn cmp_if(i:i64,f:f64)->i8{
    // Exact int-vs-float comparison like CPython: -1 less, 0 equal, 1 greater, 2 NaN.
    if f.is_nan() {return 2}
    if f>=9223372036854775808.0 {return -1}
    if f< -9223372036854775808.0 {return 1}
    let t=f.trunc(); let ti=t as i64;
    if i<ti {-1} else if i>ti {1} else {let r=f-t; if r>0.0 {-1} else if r<0.0 {1} else {0}}
}
fn tick(fuel:&mut u64)->Result<(),i32>{if *fuel==0 {Err(4)} else {*fuel-=1;Ok(())}}
'''

# math functions with one float argument and a float result (CPython math_1 rules).
MATH_FLOAT = {'sqrt':'sqrt','exp':'exp','log':'ln','log2':'log2','log10':'log10','sin':'sin',
              'cos':'cos','tan':'tan','asin':'asin','acos':'acos','atan':'atan',
              'sinh':'sinh','cosh':'cosh','tanh':'tanh'}
MATH_CONSTANTS = {'pi','e','tau','inf','nan'}
# Builtins the compiler understands, checked again at call time in case they are rebound.
KNOWN_BUILTINS = {'range','len','float','int','abs','min','max','pow'}


def float_literal(value):
    if math.isnan(value):
        return '(f64::NAN)'
    if math.isinf(value):
        return '(f64::INFINITY)' if value > 0 else '(f64::NEG_INFINITY)'
    return f'({value!r}f64)'


def _resolve_callee(obj):
    """An @optimize wrapper or a plain function; returns the original function or None."""
    original = getattr(obj, 'python', None)
    if isinstance(original, types.FunctionType) and callable(getattr(obj, 'report', None)):
        return original
    if isinstance(obj, types.FunctionType):
        return obj
    return None


class Compiler:
    def __init__(self, function, max_iterations=10_000_000, _program=None):
        if hasattr(function, '__wrapped__'):
            raise Unsupported('Wrapped functions are not supported')
        self.function = function
        self.first_line = function.__code__.co_firstlineno
        self.tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        self.node = self.tree.body[0]
        if not isinstance(self.node, ast.FunctionDef) or function.__closure__:
            raise Unsupported('Only ordinary functions without closure capture are supported')
        self.signature = inspect.signature(function)
        self.types = {}
        for name, p in self.signature.parameters.items():
            if p.kind not in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
                raise Unsupported(f'{function.__name__}: *args, **kwargs and keyword-only parameters are unsupported')
            annotation = p.annotation
            if annotation in (inspect.Parameter.empty, int, 'int'):
                kind = 'i64'
            elif annotation in (float, 'float'):
                kind = 'f64'
            elif annotation in (F64Array, 'F64Array'):
                kind = 'array'
            elif annotation in (I64Array, 'I64Array'):
                kind = 'iarray'
            else:
                raise Unsupported(f'{function.__name__}: unsupported annotation for {name}: {annotation!r} '
                                  '(use int, float, F64Array or I64Array)')
            self.types[name] = kind
        self.globals = function.__globals__
        ns = function.__builtins__
        self.builtins_ns = ns if isinstance(ns, dict) else vars(ns)
        self.names = {p: f'v{i}' for i,p in enumerate(self.types)}
        self.assigned = set(self.names)
        self.local_names = set(self.names) | {n.id for n in ast.walk(self.node)
                                             if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        targets = []
        for node in ast.walk(self.node):
            if isinstance(node, ast.Assign):
                targets += node.targets
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets.append(node.target)
        self.written = {t.value.id for t in targets if isinstance(t, ast.Subscript)
                        and isinstance(t.value, ast.Name) and self.types.get(t.value.id) in ARRAY_KINDS}
        returns = self.signature.return_annotation
        self.declared_return = {int:'i64','int':'i64',float:'f64','float':'f64',None:'none','None':'none'}.get(
            returns if isinstance(returns, (str, type)) or returns is None else object())
        self.used_builtins = set()
        self.counter = 0
        self.loop_depth = 0
        self.return_type = None
        self.max_iterations = max_iterations
        # Shared by every function compiled into one library.
        self.program = _program if _program is not None else {
            'functions': [], 'index': {}, 'stack': [], 'guards': {}, 'code_guards': {}, 'callees': [],
            'recursive': {}}

    # ------------------------------------------------------------ diagnostics
    def fail(self, node, message):
        line = self.first_line + getattr(node, 'lineno', 1) - 1
        try:
            snippet = ast.unparse(node)
        except Exception:
            snippet = type(node).__name__
        if len(snippet) > 60:
            snippet = snippet[:57] + '...'
        raise Unsupported(f'{self.function.__name__}, line {line}: {message}: `{snippet}`')

    # ---------------------------------------------------------------- helpers
    def name(self, name):
        if name not in self.names:
            self.names[name] = f'v{len(self.names)}'
        return self.names[name]

    def guard(self, name, expected):
        """Record that global/builtin `name` must still be `expected` at call time."""
        key = (id(self.globals), name)
        self.program['guards'][key] = (self.globals, self.builtins_ns, name, expected)

    def lookup(self, name):
        """Current global/builtin value of a non-local name (or a sentinel)."""
        return self.globals.get(name, self.builtins_ns.get(name, _MISSING))

    def builtin(self, name):
        if name in self.local_names:
            raise Unsupported(f'{self.function.__name__}: local {name} shadowing is unsupported')
        self.used_builtins.add(name)
        self.guard(name, getattr(builtins, name))

    def is_builtin(self, name):
        return (name in KNOWN_BUILTINS and name not in self.local_names
                and self.lookup(name) is getattr(builtins, name))

    def math_name(self, func):
        """'sqrt' for math.sqrt(...) or a `from math import sqrt` name; else None."""
        if (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                and func.value.id not in self.local_names and self.lookup(func.value.id) is math):
            self.guard(func.value.id, math)
            return func.attr
        if isinstance(func, ast.Name) and func.id not in self.local_names:
            obj = self.lookup(func.id)
            for attr in (*MATH_FLOAT, 'floor', 'ceil', 'fabs', 'isnan', 'isinf', 'isfinite', 'trunc'):
                if obj is getattr(math, attr):
                    self.guard(func.id, obj)
                    return attr
        return None

    def expect(self, node, kind):
        value, actual = self.expr(node)
        if actual != kind:
            self.fail(node, f'expected {kind}, got {actual}')
        return value

    def scalar(self, node):
        value, kind = self.expr(node)
        if kind not in ('i64', 'f64'):
            self.fail(node, 'expected an int or float value')
        return value, kind

    def array(self, node):
        value, kind = self.expr(node)
        if kind not in ARRAY_KINDS:
            self.fail(node, 'expected an F64Array or I64Array parameter')
        return value, kind

    def element(self, sub):
        """Read array[index] -> (rust, element kind)."""
        a, kind = self.array(sub.value)
        i = self.expect(sub.slice, 'i64')
        return f'at(&*{a},{i})?', ELEMENT[kind]

    def as_float(self, node):
        value, kind = self.scalar(node)
        return (f'({value} as f64)' if kind == 'i64' else value)

    # ------------------------------------------------------------ expressions
    def expr(self, n):
        if isinstance(n, ast.Constant):
            if type(n.value) is int and -(2**63) <= n.value < 2**63:
                return f'({n.value}i64)', 'i64'
            if type(n.value) is float and math.isfinite(n.value):
                return float_literal(n.value), 'f64'
            self.fail(n, 'only int and float constants are supported')
        if isinstance(n, ast.Name):
            if n.id in self.assigned:
                return self.name(n.id), self.types[n.id]
            if n.id not in self.local_names:
                return self.global_constant(n)
            self.fail(n, 'variable may be used before it is assigned')
        if isinstance(n, ast.Attribute):
            if (isinstance(n.value, ast.Name) and n.value.id not in self.local_names
                    and self.lookup(n.value.id) is math and n.attr in MATH_CONSTANTS):
                self.guard(n.value.id, math)
                return float_literal(getattr(math, n.attr)), 'f64'
            self.fail(n, 'unsupported attribute')
        if isinstance(n, ast.UnaryOp):
            if isinstance(n.op, ast.USub):
                # AST represents -2**63 as unary minus applied to the positive literal.
                if isinstance(n.operand, ast.Constant) and type(n.operand.value) is int and n.operand.value == 2**63:
                    return 'i64::MIN', 'i64'
                value, kind = self.scalar(n.operand)
                if kind == 'i64':
                    return f'({value}).checked_neg().ok_or(1)?', kind
                return f'(-({value}))', kind
            if isinstance(n.op, ast.UAdd):
                value, kind = self.scalar(n.operand)
                return f'({value})', kind
            if isinstance(n.op, ast.Invert):
                return f'(!({self.expect(n.operand, "i64")}))', 'i64'
        if isinstance(n, ast.BinOp):
            return self.binop(n, n.left, n.op, n.right)
        if isinstance(n, ast.IfExp):
            cond = self.condition(n.test)
            a, ta = self.expr(n.body)
            b, tb = self.expr(n.orelse)
            if ta != tb or ta in ARRAY_KINDS:
                self.fail(n, 'both branches of `x if c else y` must have the same scalar type')
            return f'(if {cond} {{ {a} }} else {{ {b} }})', ta
        if isinstance(n, ast.Call):
            if n.keywords:
                self.fail(n, 'keyword arguments in calls are unsupported')
            return self.call(n)
        self.fail(n, 'unsupported expression')

    def global_constant(self, n):
        value = self.lookup(n.id)
        if type(value) is int and -(2**63) <= value < 2**63:
            self.guard(n.id, value)
            return f'({value}i64)', 'i64'
        if type(value) is float:
            self.guard(n.id, value)
            return float_literal(value), 'f64'
        if value is _MISSING:
            self.fail(n, f'name {n.id!r} is not defined')
        self.fail(n, f'global {n.id!r} must be an int or float constant')

    def binop(self, n, left, op, right):
        a, ta = self.expr(left)
        b, tb = self.expr(right)
        if ta in ARRAY_KINDS or tb in ARRAY_KINDS:
            self.fail(n, 'read array elements with float(a[i]) or int(a[i])')
        if ta == tb == 'i64':
            checked = {ast.Add:'checked_add',ast.Sub:'checked_sub',ast.Mult:'checked_mul'}.get(type(op))
            if checked:
                return f'({a}).{checked}({b}).ok_or(1)?', 'i64'
            helper = {ast.FloorDiv:'floor_div',ast.Mod:'modulo',ast.Pow:'int_pow',
                      ast.LShift:'shift_left',ast.RShift:'shift_right'}.get(type(op))
            if helper:
                return f'{helper}({a},{b})?', 'i64'
            bitwise = {ast.BitAnd:'&',ast.BitOr:'|',ast.BitXor:'^'}.get(type(op))
            if bitwise:
                return f'(({a}) {bitwise} ({b}))', 'i64'
            if isinstance(op, ast.Div):
                return f'int_true_div({a},{b})?', 'f64'
        else:
            a = f'({a} as f64)' if ta == 'i64' else a
            b = f'({b} as f64)' if tb == 'i64' else b
            symbol = {ast.Add:'+',ast.Sub:'-',ast.Mult:'*'}.get(type(op))
            if symbol:
                return f'(({a}) {symbol} ({b}))', 'f64'
            helper = {ast.Div:'divide',ast.FloorDiv:'float_floor_div',ast.Mod:'float_mod',
                      ast.Pow:'float_pow'}.get(type(op))
            if helper:
                return f'{helper}({a},{b})?', 'f64'
        self.fail(n, f'unsupported operator for {ta} and {tb}')

    def call(self, n):
        func, args = n.func, n.args
        math_attr = self.math_name(func)
        if math_attr is not None:
            if len(args) != 1:
                self.fail(n, f'math.{math_attr} takes one argument here')
            if math_attr in MATH_FLOAT:
                x = self.as_float(args[0])
                self.counter += 1
                t = f'm{self.counter}'
                return f'({{ let {t}: f64 = {x}; math1({t},{t}.{MATH_FLOAT[math_attr]}())? }})', 'f64'
            if math_attr == 'fabs':
                return f'(({self.as_float(args[0])}).abs())', 'f64'
            if math_attr in ('floor', 'ceil', 'trunc'):
                value, kind = self.scalar(args[0])
                if kind == 'i64':
                    return f'({value})', 'i64'
                return f'to_int(({value}).{math_attr}())?', 'i64'
            self.fail(n, f'math.{math_attr} returns a bool; use it in a condition')
        if not isinstance(func, ast.Name):
            self.fail(n, 'only calls to builtins, math functions and plain functions are supported')
        name = func.id
        if name in self.local_names:
            self.fail(n, f'{name!r} is a local variable, not a function')
        if self.is_builtin(name):
            return self.builtin_call(n, name, args)
        callee = _resolve_callee(self.lookup(name))
        if callee is not None:
            return self.user_call(n, name, callee, args)
        self.fail(n, f'call to {name!r} is unsupported')

    def builtin_call(self, n, name, args):
        self.builtin(name)
        if name == 'len' and len(args) == 1:
            a, _ = self.array(args[0])
            return f'({a}.len() as i64)', 'i64'
        if name == 'float' and len(args) == 1:
            arg = args[0]
            if isinstance(arg, ast.Subscript):
                value, kind = self.element(arg)
                return (value if kind == 'f64' else f'({value} as f64)'), 'f64'
            return self.as_float(arg), 'f64'
        if name == 'int' and len(args) == 1:
            if isinstance(args[0], ast.Subscript):
                value, kind = self.element(args[0])
                return (value, 'i64') if kind == 'i64' else (f'to_int({value})?', 'i64')
            value, kind = self.scalar(args[0])
            return (f'({value})', 'i64') if kind == 'i64' else (f'to_int({value})?', 'i64')
        if name == 'abs' and len(args) == 1:
            value, kind = self.scalar(args[0])
            if kind == 'i64':
                return f'({value}).checked_abs().ok_or(1)?', 'i64'
            return f'(({value}).abs())', 'f64'
        if name in ('min', 'max') and len(args) >= 2:
            values = [self.scalar(a) for a in args]
            kinds = {k for _, k in values}
            if len(kinds) != 1:
                # Python returns whichever argument wins, keeping its own type.
                self.fail(n, f'{name}() arguments must all be int or all be float')
            kind = kinds.pop()
            helper = f'{name}_{"i" if kind == "i64" else "f"}'
            out = values[0][0]
            for value, _ in values[1:]:
                out = f'{helper}({out},{value})'
            return out, kind
        if name == 'pow' and len(args) == 2:
            return self.binop(n, args[0], ast.Pow(), args[1])
        self.fail(n, f'unsupported use of {name}()')

    def register(self, rust_name):
        """Make this function callable (by recursive calls) before its body is compiled."""
        params = [self.types[p] for p in self.signature.parameters]
        self.program['index'][id(self.function)] = (self.function, rust_name, params, self.declared_return)

    def user_call(self, n, name, callee, args):
        program = self.program
        key = id(callee)
        if callee in program['stack']:
            entry = program['index'].get(key)
            if entry is None or entry[3] in (None, 'none'):
                self.fail(n, f'recursive call to {name!r}: annotate its return type (-> int or -> float)')
            program['recursive'][key] = entry[3]
        elif key not in program['index']:
            program['stack'].append(callee)
            try:
                try:
                    sub = Compiler(callee, self.max_iterations, _program=program)
                except Unsupported as exc:
                    raise Unsupported(f'called function {name!r}: {exc}') from None
                except (OSError, TypeError) as exc:
                    raise Unsupported(f'called function {name!r}: source unavailable ({exc})') from None
                if sub.written:
                    self.fail(n, f'called function {name!r} writes to an array; '
                                 'only the decorated function itself can write arrays')
                rust_name = f'k{program.setdefault("next", 0)}'
                program['next'] += 1
                sub.register(rust_name)
                text = sub.function_text(rust_name)
            finally:
                program['stack'].pop()
            if sub.return_type == 'none':
                self.fail(n, f'called function {name!r} does not return a value')
            params = [sub.types[p] for p in sub.signature.parameters]
            program['index'][key] = (callee, rust_name, params, sub.return_type)
            program['functions'].append(text)
            program['callees'].append(callee.__qualname__)
            program['code_guards'][key] = (callee, callee.__code__)
        _, rust_name, params, return_type = program['index'][key]
        self.guard(name, self.lookup(name))
        if len(args) != len(params):
            self.fail(n, f'{name}() needs exactly {len(params)} positional arguments here')
        values = []
        for arg, kind in zip(args, params):
            value, actual = self.expr(arg)
            if actual != kind:
                self.fail(arg, f'argument to {name}() must be {kind}, got {actual}')
            values.append(f'&*{value}' if kind in ARRAY_KINDS else value)
        # Each call spends one unit of the iteration budget, so runaway recursion stops too.
        return f'({{ tick(fuel)?; {rust_name}({", ".join(values + ["fuel", "depth+1"])})? }})', return_type

    # ------------------------------------------------------------- conditions
    def condition(self, n):
        if isinstance(n, ast.Constant) and type(n.value) is bool:
            return 'true' if n.value else 'false'
        if isinstance(n, ast.BoolOp) and isinstance(n.op, (ast.And, ast.Or)):
            op = ' && ' if isinstance(n.op, ast.And) else ' || '
            return '(' + op.join(f'({self.condition(v)})' for v in n.values) + ')'
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            return f'!({self.condition(n.operand)})'
        if isinstance(n, ast.Call) and not n.keywords and len(n.args) == 1:
            attr = self.math_name(n.func)
            if attr in ('isnan', 'isinf', 'isfinite'):
                x = self.as_float(n.args[0])
                method = {'isnan':'is_nan','isinf':'is_infinite','isfinite':'is_finite'}[attr]
                return f'(({x}).{method}())'
        ops = {ast.Lt:'<',ast.LtE:'<=',ast.Gt:'>',ast.GtE:'>=',ast.Eq:'==',ast.NotEq:'!='}
        if isinstance(n, ast.Compare) and all(type(op) in ops for op in n.ops):
            # Chained comparisons: operands are pure, so evaluating one that Python
            # would have short-circuited can at most cause a guard failure and replay.
            operands = [self.compare_operand(x) for x in [n.left, *n.comparators]]
            parts = []
            for (a, ta, ca), op, (b, tb, cb) in zip(operands, n.ops, operands[1:]):
                if ta != tb:
                    # int literal vs float: exact when |literal| <= 2**53.
                    if ta == 'i64' and tb == 'f64' and ca is not None:
                        a, ta = f'({float(ca)!r}f64)', 'f64'
                    elif tb == 'i64' and ta == 'f64' and cb is not None:
                        b, tb = f'({float(cb)!r}f64)', 'f64'
                if {ta, tb} == {'i64', 'f64'}:
                    # Variables of different types: compare exactly, as CPython does.
                    c = f'cmp_if({a},{b})' if ta == 'i64' else f'cmp_if({b},{a})'
                    flip = {ast.Lt:ast.Gt,ast.LtE:ast.GtE,ast.Gt:ast.Lt,ast.GtE:ast.LtE}
                    kind = type(op) if ta == 'i64' else flip.get(type(op), type(op))
                    test = {ast.Lt:'c==-1',ast.LtE:'c==-1 || c==0',ast.Gt:'c==1',ast.GtE:'c==1 || c==0',
                            ast.Eq:'c==0',ast.NotEq:'c!=0'}[kind]
                    parts.append(f'({{ let c={c}; {test} }})')
                    continue
                if ta != tb or ta in ARRAY_KINDS:
                    self.fail(n, 'comparison operands must be int or float values')
                parts.append(f'(({a}) {ops[type(op)]} ({b}))')
            return '(' + ' && '.join(parts) + ')'
        try:
            value, kind = self.expr(n)
        except Unsupported:
            self.fail(n, 'unsupported condition')
        if kind == 'i64':
            return f'(({value}) != 0i64)'
        if kind == 'f64':
            return f'(({value}) != 0.0f64)'  # NaN is truthy, -0.0 is falsy, as in Python
        self.fail(n, 'arrays cannot be used as conditions')

    def compare_operand(self, node):
        """Return (rust, kind, exact_int_literal_or_None)."""
        literal = None
        if isinstance(node, ast.Constant) and type(node.value) is int:
            literal = node.value
        elif (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
              and isinstance(node.operand, ast.Constant) and type(node.operand.value) is int):
            literal = -node.operand.value
        if literal is not None and abs(literal) > 2**53:
            literal = None
        value, kind = self.expr(node)
        return value, kind, literal

    # ------------------------------------------------------------- statements
    def store(self, node, target, value_node, op=None):
        """array[index] = value (or op=) -> Rust statement."""
        if not (isinstance(target.value, ast.Name) and target.value.id in self.written):
            self.fail(node, 'only F64Array / I64Array parameters can be written with a[i] = value')
        a, kind = self.array(target.value)
        element = ELEMENT[kind]
        i = self.expect(target.slice, 'i64')
        self.counter += 1
        j = self.counter
        if op is None:
            value, vkind = self.scalar(value_node)
        else:
            if not isinstance(op, (ast.Add, ast.Sub, ast.Mult)) and not (element == 'f64' and isinstance(op, ast.Div)):
                self.fail(node, 'array elements support +=, -=, *= (and /= for F64Array)')
            current = ast.Name(id=f'__element{j}', ctx=ast.Load())
            self.types[current.id] = element
            self.names[current.id] = f'e{j}'
            self.assigned.add(current.id)
            value, vkind = self.binop(node, current, op, value_node)
            self.assigned.discard(current.id)
            del self.types[current.id]
        if element == 'f64' and vkind == 'i64':
            value, vkind = f'({value} as f64)', 'f64'
        if vkind != element:
            self.fail(node, 'I64Array elements can only be set to int values')
        read = f'let e{j}: {element} = at(&*{a},i{j})?; ' if op is not None else ''
        return f'{{ let i{j}: i64 = {i}; {read}let w{j}: {element} = {value}; put(&mut *{a},i{j},w{j})?; }}'

    def assign(self, node, name, value, kind):
        if kind in ARRAY_KINDS:
            self.fail(node, 'arrays cannot be assigned to other variables')
        if name in self.types and self.types[name] != kind:
            self.fail(node, f'{name!r} changes type from {self.types[name]} to {kind}; '
                            'start float variables at 0.0, not 0')
        self.types[name] = kind
        self.assigned.add(name)
        return f'{self.name(name)} = {value};'

    def block(self, nodes):
        """Compile statements. Sets self.exits to True when control can never fall
        off the end of this block (return/break/continue on every path)."""
        lines = []
        exits = False
        for n in nodes:
            if exits:
                break  # unreachable in Python too
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value,str):
                continue
            if isinstance(n, ast.Pass):
                continue
            if isinstance(n, ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Subscript):
                lines.append(self.store(n, n.targets[0], n.value))
            elif isinstance(n, ast.AugAssign) and isinstance(n.target,ast.Subscript):
                lines.append(self.store(n, n.target, n.value, n.op))
            elif isinstance(n, ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
                value, kind = self.expr(n.value)
                lines.append(self.assign(n, n.targets[0].id, value, kind))
            elif isinstance(n, ast.AnnAssign) and isinstance(n.target,ast.Name) and n.value is not None:
                value, kind = self.expr(n.value)
                lines.append(self.assign(n, n.target.id, value, kind))
            elif isinstance(n, ast.AugAssign) and isinstance(n.target,ast.Name):
                value, kind = self.binop(n, ast.Name(id=n.target.id, ctx=ast.Load()), n.op, n.value)
                lines.append(self.assign(n, n.target.id, value, kind))
            elif isinstance(n, ast.Return) and n.value is not None and not (
                    isinstance(n.value, ast.Constant) and n.value.value is None):
                value, kind = self.expr(n.value)
                if kind in ARRAY_KINDS or (self.return_type and kind != self.return_type):
                    self.fail(n, 'every return must have the same scalar type (int or float)')
                self.return_type = kind
                lines.append(f'return Ok({value});')
                exits = True
            elif isinstance(n, ast.Return):
                if self.return_type not in (None, 'none'):
                    self.fail(n, 'a function cannot both return a value and return None')
                self.return_type = 'none'
                lines.append('return Ok(());')
                exits = True
            elif isinstance(n, ast.If):
                cond = self.condition(n.test)
                before = self.assigned.copy()
                yes = self.block(n.body)
                yes_exits = self.exits
                yes_assigned = self.assigned.copy()
                self.assigned = before.copy()
                no = self.block(n.orelse)
                no_exits = self.exits
                no_assigned = self.assigned
                # A branch that never falls through does not constrain what is
                # definitely assigned after the if statement.
                if yes_exits and not no_exits:
                    self.assigned = no_assigned
                elif no_exits and not yes_exits:
                    self.assigned = yes_assigned
                else:
                    self.assigned = yes_assigned & no_assigned
                exits = yes_exits and no_exits
                lines.append(f'if {cond} {{\n{yes}\n}} else {{\n{no}\n}}')
            elif isinstance(n, (ast.For,ast.While)) and not n.orelse:
                before = self.assigned.copy()
                self.counter += 1
                j = self.counter
                if isinstance(n,ast.For):
                    call = n.iter
                    if not (isinstance(n.target,ast.Name) and isinstance(call,ast.Call)
                            and isinstance(call.func,ast.Name) and call.func.id=='range'
                            and 1 <= len(call.args) <= 3 and not call.keywords):
                        self.fail(n.iter, 'for loops must use range(...)')
                    self.builtin('range')
                    args = [self.expect(a,'i64') for a in call.args]
                    start, stop, step = ('0i64',args[0],'1i64') if len(args)==1 else (args[0],args[1],args[2] if len(args)==3 else '1i64')
                    target = self.assign(n, n.target.id, f'current{j} as i64','i64')
                    header = f'''{{ let mut cursor{j} = ({start}) as i128;
let stop{j} = ({stop}) as i128; let step{j} = ({step}) as i128;
if step{j}==0 {{return Err(2)}}
while if step{j}>0 {{cursor{j}<stop{j}}} else {{cursor{j}>stop{j}}} {{
tick(fuel)?;
let current{j}=cursor{j}; cursor{j}+=step{j};
{target}'''
                    tail = '}}'
                else:
                    header = f'while {self.condition(n.test)} {{\ntick(fuel)?;'
                    tail = '}'
                self.loop_depth += 1
                body = self.block(n.body)
                self.loop_depth -= 1
                self.assigned = before
                lines.append(f'{header}\n{body}\n{tail}')
            elif isinstance(n,(ast.Break,ast.Continue)) and self.loop_depth:
                lines.append('break;' if isinstance(n,ast.Break) else 'continue;')
                exits = True
            else:
                self.fail(n, f'unsupported statement ({type(n).__name__})')
        self.exits = exits
        return '\n'.join(lines)

    # ------------------------------------------------------------- generation
    def function_text(self, rust_name):
        params = list(self.signature.parameters)
        body = self.block(self.node.body)
        if self.return_type is None:
            self.return_type = 'none'  # no return statement: Python returns None
        recursive = self.program['recursive'].get(id(self.function))
        if recursive is not None and recursive != self.return_type:
            raise Unsupported(f'{self.function.__name__}: recursive function returns {self.return_type} '
                              f'but is annotated {recursive}')
        rust_type = '()' if self.return_type == 'none' else self.return_type
        # A value-returning path that falls off the end returns Err(5): the call is
        # replayed in Python, which returns None exactly as the original function does.
        end = 'Ok(())' if self.return_type == 'none' else 'Err(5)'
        declarations = '\n'.join(f'let mut {self.name(k)}: {t} = {"0.0" if t=="f64" else "0"};'
                                 for k,t in self.types.items() if k not in params and t in ('i64','f64'))
        arguments = []
        for p in params:
            kind = self.types[p]
            if kind in ARRAY_KINDS:
                ref = '&mut ' if p in self.written else '&'
                arguments.append(f'{self.name(p)}: {ref}[{ELEMENT[kind]}]')
            else:
                arguments.append(f'mut {self.name(p)}: {kind}')
        arguments += ['fuel: &mut u64', 'depth: u32']
        return (f'fn {rust_name}({", ".join(arguments)})->Result<{rust_type},i32>{{\n'
                f'if depth>{MAX_DEPTH} {{return Err(6)}}\n'
                f'{declarations}\n{body}\n{end}\n}}')

    def generate(self):
        params = list(self.signature.parameters)
        self.program['stack'].append(self.function)
        self.register('compute')
        try:
            main = self.function_text('compute')
        finally:
            self.program['stack'].pop()
        ffi=[]; calls=[]; setup=[]; publish=[]
        for p in params:
            name=self.name(p); kind=self.types[p]
            if kind in ARRAY_KINDS:
                t = ELEMENT[kind]
                if p in self.written:
                    # Copy-on-write: the caller's array changes only if the call succeeds.
                    ffi.extend([f'{name}: *mut {t}', f'{name}_len: usize'])
                    setup.append(f'let mut copy_{name}: Vec<{t}> = if {name}_len==0 {{ Vec::new() }} '
                                 f'else {{ std::slice::from_raw_parts({name},{name}_len).to_vec() }};')
                    calls.append(f'&mut copy_{name}[..]')
                    publish.append(f'if {name}_len>0 {{ std::slice::from_raw_parts_mut({name},{name}_len)'
                                   f'.copy_from_slice(&copy_{name}); }}')
                else:
                    ffi.extend([f'{name}: *const {t}', f'{name}_len: usize'])
                    calls.append(f'if {name}_len==0 {{ &[] }} else {{ std::slice::from_raw_parts({name},{name}_len) }}')
            else:
                ffi.append(f'{name}: {kind}')
                calls.append(name)
        out_type = 'i64' if self.return_type == 'none' else self.return_type
        store = '' if self.return_type == 'none' else '*out=v;'
        ffi.append(f'out: *mut {out_type}')
        calls += ['&mut fuel', '0']
        callees = '\n'.join(self.program['functions'])
        return f'''// Generated by PyRust Engine 0.6.
#![allow(unused_mut,unused_assignments,unused_parens,unreachable_code,dead_code,unused_variables,while_true)]
{HELPERS}
{callees}
{main}
#[no_mangle]
pub unsafe extern "C" fn pyrust_run({','.join(ffi)})->i32{{
let mut fuel:u64={self.max_iterations};
{chr(10).join(setup)}
match compute({','.join(calls)}) {{Ok(v)=>{{{' '.join(publish)} {store} 0}},Err(code)=>code}}
}}
'''

    # ---------------------------------------------------------- introspection
    @property
    def guards(self):
        return tuple(self.program['guards'].values())

    @property
    def code_guards(self):
        return tuple(self.program['code_guards'].values())

    @property
    def callees(self):
        return tuple(self.program['callees'])


class _Missing:
    def __repr__(self):
        return '<missing>'

_MISSING = _Missing()
