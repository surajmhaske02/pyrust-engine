"""Conservative, side-effect-free Python integer subset to checked Rust.

Every generated operation either produces exactly the value CPython would, or
makes the native kernel return ``None``.  ``None`` means "could not finish
natively" (i64 overflow, division by zero, ``range`` step of zero, ...); the
runtime then replays the original function in CPython, which produces the
correct big-integer result or raises the correct Python exception.  The
accepted subset is pure, so the replay cannot duplicate side effects.
"""
import ast
import inspect
import textwrap

I64_MIN = -(2 ** 63)
I64_MAX = 2 ** 63 - 1


class Unsupported(ValueError):
    pass


# Python floor-division and modulo semantics on i64. ``None`` on b == 0 and on
# the single overflowing case (i64::MIN // -1 and i64::MIN % -1); the runtime
# replays those calls in Python.
_HELPERS = '''
#[inline(always)]
fn py_floordiv(a: i64, b: i64) -> Option<i64> {
    let q = a.checked_div(b)?;
    if (a % b != 0) && ((a < 0) != (b < 0)) { Some(q - 1) } else { Some(q) }
}
#[inline(always)]
fn py_mod(a: i64, b: i64) -> Option<i64> {
    let r = a.checked_rem(b)?;
    if r != 0 && ((r < 0) != (b < 0)) { Some(r + b) } else { Some(r) }
}
'''

_BINOPS = {ast.Add: 'checked_add', ast.Sub: 'checked_sub', ast.Mult: 'checked_mul'}
_HELPER_BINOPS = {ast.FloorDiv: 'py_floordiv', ast.Mod: 'py_mod'}
_CMPOPS = {ast.Lt: '<', ast.LtE: '<=', ast.Gt: '>', ast.GtE: '>=', ast.Eq: '==', ast.NotEq: '!='}


class Compiler:
    def __init__(self, function):
        if hasattr(function, '__wrapped__'):
            raise Unsupported('Wrapped functions are not supported')
        self.function = function
        self.tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        self.node = self.tree.body[0]
        if not isinstance(self.node, ast.FunctionDef):
            raise Unsupported('Only ordinary functions are supported')
        self.signature = inspect.signature(function)
        for p in self.signature.parameters.values():
            if p.kind not in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
                raise Unsupported('Variadic and keyword-only parameters are not supported')
        if function.__closure__:
            raise Unsupported('Closures are not supported')
        if any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id == 'range'
               for n in ast.walk(self.node)) or 'range' in self.signature.parameters:
            raise Unsupported('Local range shadowing is unsupported')
        self.names = {p: f'v{i}' for i, p in enumerate(self.signature.parameters)}
        self.assigned = set(self.names)
        self.counter = 0
        self.uses_range = False

    def name(self, value):
        if value not in self.names:
            self.names[value] = f'v{len(self.names)}'
        return self.names[value]

    # ---------------------------------------------------------------- expressions
    def expr(self, n):
        if isinstance(n, ast.Constant) and type(n.value) is int and I64_MIN <= n.value <= I64_MAX:
            return f'({n.value}i64)'
        if isinstance(n, ast.Name):
            if n.id in self.assigned:
                return self.name(n.id)
            raise Unsupported(f'Name {n.id!r} is not a definitely assigned local integer')
        if isinstance(n, ast.BinOp) and type(n.op) in _BINOPS:
            return f'({self.expr(n.left)}).{_BINOPS[type(n.op)]}({self.expr(n.right)})?'
        if isinstance(n, ast.BinOp) and type(n.op) in _HELPER_BINOPS:
            return f'{_HELPER_BINOPS[type(n.op)]}({self.expr(n.left)}, {self.expr(n.right)})?'
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return f'({self.expr(n.operand)}).checked_neg()?'
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.UAdd):
            return f'({self.expr(n.operand)})'
        raise Unsupported(f'Unsupported integer expression: {ast.dump(n)}')

    def condition(self, n):
        """Translate a test to a Rust bool. Operands are pure, so evaluating a
        sub-expression Python would have skipped can only cause a replay."""
        if isinstance(n, ast.Compare) and all(type(op) in _CMPOPS for op in n.ops):
            operands = [n.left] + n.comparators
            parts = [f'({self.expr(a)} {_CMPOPS[type(op)]} {self.expr(b)})'
                     for a, op, b in zip(operands, n.ops, operands[1:])]
            return '(' + ' && '.join(parts) + ')'
        if isinstance(n, ast.BoolOp):
            joiner = ' && ' if isinstance(n.op, ast.And) else ' || '
            return '(' + joiner.join(self.condition(v) for v in n.values) + ')'
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            return f'(!{self.condition(n.operand)})'
        try:
            return f'({self.expr(n)} != 0i64)'  # integer truthiness: `if n:`
        except Unsupported:
            raise Unsupported('Conditions must be integer comparisons, and/or/not, or integer truthiness')

    # ----------------------------------------------------------------- statements
    def block(self, nodes):
        """Return (rust_code, always_returns)."""
        lines = []
        for n in nodes:
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str):
                continue
            if isinstance(n, ast.Pass):
                continue
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                self._assign(lines, n.targets[0].id, n.value)
            elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.simple:
                if n.value is not None:  # local annotations are never evaluated at runtime
                    self._assign(lines, n.target.id, n.value)
            elif isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name):
                value = self.expr(ast.BinOp(left=ast.Name(id=n.target.id, ctx=ast.Load()), op=n.op, right=n.value))
                lines.append(f'{self.name(n.target.id)} = {value};')
            elif isinstance(n, ast.Return) and n.value is not None:
                lines.append(f'return Some({self.expr(n.value)});')
                return '\n'.join(lines), True
            elif isinstance(n, ast.If):
                cond = self.condition(n.test)
                before = self.assigned.copy()
                yes, yes_returns = self.block(n.body)
                yes_assigned = self.assigned
                self.assigned = before.copy()
                no, no_returns = self.block(n.orelse)
                no_assigned = self.assigned
                # A branch that always returns never reaches the next statement,
                # so it does not constrain what is definitely assigned afterwards.
                if yes_returns and no_returns:
                    self.assigned = yes_assigned | no_assigned
                elif yes_returns:
                    self.assigned = no_assigned
                elif no_returns:
                    self.assigned = yes_assigned
                else:
                    self.assigned = yes_assigned & no_assigned
                lines.append(f'if {cond} {{\n{yes}\n}} else {{\n{no}\n}}')
                if yes_returns and no_returns:
                    return '\n'.join(lines), True
            elif isinstance(n, ast.For) and isinstance(n.target, ast.Name) and not n.orelse:
                lines.append(self._for(n))
            else:
                raise Unsupported(f'Unsupported statement: {type(n).__name__}')
        return '\n'.join(lines), False

    def _assign(self, lines, name, value_node):
        value = self.expr(value_node)
        lines.append(f'{self.name(name)} = {value};')
        self.assigned.add(name)

    def _for(self, n):
        call = n.iter
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'range'
                and len(call.args) in (1, 2, 3) and not call.keywords):
            raise Unsupported('Only range(stop), range(start, stop) and range(start, stop, step) loops are supported')
        self.uses_range = True
        start = self.expr(call.args[0]) if len(call.args) >= 2 else '0i64'
        stop = self.expr(call.args[1] if len(call.args) >= 2 else call.args[0])
        step = self.expr(call.args[2]) if len(call.args) == 3 else None
        before = self.assigned.copy()
        target = self.name(n.target.id)
        self.assigned.add(n.target.id)
        self.counter += 1
        c = self.counter
        body, _ = self.block(n.body)
        # The body may run zero times: nothing assigned inside it is definite afterwards.
        self.assigned = before
        if step is None:
            return f'for index{c} in ({start})..({stop}) {{\n{target} = index{c};\n{body}\n}}'
        # range() with an explicit step. step == 0 raises ValueError in Python -> replay.
        # If index + step overflows i64 it is necessarily past `stop`, so the loop ends.
        return (f'{{\nlet start{c}: i64 = {start};\nlet stop{c}: i64 = {stop};\nlet step{c}: i64 = {step};\n'
                f'if step{c} == 0 {{ return None; }}\n'
                f'let mut index{c}: i64 = start{c};\n'
                f'while (step{c} > 0 && index{c} < stop{c}) || (step{c} < 0 && index{c} > stop{c}) {{\n'
                f'{target} = index{c};\n{body}\n'
                f'index{c} = match index{c}.checked_add(step{c}) {{ Some(v) => v, None => break }};\n}}\n}}')

    def generate(self):
        params = list(self.signature.parameters)
        body, returns = self.block(self.node.body)
        if not returns:
            raise Unsupported('Every path must end in an integer return')
        declarations = '\n'.join(f'let mut {v}: i64 = 0;' for k, v in self.names.items() if k not in params)
        arguments = ', '.join(f'mut {self.name(p)}: i64' for p in params)
        ffi_args = ', '.join(f'{self.name(p)}: i64' for p in params)
        calls = ', '.join(self.name(p) for p in params)
        return f'''// Generated by pyrust-engine. Do not edit.
#![allow(unused_mut, unused_assignments, unused_parens, unreachable_code, unused_variables, dead_code)]
{_HELPERS}
fn compute({arguments}) -> Option<i64> {{
{declarations}
{body}
}}
#[no_mangle]
pub unsafe extern "C" fn pyrust_run({ffi_args}{', ' if params else ''}out: *mut i64) -> i32 {{
    match compute({calls}) {{ Some(value) => {{ *out = value; 0 }}, None => 1 }}
}}
'''
