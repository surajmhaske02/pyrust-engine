"""Conservative, side-effect-free Python integer subset to checked Rust."""
import ast
import inspect
import textwrap

class Unsupported(ValueError):
    pass

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

    def name(self, value):
        if value not in self.names:
            self.names[value] = f'v{len(self.names)}'
        return self.names[value]

    def expr(self, n):
        if isinstance(n, ast.Constant) and type(n.value) is int and -(2**63) < n.value < 2**63:
            return f'({n.value}i64)'
        if isinstance(n, ast.Name) and n.id in self.assigned:
            return self.name(n.id)
        if isinstance(n, ast.BinOp) and type(n.op) in (ast.Add, ast.Sub, ast.Mult):
            op = {ast.Add: 'checked_add', ast.Sub: 'checked_sub', ast.Mult: 'checked_mul'}[type(n.op)]
            return f'({self.expr(n.left)}).{op}({self.expr(n.right)})?'
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return f'({self.expr(n.operand)}).checked_neg()?'
        raise Unsupported(f'Unsupported integer expression: {ast.dump(n)}')

    def condition(self, n):
        ops = {ast.Lt: '<', ast.LtE: '<=', ast.Gt: '>', ast.GtE: '>=', ast.Eq: '==', ast.NotEq: '!='}
        if isinstance(n, ast.Compare) and len(n.ops) == 1 and type(n.ops[0]) in ops:
            return f'{self.expr(n.left)} {ops[type(n.ops[0])]} {self.expr(n.comparators[0])}'
        raise Unsupported('Conditions must be single integer comparisons')

    def block(self, nodes):
        lines = []
        for n in nodes:
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str):
                continue
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                value = self.expr(n.value)
                name = n.targets[0].id
                lines.append(f'{self.name(name)} = {value};')
                self.assigned.add(name)
            elif isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name):
                value = self.expr(ast.BinOp(left=n.target, op=n.op, right=n.value))
                lines.append(f'{self.name(n.target.id)} = {value};')
            elif isinstance(n, ast.Return) and n.value is not None:
                lines.append(f'return Some({self.expr(n.value)});')
            elif isinstance(n, ast.If):
                cond = self.condition(n.test)
                before = self.assigned.copy()
                yes = self.block(n.body)
                yes_assigned = self.assigned.copy()
                self.assigned = before.copy()
                no = self.block(n.orelse)
                self.assigned &= yes_assigned
                lines.append(f'if {cond} {{\n{yes}\n}} else {{\n{no}\n}}')
            elif isinstance(n, ast.For) and isinstance(n.target, ast.Name) and not n.orelse:
                call = n.iter
                if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'range'
                        and len(call.args) in (1, 2) and not call.keywords):
                    raise Unsupported('Only range(stop) and range(start, stop) loops are supported')
                if 'range' in self.names:
                    raise Unsupported('Local range shadowing is unsupported')
                start = self.expr(call.args[0]) if len(call.args) == 2 else '0i64'
                stop = self.expr(call.args[-1])
                before = self.assigned.copy()
                target = self.name(n.target.id)
                self.assigned.add(n.target.id)
                self.counter += 1
                iterator = f'index{self.counter}'
                body = self.block(n.body)
                self.assigned = before
                lines.append(f'for {iterator} in ({start})..({stop}) {{\n{target} = {iterator};\n{body}\n}}')
            else:
                raise Unsupported(f'Unsupported statement: {type(n).__name__}')
        return '\n'.join(lines)

    def generate(self):
        if not self.node.body or not isinstance(self.node.body[-1], ast.Return):
            raise Unsupported('A final integer return is required')
        params = list(self.signature.parameters)
        body = self.block(self.node.body)
        declarations = '\n'.join(f'let mut {v}: i64 = 0;' for k, v in self.names.items() if k not in params)
        arguments = ', '.join(f'mut {self.name(p)}: i64' for p in params)
        ffi_args = ', '.join(f'{self.name(p)}: i64' for p in params)
        calls = ', '.join(self.name(p) for p in params)
        return f'''// Generated by pyrust-engine. Do not edit.
#![allow(unused_mut, unused_assignments, unused_parens, unreachable_code)]
fn compute({arguments}) -> Option<i64> {{
{declarations}
{body}
None
}}
#[no_mangle]
pub unsafe extern "C" fn pyrust_run({ffi_args}{', ' if params else ''}out: *mut i64) -> i32 {{
    match compute({calls}) {{ Some(value) => {{ *out = value; 0 }}, None => 1 }}
}}
'''
