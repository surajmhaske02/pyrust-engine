"""Conservative typed Python AST to Rust. Unsupported semantics fall back."""
import ast
import inspect
import math
import textwrap

class Unsupported(ValueError):
    pass

class F64Array:
    """Annotation for a contiguous, aligned, native-endian 1-D NumPy float64 array.

    Read elements using float(array[index]); native kernels never write the array.
    """

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
fn at(a:&[f64],i:i64)->Result<f64,i32>{
    let j=if i<0 {(a.len() as i128)+(i as i128)} else {i as i128};
    if j<0 || j>=a.len() as i128 {Err(3)} else {Ok(a[j as usize])}
}
fn tick(fuel:&mut u64)->Result<(),i32>{if *fuel==0 {Err(4)} else {*fuel-=1;Ok(())}}
'''

class Compiler:
    def __init__(self, function, max_iterations=10_000_000):
        if hasattr(function, '__wrapped__'):
            raise Unsupported('Wrapped functions are not supported')
        self.tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        self.node = self.tree.body[0]
        if not isinstance(self.node, ast.FunctionDef) or function.__closure__:
            raise Unsupported('Only ordinary functions without closure capture are supported')
        self.signature = inspect.signature(function)
        self.types = {}
        for name, p in self.signature.parameters.items():
            if p.kind not in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
                raise Unsupported('Variadic and keyword-only parameters are unsupported')
            annotation = p.annotation
            if annotation in (inspect.Parameter.empty, int, 'int'):
                kind = 'i64'
            elif annotation in (float, 'float'):
                kind = 'f64'
            elif annotation in (F64Array, 'F64Array'):
                kind = 'array'
            else:
                raise Unsupported(f'Unsupported annotation for {name}: {annotation!r}')
            self.types[name] = kind
        self.names = {p: f'v{i}' for i,p in enumerate(self.types)}
        self.assigned = set(self.names)
        self.local_names = set(self.names) | {n.id for n in ast.walk(self.node)
                                             if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        self.used_builtins = set()
        self.counter = 0
        self.loop_depth = 0
        self.return_type = None
        self.max_iterations = max_iterations

    def name(self, name):
        if name not in self.names:
            self.names[name] = f'v{len(self.names)}'
        return self.names[name]

    def builtin(self, name):
        if name in self.local_names:
            raise Unsupported(f'Local {name} shadowing is unsupported')
        self.used_builtins.add(name)

    def expect(self, node, kind):
        value, actual = self.expr(node)
        if actual != kind:
            raise Unsupported(f'Expected {kind}, got {actual}')
        return value

    def expr(self, n):
        if isinstance(n, ast.Constant):
            if type(n.value) is int and -(2**63) <= n.value < 2**63:
                return f'({n.value}i64)', 'i64'
            if type(n.value) is float and math.isfinite(n.value):
                return f'({n.value!r}f64)', 'f64'
        if isinstance(n, ast.Name) and n.id in self.assigned:
            return self.name(n.id), self.types[n.id]
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            # AST represents -2**63 as unary minus applied to the positive literal.
            if isinstance(n.operand, ast.Constant) and type(n.operand.value) is int and n.operand.value == 2**63:
                return 'i64::MIN', 'i64'
            value, kind = self.expr(n.operand)
            if kind == 'i64':
                return f'({value}).checked_neg().ok_or(1)?', kind
            if kind == 'f64':
                return f'(-({value}))', kind
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.UAdd):
            value, kind = self.expr(n.operand)
            if kind in ('i64', 'f64'):
                return f'({value})', kind
        if isinstance(n, ast.BinOp):
            a, ta = self.expr(n.left)
            b, tb = self.expr(n.right)
            if 'array' in (ta, tb):
                raise Unsupported('Arrays must be indexed inside float(array[index])')
            if ta == tb == 'i64':
                op = {ast.Add:'checked_add',ast.Sub:'checked_sub',ast.Mult:'checked_mul'}.get(type(n.op))
                if op:
                    return f'({a}).{op}({b}).ok_or(1)?', 'i64'
                if isinstance(n.op, (ast.FloorDiv, ast.Mod)):
                    op = 'floor_div' if isinstance(n.op, ast.FloorDiv) else 'modulo'
                    return f'{op}({a},{b})?', 'i64'
            elif 'f64' in (ta, tb):
                a = f'({a} as f64)' if ta == 'i64' else a
                b = f'({b} as f64)' if tb == 'i64' else b
                op = {ast.Add:'+',ast.Sub:'-',ast.Mult:'*'}.get(type(n.op))
                if op:
                    return f'(({a}) {op} ({b}))', 'f64'
                if isinstance(n.op, ast.Div):
                    return f'divide({a},{b})?', 'f64'
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and len(n.args)==1 and not n.keywords:
            name = n.func.id
            arg = n.args[0]
            if name == 'len':
                self.builtin(name)
                a = self.expect(arg, 'array')
                return f'({a}.len() as i64)', 'i64'
            if name == 'float':
                self.builtin(name)
                if isinstance(arg, ast.Subscript):
                    a = self.expect(arg.value, 'array')
                    i = self.expect(arg.slice, 'i64')
                    return f'at({a},{i})?', 'f64'
                a, kind = self.expr(arg)
                if kind in ('i64','f64'):
                    return f'({a} as f64)', 'f64'
        raise Unsupported(f'Unsupported expression: {ast.dump(n)}')

    def condition(self, n):
        if isinstance(n, ast.BoolOp) and isinstance(n.op, (ast.And, ast.Or)):
            op = ' && ' if isinstance(n.op, ast.And) else ' || '
            return '(' + op.join(f'({self.condition(v)})' for v in n.values) + ')'
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            return f'!({self.condition(n.operand)})'
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
                if ta != tb or ta == 'array':
                    raise Unsupported('Comparison operands must have the same scalar type')
                parts.append(f'(({a}) {ops[type(op)]} ({b}))')
            return '(' + ' && '.join(parts) + ')'
        try:
            value, kind = self.expr(n)
        except Unsupported:
            raise Unsupported('Use scalar comparisons in conditions') from None
        if kind == 'i64':
            return f'(({value}) != 0i64)'
        if kind == 'f64':
            return f'(({value}) != 0.0f64)'  # NaN is truthy, -0.0 is falsy, as in Python
        raise Unsupported('Use scalar comparisons in conditions')

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

    def assign(self, name, value, kind):
        if kind == 'array' or (name in self.types and self.types[name] != kind):
            raise Unsupported('Array aliases and changing local variable types are unsupported')
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
            if isinstance(n, ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
                value, kind = self.expr(n.value)
                lines.append(self.assign(n.targets[0].id, value, kind))
            elif isinstance(n, ast.AnnAssign) and isinstance(n.target,ast.Name) and n.value is not None:
                value, kind = self.expr(n.value)
                lines.append(self.assign(n.target.id, value, kind))
            elif isinstance(n, ast.AugAssign) and isinstance(n.target,ast.Name):
                value, kind = self.expr(ast.BinOp(left=n.target,op=n.op,right=n.value))
                lines.append(self.assign(n.target.id,value,kind))
            elif isinstance(n, ast.Return) and n.value is not None:
                value, kind = self.expr(n.value)
                if kind == 'array' or (self.return_type and kind != self.return_type):
                    raise Unsupported('All returns must have the same scalar type')
                self.return_type = kind
                lines.append(f'return Ok({value});')
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
                        raise Unsupported('For loops must use built-in range')
                    self.builtin('range')
                    args = [self.expect(a,'i64') for a in call.args]
                    start, stop, step = ('0i64',args[0],'1i64') if len(args)==1 else (args[0],args[1],args[2] if len(args)==3 else '1i64')
                    target = self.assign(n.target.id, f'current{j} as i64','i64')
                    header = f'''{{ let mut cursor{j} = ({start}) as i128;
let stop{j} = ({stop}) as i128; let step{j} = ({step}) as i128;
if step{j}==0 {{return Err(2)}}
while if step{j}>0 {{cursor{j}<stop{j}}} else {{cursor{j}>stop{j}}} {{
tick(&mut fuel)?;
let current{j}=cursor{j}; cursor{j}+=step{j};
{target}'''
                    tail = '}}'
                else:
                    header = f'while {self.condition(n.test)} {{\ntick(&mut fuel)?;'
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
                raise Unsupported(f'Unsupported statement: {type(n).__name__}')
        self.exits = exits
        return '\n'.join(lines)

    def generate(self):
        params = list(self.signature.parameters)
        body = self.block(self.node.body)
        if self.return_type is None:
            raise Unsupported('At least one scalar return is required')
        # A path that falls off the end returns Err(5): the call is replayed in
        # Python, which returns None exactly as the original function does.
        declarations = '\n'.join(f'let mut {self.name(k)}: {t} = {"0.0" if t=="f64" else "0"};'
                                 for k,t in self.types.items() if k not in params)
        arguments=[]; ffi=[]; calls=[]
        for p in params:
            name=self.name(p); kind=self.types[p]
            if kind=='array':
                arguments.append(f'{name}: &[f64]')
                ffi.extend([f'{name}: *const f64', f'{name}_len: usize'])
                calls.append(f'if {name}_len==0 {{ &[] }} else {{ std::slice::from_raw_parts({name},{name}_len) }}')
            else:
                arguments.append(f'mut {name}: {kind}')
                ffi.append(f'{name}: {kind}')
                calls.append(name)
        ffi.append(f'out: *mut {self.return_type}')
        return f'''// Generated by PyRust Engine 0.3.
#![allow(unused_mut,unused_assignments,unused_parens,unreachable_code,dead_code,unused_variables)]
{HELPERS}
fn compute({','.join(arguments)})->Result<{self.return_type},i32>{{
let mut fuel:u64={self.max_iterations};
{declarations}
{body}
Err(5)
}}
#[no_mangle]
pub unsafe extern "C" fn pyrust_run({','.join(ffi)})->i32{{
match compute({','.join(calls)}) {{Ok(v)=>{{*out=v;0}},Err(code)=>code}}
}}
'''
