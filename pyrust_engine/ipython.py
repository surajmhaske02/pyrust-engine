"""Jupyter / IPython magics. Load with `%load_ext pyrust_engine`.

%pyrust_check fn            explain whether fn compiles, and why not
%pyrust_benchmark fn(args)  time plain Python vs PyRust and check the results match
%%pyrust_benchmark          same, with the call on the lines of the cell
"""

import ast
import timeit


def _time(fn, args):
    timer = timeit.Timer(lambda: fn(*args))
    number, _ = timer.autorange()
    best = min(timer.repeat(repeat=5, number=number)) / number
    return best


def _show(seconds):
    if seconds >= 1:
        return f'{seconds:.3f} s'
    if seconds >= 1e-3:
        return f'{seconds * 1e3:.3f} ms'
    return f'{seconds * 1e6:.3f} µs'


def check(name, namespace):
    from .runtime import optimize

    obj = namespace.get(name.strip())
    if obj is None:
        print(f'{name!r} is not defined in this notebook')
        return
    fn = getattr(obj, 'python', obj)
    explain = getattr(obj, 'explain', None) or optimize(fn, warn_on_fallback=False).explain
    print(explain())


def benchmark(expression, namespace):
    from .runtime import optimize

    try:
        call = ast.parse(expression.strip(), mode='eval').body
    except SyntaxError:
        call = None
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.keywords:
        print('Usage: %pyrust_benchmark function(arg1, arg2, ...)')
        return
    obj = namespace.get(call.func.id)
    if obj is None:
        print(f'{call.func.id!r} is not defined in this notebook')
        return
    args = [eval(compile(ast.Expression(a), '<pyrust_benchmark>', 'eval'), namespace) for a in call.args]
    plain = getattr(obj, 'python', obj)
    fast = obj if hasattr(obj, 'report') else optimize(plain)
    expected = plain(*args)
    actual = fast(*args)  # compiles on the first call
    if repr(actual) != repr(expected) and not (actual == expected):
        print(f'Results differ: Python {expected!r}, PyRust {actual!r}')
        return
    t_python = _time(plain, args)
    t_fast = _time(fast, args)
    backend = fast.report()['backend']
    ratio = t_python / t_fast
    verdict = (
        f'{ratio:.1f}x faster'
        if ratio >= 1.1
        else f'{1 / ratio:.1f}x slower'
        if ratio <= 0.9
        else 'about the same speed'
    )
    print(f'{call.func.id}: same result ({actual!r})')
    print(f'  plain Python : {_show(t_python)}')
    print(f'  PyRust       : {_show(t_fast)}   (ran as: {backend})')
    print(f'  ==> {verdict}')
    if backend != 'rust':
        print('  ' + fast.explain().replace('\n', '\n  '))


def load(ipython):
    def pyrust_check(line):
        check(line, ipython.user_ns)

    def pyrust_benchmark(line, cell=None):
        benchmark(cell if cell is not None else line, ipython.user_ns)

    ipython.register_magic_function(pyrust_check, 'line', 'pyrust_check')
    ipython.register_magic_function(pyrust_benchmark, 'line_cell', 'pyrust_benchmark')
