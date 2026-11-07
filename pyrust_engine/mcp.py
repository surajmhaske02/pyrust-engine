"""MCP server: lets coding agents (Claude Code, Cursor, ...) use PyRust as a tool.

Run:  python -m pyrust_engine mcp      (or the `pyrust-mcp` command)

Claude Code:  claude mcp add pyrust -- python -m pyrust_engine mcp

Tools:
  check_code    does this Python code compile to Rust? which line blocks it, and the hint
  suggest_file  which functions in a file to decorate (same as `suggest --json`)
  apply_file    add @optimize where it compiles; returns a diff, writes only if asked
  rust_source   the Rust code PyRust generates for a function
  benchmark     time plain Python against PyRust for one call, and check the results match
  doctor        is Rust installed and working?

The protocol is JSON-RPC 2.0 over stdin/stdout (Model Context Protocol, tools only),
with no dependencies. check_code, rust_source and benchmark execute the code they are
given, like running it yourself: the server is a local tool for code you trust.
"""

import contextlib
import io
import json
import sys
import tempfile
import timeit
import traceback
from pathlib import Path

PROTOCOL = '2025-06-18'


def _load_code(code):
    """Run `code` as a module from a temporary file (so the compiler can read its source)."""
    import runpy

    folder = tempfile.mkdtemp(prefix='pyrust-mcp-')
    path = Path(folder) / 'snippet.py'
    path.write_text(code)
    with contextlib.redirect_stdout(io.StringIO()):
        namespace = runpy.run_path(str(path), run_name='__pyrust_mcp__')
    return path, namespace


def _functions(namespace, path, name=None):
    import inspect

    found = {}
    for key, value in namespace.items():
        fn = getattr(value, 'python', value)
        if inspect.isfunction(fn) and fn.__code__.co_filename == str(path):
            found[key] = fn
    if name is not None:
        if name not in found:
            raise ValueError(f'no function named {name!r} in the code (found: {", ".join(found) or "none"})')
        return {name: found[name]}
    return found


def check_code(code, function=None):
    from .advisor import _parse_reason
    from .compiler import Compiler

    path, namespace = _load_code(code)
    results = []
    for name, fn in _functions(namespace, path, function).items():
        row = {'function': name, 'compiles': False}
        try:
            compiler = Compiler(fn)
            compiler.generate()
            row.update(
                compiles=True,
                parameters={p: compiler.types[p] for p in compiler.params},
                returns=compiler.return_type,
                helpers=list(compiler.callees),
                parallel=compiler.program['parallel'],
            )
        except Exception as exc:
            row['problem'] = _parse_reason(str(exc))
        results.append(row)
    return {'functions': results}


def rust_source(code, function):
    from .compiler import Compiler

    path, namespace = _load_code(code)
    fn = _functions(namespace, path, function)[function]
    return {'function': function, 'rust': Compiler(fn).generate()}


def benchmark(code, call):
    """Time `call` (an expression such as "score(data, 3)") as plain Python and with @optimize."""
    from .runtime import optimize

    path, namespace = _load_code(code)
    functions = _functions(namespace, path)
    name = call.split('(', 1)[0].strip()
    if name not in functions:
        raise ValueError(f'call must start with a function defined in the code, like {next(iter(functions), "f")}(...)')
    plain = functions[name]
    fast = optimize(plain, warn_on_fallback=False)
    env = dict(namespace)
    env[name] = plain
    expected = eval(call, env)
    env[name] = fast
    actual = eval(call, env)
    same = repr(expected) == repr(actual) or expected == actual

    def best(fn):
        env[name] = fn
        timer = timeit.Timer(lambda: eval(call, env))
        number, _ = timer.autorange()
        return min(timer.repeat(repeat=3, number=number)) / number

    python_seconds = best(plain)
    pyrust_seconds = best(fast)
    return {
        'call': call,
        'same_result': bool(same),
        'python_seconds': python_seconds,
        'pyrust_seconds': pyrust_seconds,
        'speedup': python_seconds / pyrust_seconds if pyrust_seconds else None,
        'backend': fast.report()['backend'],
        'reason': fast.report()['reason'],
    }


def suggest_file(path):
    from .advisor import suggest

    return suggest(path)


def apply_file(path, write=False):
    from .advisor import apply

    return apply(path, write=bool(write))


def doctor():
    from .toolchain import check_rust

    return check_rust()


TOOLS = {
    'check_code': (
        check_code,
        'Check whether Python functions compile to native Rust with PyRust (@optimize). Returns, per function, '
        'whether it compiles and, if not, the blocking line, the reason and a rewrite hint.',
        {
            'code': {'type': 'string', 'description': 'Python source code (a module) defining the functions'},
            'function': {'type': 'string', 'description': 'Only check this function (optional)'},
        },
        ['code'],
    ),
    'suggest_file': (
        suggest_file,
        'List the functions in a Python file and what to do with each: add @optimize, skip (no loop), '
        'or a rewrite hint. Loads the file (its top-level code runs, the __main__ block does not).',
        {'path': {'type': 'string', 'description': 'Path to a .py file'}},
        ['path'],
    ),
    'apply_file': (
        apply_file,
        'Add @optimize (and the import) to every function in the file that compiles and has a loop. '
        'Returns a unified diff; writes the file only when write is true.',
        {
            'path': {'type': 'string', 'description': 'Path to a .py file'},
            'write': {'type': 'boolean', 'description': 'Edit the file (default false: only show the diff)'},
        },
        ['path'],
    ),
    'rust_source': (
        rust_source,
        'Show the Rust code PyRust generates for one function.',
        {
            'code': {'type': 'string', 'description': 'Python source code defining the function'},
            'function': {'type': 'string', 'description': 'Function name'},
        },
        ['code', 'function'],
    ),
    'benchmark': (
        benchmark,
        'Run one call as plain Python and with PyRust, check both give the same result, and report the speedup. '
        'Executes the code.',
        {
            'code': {'type': 'string', 'description': 'Python source code defining the function and any data'},
            'call': {'type': 'string', 'description': 'Expression to time, e.g. "score(values, 3)"'},
        },
        ['code', 'call'],
    ),
    'doctor': (doctor, 'Check whether Rust is installed and working for PyRust.', {}, []),
}


def _tool_list():
    return [
        {
            'name': name,
            'description': description,
            'inputSchema': {'type': 'object', 'properties': properties, 'required': required},
        }
        for name, (_, description, properties, required) in TOOLS.items()
    ]


def handle(message):
    """Response for one JSON-RPC message (None for notifications)."""
    method = message.get('method')
    ident = message.get('id')
    if ident is None:
        return None  # notifications/initialized and other notifications
    if method == 'initialize':
        from . import __version__

        params = message.get('params') or {}
        result = {
            'protocolVersion': params.get('protocolVersion', PROTOCOL),
            'capabilities': {'tools': {'listChanged': False}},
            'serverInfo': {'name': 'pyrust-engine', 'version': __version__},
            'instructions': 'Use check_code before adding @optimize; apply_file shows a diff unless write=true.',
        }
    elif method == 'ping':
        result = {}
    elif method == 'tools/list':
        result = {'tools': _tool_list()}
    elif method == 'tools/call':
        params = message.get('params') or {}
        name = params.get('name')
        if name not in TOOLS:
            return {'jsonrpc': '2.0', 'id': ident, 'error': {'code': -32602, 'message': f'unknown tool {name!r}'}}
        try:
            with contextlib.redirect_stdout(sys.stderr):
                value = TOOLS[name][0](**(params.get('arguments') or {}))
            result = {'content': [{'type': 'text', 'text': json.dumps(value, indent=2)}], 'isError': False}
        except Exception as exc:
            text = ''.join(traceback.format_exception_only(type(exc), exc)).strip()
            result = {'content': [{'type': 'text', 'text': text}], 'isError': True}
    else:
        return {'jsonrpc': '2.0', 'id': ident, 'error': {'code': -32601, 'message': f'method not found: {method}'}}
    return {'jsonrpc': '2.0', 'id': ident, 'result': result}


def main(stdin=None, stdout=None):
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except ValueError:
            reply = {'jsonrpc': '2.0', 'id': None, 'error': {'code': -32700, 'message': 'parse error'}}
        else:
            reply = handle(message) if isinstance(message, dict) else None
        if reply is not None:
            stdout.write(json.dumps(reply) + '\n')
            stdout.flush()


if __name__ == '__main__':
    main()
