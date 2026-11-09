"""`suggest` and `apply`: find functions worth compiling and add @optimize to them.

Made for people and for coding agents: both commands print JSON with --json, never
change a file unless asked (--write), and only touch functions that PyRust compiles.

    python -m pyrust_engine suggest app/scoring.py --json
    python -m pyrust_engine apply app/scoring.py            # shows a diff
    python -m pyrust_engine apply app/scoring.py --write    # edits the file

The file is loaded to look at its functions, like `check`: its top-level code runs, its
`if __name__ == '__main__':` block does not. Only use it on code you trust.
"""

import ast
import difflib
import inspect
import re
import runpy
import sys
import textwrap
from pathlib import Path

_REASON = re.compile(r'^(?P<function>[^,]+), line (?P<line>\d+): (?P<message>.*?)(?:: `(?P<code>.*)`)?$')
_DECORATORS = ('optimize', 'vectorize')


def _parse_reason(text):
    """Split a compiler message into line, message, code and hint."""
    lines = text.splitlines()
    first = lines[0] if lines else text
    hint = next((x.strip()[len('hint: ') :] for x in lines[1:] if x.strip().startswith('hint: ')), None)
    match = _REASON.match(
        first.replace('called function ', '').split("': ", 1)[-1] if first.startswith('called function') else first
    )
    if match:
        return {
            'line': int(match['line']),
            'message': match['message'],
            'code': match['code'],
            'hint': hint,
        }
    return {'line': None, 'message': first, 'code': None, 'hint': hint}


def _has_loop(node):
    for sub in ast.walk(node):
        if isinstance(sub, (ast.For, ast.While, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            return True
    return False


def _decorator_names(node):
    names = []
    for d in node.decorator_list:
        target = d.func if isinstance(d, ast.Call) else d
        if isinstance(target, ast.Attribute):
            names.append(target.attr)
        elif isinstance(target, ast.Name):
            names.append(target.id)
    return names


def _load(path):
    old_path = sys.path[:]
    try:
        sys.path.insert(0, str(path.parent))
        return runpy.run_path(str(path), run_name='__pyrust_check__')
    finally:
        sys.path[:] = old_path


def _annotation_kind(annotation):
    from .compiler import ANNOTATIONS

    try:
        kind = ANNOTATIONS.get(annotation)
    except TypeError:
        return None
    if kind is None and annotation in (str, 'str'):
        return 'str'
    if kind is None and annotation in (bytes, 'bytes'):
        return 'bytes'
    return kind


def _init_kinds(cls):
    """Attribute kinds from __init__: self.x = <number/str constant> or self.x = <annotated parameter>."""
    init = cls.__dict__.get('__init__')
    if not inspect.isfunction(init):
        return {}
    try:
        node = ast.parse(textwrap.dedent(inspect.getsource(init))).body[0]
    except (OSError, TypeError, SyntaxError, IndentationError):
        return {}
    params = {a.arg: a.annotation for a in node.args.args}
    hints = getattr(init, '__annotations__', {})
    found = {}
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Assign):
            continue
        for target in sub.targets:
            if not (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)):
                continue
            value = sub.value
            if isinstance(value, ast.Constant) and type(value.value) in (int, float, bool, str):
                found.setdefault(target.attr, _annotation_kind(type(value.value)))
            elif isinstance(value, ast.Name) and value.id in params and value.id in hints:
                found.setdefault(target.attr, _annotation_kind(hints[value.id]))
    return {k: v for k, v in found.items() if v}


def method_kinds(cls, function, node):
    """Kinds to check a method with: `self` as an object whose attribute kinds come from the
    class (annotations, dataclass fields, class attributes, simple __init__ assignments),
    including the attributes used by methods it calls on self. Returns (kinds, None) or
    (None, reason)."""
    from .compiler import class_object_uses, default_kinds, object_kind, object_uses

    params = [a.arg for a in node.args.args]
    if not params:
        return None, 'a method without self'
    first = params[0]
    uses = object_uses(node, [first], include_empty=True).get(first)
    if uses is None:
        return None, (
            f'{first!r} is used as a whole (passed on, compared or reassigned)\n'
            f'  hint: pass the values the loop needs, like helper({first}.x)'
        )
    attrs, methods = uses
    if methods:
        found = class_object_uses(cls, attrs, methods)
        if found is None:
            return None, (
                f'a method called on {first} cannot be compiled (a static/class method, a property, '
                f'or it uses {first} as a whole)\n'
                '  hint: make that helper a plain function that takes the values'
            )
        attrs = found[0]
    known = {}
    for klass in reversed(cls.__mro__[:-1]):
        for attr, value in vars(klass).items():
            if not attr.startswith('__') and type(value) in (int, float, bool, str, bytes):
                known[attr] = _annotation_kind(type(value))
        try:
            annotations = inspect.get_annotations(klass)  # also works with Python 3.14's lazy annotations
        except Exception:
            annotations = {}
        for attr, annotation in annotations.items():
            kind = _annotation_kind(annotation)
            if kind:
                known[attr] = kind
        known.update(_init_kinds(klass))
    missing = sorted(a for a in attrs if a not in known)
    if missing:
        return None, (
            f'types of {", ".join(f"{first}.{a}" for a in missing)} are not known from the class\n'
            '  hint: annotate them in the class body (x: float) or use a dataclass; a real call compiles anyway'
        )
    try:
        kinds = default_kinds(function)
    except Exception as exc:
        return None, str(exc)
    kinds[first] = object_kind([(a, known[a], attrs[a]) for a in sorted(attrs)], cls if methods else None)
    return kinds, None


def _targets(tree, namespace):
    """(qualified name, ast node, function or None, class or None) for top-level functions and methods."""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            value = namespace.get(node.name)
            yield node.name, node, getattr(value, 'python', value), None
        elif isinstance(node, ast.ClassDef):
            cls = namespace.get(node.name)
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and not item.name.startswith('__'):
                    raw = vars(cls).get(item.name) if isinstance(cls, type) else None
                    if isinstance(raw, (staticmethod, classmethod)):
                        continue
                    yield f'{node.name}.{item.name}', item, getattr(raw, 'python', raw), cls


def suggest(path):
    """For each top-level function and method: does it compile, is it worth it, and what to do."""
    from .compiler import Compiler

    path = Path(path).resolve()
    source = path.read_text()
    tree = ast.parse(source)
    namespace = _load(path)
    rows = []
    for qualified, node, function, cls in _targets(tree, namespace):
        decorators = _decorator_names(node)
        row = {
            'function': qualified,
            'line': node.lineno,
            'decorated': any(d in _DECORATORS for d in decorators),
            'other_decorators': [d for d in decorators if d not in _DECORATORS],
            'has_loop': _has_loop(node),
            'compiles': False,
            'return_type': None,
            'problem': None,
        }
        if inspect.isfunction(function):
            kinds, why = method_kinds(cls, function, node) if cls is not None else (None, None)
            try:
                if why is not None:
                    raise ValueError(why)
                compiler = Compiler(function, kinds=kinds)
                compiler.generate()
                row['compiles'] = True
                row['return_type'] = compiler.return_type
            except Exception as exc:
                row['problem'] = _parse_reason(str(exc))
        else:
            row['problem'] = {
                'line': None,
                'message': 'not a plain function after loading the file',
                'code': None,
                'hint': None,
            }
        if row['decorated']:
            row['action'] = 'none: already uses PyRust'
        elif not row['has_loop']:
            row['action'] = 'skip: no loop, so native code cannot make it faster'
        elif row['other_decorators']:
            row['action'] = 'review: has other decorators; add @optimize by hand if it is the hot part'
        elif row['compiles']:
            row['action'] = 'add @optimize'
        else:
            row['action'] = 'rewrite: ' + (row['problem']['hint'] or row['problem']['message'])
        rows.append(row)
    summary = {
        'functions': len(rows),
        'add_optimize': sum(r['action'] == 'add @optimize' for r in rows),
        'already_optimized': sum(r['decorated'] for r in rows),
        'needs_rewrite': sum(r['action'].startswith('rewrite') for r in rows),
    }
    return {'file': str(path), 'functions': rows, 'summary': summary}


def apply(path, write=False):
    """Add @optimize (and its import) to every function `suggest` marks 'add @optimize'."""
    path = Path(path).resolve()
    report = suggest(path)
    targets = {r['function'] for r in report['functions'] if r['action'] == 'add @optimize'}
    source = path.read_text()
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    inserts = []  # (line index, text)
    for node in tree.body:
        nodes = [(node.name, node)] if isinstance(node, ast.FunctionDef) else []
        if isinstance(node, ast.ClassDef):
            nodes = [(f'{node.name}.{n.name}', n) for n in node.body if isinstance(n, ast.FunctionDef)]
        for qualified, fn_node in nodes:
            if qualified in targets:
                first = fn_node.decorator_list[0].lineno if fn_node.decorator_list else fn_node.lineno
                indent = re.match(r'\s*', lines[first - 1]).group(0)
                inserts.append((first - 1, f'{indent}@optimize\n'))
    imported = any(
        isinstance(n, ast.ImportFrom)
        and n.module == 'pyrust_engine'
        and any(a.name == 'optimize' and a.asname is None for a in n.names)
        for n in tree.body
    )
    if inserts and not imported:
        position = 0
        for n in tree.body:
            is_docstring = (
                isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)
            )
            if isinstance(n, (ast.Import, ast.ImportFrom)) or (is_docstring and n is tree.body[0]):
                position = n.end_lineno
            else:
                break
        inserts.append((position, 'from pyrust_engine import optimize\n'))
    new_lines = list(lines)
    for index, text in sorted(inserts, key=lambda x: x[0], reverse=True):
        new_lines.insert(index, text)
    new_source = ''.join(new_lines)
    diff = ''.join(difflib.unified_diff(lines, new_lines, fromfile=str(path), tofile=str(path)))
    if write and new_source != source:
        compile(new_source, str(path), 'exec')  # never write a file that does not parse
        path.write_text(new_source)
    return {
        'file': str(path),
        'decorated': sorted(targets),
        'written': bool(write and new_source != source),
        'diff': diff,
        'note': 'Decorated functions give the same results as before: anything PyRust cannot run natively '
        'runs as plain Python. Run your tests to confirm, and check speed with %pyrust_benchmark.',
    }


def format_suggest(report):
    out = []
    for r in report['functions']:
        status = 'compiles' if r['compiles'] else 'does not compile'
        out.append(f'{r["function"]} (line {r["line"]}): {status}; {r["action"]}')
        problem = r['problem']
        if problem and not r['compiles'] and r['has_loop']:
            where = f'line {problem["line"]}: ' if problem['line'] else ''
            code = f' `{problem["code"]}`' if problem['code'] else ''
            out.append(f'    {where}{problem["message"]}{code}')
            if problem['hint']:
                out.append(f'    hint: {problem["hint"]}')
    s = report['summary']
    out.append('')
    out.append(
        f'{s["functions"]} functions: {s["add_optimize"]} ready for @optimize, '
        f'{s["already_optimized"]} already optimized, {s["needs_rewrite"]} need a small rewrite.'
    )
    return '\n'.join(out)
