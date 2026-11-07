"""Ahead-of-time builds: compile once, ship the libraries, run without Rust.

`python -m pyrust_engine build mymodule.py` compiles every function in the file that
PyRust can compile and stores the libraries in `__pyrust__/` next to the file (or in
`--out DIR`). At run time PyRust looks there first, so machines without Rust (servers,
Docker images, end users) still get native speed for the shipped functions.

A library is matched by a hash of the generated Rust source and the platform, and its
own SHA-256 is checked before loading, so a stale or tampered file is ignored.
"""

import hashlib
import inspect
import json
import os
import platform
import sys
from pathlib import Path

from .cache import SUFFIX, digest

MANIFEST = 'manifest.json'


def platform_tag():
    return f'{sys.platform}-{platform.machine().lower()}'


def source_key(source):
    return hashlib.sha256((source + '\0' + platform_tag()).encode()).hexdigest()


def search_dirs(fn):
    dirs = [Path(p) for p in os.environ.get('PYRUST_PREBUILT', '').split(os.pathsep) if p]
    try:
        module_file = inspect.getsourcefile(fn)
    except TypeError:
        module_file = None
    if module_file:
        dirs.append(Path(module_file).resolve().parent / '__pyrust__')
    return dirs


def _manifest(directory):
    try:
        return json.loads((directory / MANIFEST).read_text())
    except (OSError, ValueError):
        return {}


def find(fn, source):
    """Path of a verified prebuilt library for this exact source on this platform, or None."""
    key = source_key(source)
    for directory in search_dirs(fn):
        entry = _manifest(directory).get(key)
        if not isinstance(entry, dict):
            continue
        path = directory / str(entry.get('file', ''))
        try:
            if path.is_file() and path.parent == directory and digest(path) == entry.get('sha256'):
                return path
        except OSError:
            continue
    return None


def build(script, out=None, stream=None):
    """Compile every compilable function in a trusted file into a prebuilt directory.

    Returns a list of {'function', 'ok', 'detail'} rows.
    """
    import runpy
    import shutil
    import tempfile

    from . import toolchain
    from .cache import compile_library
    from .compiler import Compiler, default_kinds

    out_stream = stream or sys.stdout
    script = Path(script).resolve()
    target = Path(out).resolve() if out else script.parent / '__pyrust__'
    rustc = toolchain.find_rustc()
    if not rustc:
        raise toolchain.not_found_error()
    toolchain.rustc_version(rustc)
    old_path = sys.path[:]
    try:
        sys.path.insert(0, str(script.parent))
        namespace = runpy.run_path(str(script), run_name='__pyrust_build__')
    finally:
        sys.path[:] = old_path
    target.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(target)
    rows = []
    for value in namespace.values():
        fn = getattr(value, 'python', value)
        if not inspect.isfunction(fn) or Path(fn.__code__.co_filename).resolve() != script:
            continue
        try:
            default_kinds(fn)
            source = Compiler(fn).generate()
        except Exception as exc:
            rows.append({'function': fn.__qualname__, 'ok': False, 'detail': str(exc).splitlines()[0]})
            continue
        key = source_key(source)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as work:
            try:
                built = compile_library(Path(work), key, rustc, source)
            except Exception as exc:
                rows.append({'function': fn.__qualname__, 'ok': False, 'detail': str(exc).splitlines()[0]})
                continue
            name = f'{fn.__name__}-{key[:16]}{SUFFIX}'
            shutil.copyfile(built, target / name)
        manifest[key] = {
            'file': name,
            'sha256': digest(target / name),
            'function': fn.__qualname__,
            'platform': platform_tag(),
        }
        rows.append({'function': fn.__qualname__, 'ok': True, 'detail': name})
    (target / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    for row in rows:
        mark = 'built ' if row['ok'] else 'skipped'
        print(f'{mark} {row["function"]}: {row["detail"]}', file=out_stream)
    print(
        f'{sum(r["ok"] for r in rows)} of {len(rows)} functions built into {target} for {platform_tag()}',
        file=out_stream,
    )
    return rows
