"""Lazy compilation with checked arithmetic and CPython fallback."""
import builtins
import ctypes
import functools
import hashlib
import inspect
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
from .compiler import Compiler, Unsupported

_BUILTIN_RANGE = range

class OptimizationError(RuntimeError):
    pass

def optimize(function=None, *, backend='rust', fallback=True, cache_dir=None):
    if backend != 'rust':
        raise ValueError('Only backend="rust" is supported')
    def decorate(fn):
        signature = inspect.signature(fn)
        lock = threading.Lock()
        state = {'backend': 'pending', 'reason': None, 'native_calls': 0, 'fallback_calls': 0, 'cache_hit': False}
        native = None
        library = None
        source = None
        attempted = False

        def python_call(reason, args, kwargs):
            state['reason'] = reason
            if not fallback:
                raise OptimizationError(reason)
            state['fallback_calls'] += 1
            return fn(*args, **kwargs)

        def compile_once():
            nonlocal native, library, source, attempted
            attempted = True
            try:
                source = Compiler(fn).generate()
                compiler = shutil.which('rustc')
                if not compiler:
                    raise Unsupported('rustc is not installed or not on PATH')
                version = subprocess.check_output([compiler, '-vV'], text=True, timeout=15)
                key = hashlib.sha256((source + version + platform.platform() + 'abi1-O3-ed2021').encode()).hexdigest()
                root = Path(cache_dir) if cache_dir else Path.home() / '.cache' / 'pyrust_engine'
                root.mkdir(parents=True, exist_ok=True, mode=0o700)
                suffix = '.dll' if sys.platform == 'win32' else '.dylib' if sys.platform == 'darwin' else '.so'
                target = root / (key + suffix)
                state['cache_hit'] = target.exists()
                if not target.exists():
                    with tempfile.TemporaryDirectory(dir=root) as temporary:
                        rs = Path(temporary) / 'kernel.rs'
                        output = Path(temporary) / ('kernel' + suffix)
                        rs.write_text(source)
                        result = subprocess.run([compiler, str(rs), '--crate-type', 'cdylib', '--edition', '2021',
                                                 '-C', 'opt-level=3', '-C', 'panic=abort', '-o', str(output)],
                                                capture_output=True, text=True, timeout=120)
                        if result.returncode:
                            raise Unsupported('Rust compilation failed: ' + result.stderr[-3000:])
                        os.replace(output, target)
                library = ctypes.CDLL(str(target.resolve()))
                native = library.pyrust_run
                native.argtypes = [ctypes.c_int64] * len(signature.parameters) + [ctypes.POINTER(ctypes.c_int64)]
                native.restype = ctypes.c_int32
                state.update(backend='rust', reason=None)
            except (Unsupported, OSError, TypeError, SyntaxError, subprocess.SubprocessError) as exc:
                state.update(backend='python', reason=str(exc))

        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            # Exact built-in ints only: bools and subclasses may have different operators.
            values = list(bound.arguments.values())
            if any(type(v) is not int or not -(2**63) <= v < 2**63 for v in values):
                return python_call('Arguments must be exact signed 64-bit Python ints', args, kwargs)
            # Check the function's actual builtins mapping, including custom exec environments.
            if fn.__globals__.get('range', fn.__builtins__.get('range')) is not _BUILTIN_RANGE:
                return python_call('range has been rebound', args, kwargs)
            with lock:
                if not attempted:
                    compile_once()
            if native is None:
                return python_call(state['reason'], args, kwargs)
            out = ctypes.c_int64()
            status = native(*values, ctypes.byref(out))
            if status:
                return python_call('Native integer overflow; replaying pure function in Python', args, kwargs)
            state['native_calls'] += 1
            state['reason'] = None
            return out.value

        wrapped.python = fn
        wrapped.report = lambda: dict(state)
        wrapped.rust_source = lambda: source if source is not None else Compiler(fn).generate()
        return wrapped
    return decorate(function) if function is not None else decorate
