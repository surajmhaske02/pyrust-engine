"""The @optimize decorator: specialization, native dispatch, fallback and metrics."""

import array as _array
import ast
import ctypes
import functools
import hashlib
import inspect
import platform
import random as _random
import textwrap
import threading
import time
import types
import warnings
from collections import Counter
from typing import Any, Callable, Optional, ParamSpec, Protocol, TypeVar, overload

from . import packing, prebuilt, toolchain
from .cache import cache_root, cached_path, compile_library, process_lock
from .compiler import (
    ARRAY_INFO,
    KNOWN_BUILTINS,
    LIST_KINDS,
    Compiler,
    Unsupported,
    default_kinds,
    is_packed,
    kind_of_value,
    tuple_parts,
)
from .toolchain import ToolchainError

_STATUS = {
    1: 'integer overflow',
    2: 'zero divisor or zero range step',
    3: 'array index out of bounds',
    4: 'native iteration budget exceeded',
    5: 'no scalar return',
    6: 'value outside the native fast path (math domain, float overflow, negative exponent, huge int, '
    'or recursion deeper than 300 levels)',
    7: 'interrupted',
    8: 'internal error in generated code (replayed in Python; please report it)',
}
_TYPE_NAMES = {
    'i64': 'int',
    'f64': 'float',
    'bool': 'bool',
    'array': 'F64Array',
    'iarray': 'I64Array',
    'array2': 'F64Array2',
    'iarray2': 'I64Array2',
    'flist': 'list[float]',
    'ilist': 'list[int]',
    'series': 'pandas.Series[float64]',
    'iseries': 'pandas.Series[int64]',
    'fseries': 'pandas.Series[float32]',
    'sseries': 'pandas.Series[str]',
    'plseries': 'polars.Series[float]',
    'iplseries': 'polars.Series[int]',
    'plsseries': 'polars.Series[str]',
    'str': 'str',
    'bytes': 'bytes',
    'char': 'str (one character)',
    'strv': 'str',
    'slist': 'list[str]',
    'f32': 'numpy.float32',
    'farray': 'F32Array',
    'farray2': 'F32Array2',
    'none': 'None',
}
_ITEM_NAMES = {'strv': 'str', 'i64': 'int', 'f64': 'float', 'bool': 'bool'}
_CTYPE = {
    'i64': ctypes.c_int64,
    'f64': ctypes.c_double,
    'bool': ctypes.c_bool,
    'char': ctypes.c_uint32,
    'f32': ctypes.c_float,
}
_ELEMENT_CTYPE = {'f64': ctypes.c_double, 'i64': ctypes.c_int64, 'f32': ctypes.c_float}
_LIST_TYPECODE = {'flist': 'd', 'ilist': 'q'}
_SERIES_DTYPE = {
    'series': 'float64',
    'iseries': 'int64',
    'fseries': 'float32',
    'plseries': 'float64',
    'iplseries': 'int64',
}
_NUMPY_DTYPE = {'f64': 'float64', 'i64': 'int64', 'f32': 'float32'}


def _fast_result(ctype):
    class Result(ctypes.Structure):
        _fields_ = [('value', ctype), ('status', ctypes.c_int32)]

    return Result


_FAST_RESULT = {
    'i64': _fast_result(ctypes.c_int64),
    'f64': _fast_result(ctypes.c_double),
    'bool': _fast_result(ctypes.c_bool),
    'none': _fast_result(ctypes.c_uint8),
}
FAST_BUDGET = 50_000_000  # iterations run inline; longer calls move to an interruptible thread
UNLIMITED = 2**64 - 1
MAX_SPECIALIZATIONS = 16


def _type_name(kind):
    if kind.startswith('dict('):
        k, v = kind[5:-1].split(',')
        return f'dict[{_ITEM_NAMES[k]}, {_ITEM_NAMES[v]}]'
    if kind.startswith('set('):
        return f'set[{_ITEM_NAMES[kind[4:-1]]}]'
    parts = tuple_parts(kind)
    if parts:
        return 'tuple[' + ', '.join(_TYPE_NAMES[p] for p in parts) + ']'
    return _TYPE_NAMES.get(kind, kind)


class OptimizationError(RuntimeError):
    pass


class PyRustFallbackWarning(RuntimeWarning):
    """A decorated function ran as plain Python instead of native code."""


class RustToolchainWarning(PyRustFallbackWarning):
    """Rust is missing or broken on this machine, so every decorated function runs as plain Python."""


# Toolchain problems affect every function, so they are reported once per process.
_TOOLCHAIN_WARNED: set[str] = set()

LOOP_FREE_REASON = (
    'loop-free function: a native call costs more than the work it saves, '
    'so it runs as plain Python (pass native_loop_free=True to force Rust)'
)


def _is_loop_free(fn):
    """True when the function body has no loop and calls no other function.

    Such a function does a fixed, tiny amount of work per call, so native dispatch can
    never pay for itself. Unknown source is treated as not loop-free.
    """
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    except (OSError, TypeError, SyntaxError, IndentationError):
        return False
    if not tree.body or not isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    # Walk the body only: decorators such as @optimize(...) are calls too.
    for node in (n for statement in tree.body[0].body for n in ast.walk(statement)):
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            return False
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'sum':
            return False
        # A call to another (possibly looping) function is real work that can be compiled.
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id not in KNOWN_BUILTINS
            and isinstance(fn.__globals__.get(node.func.id), types.FunctionType)
        ):
            return False
    return True


def _describe(fn, max_iterations, kinds=None):
    """Try compiling; return (ok, details dict). Never raises."""
    try:
        compiler = Compiler(fn, max_iterations or FAST_BUDGET, kinds=kinds)
        compiler.generate()
    except Exception as exc:  # Unsupported, missing source, ...
        return False, {'reason': str(exc)}
    return True, {
        'parameters': {p: _TYPE_NAMES[compiler.types[p]] for p in compiler.params},
        'returns': _type_name(compiler.return_type),
        'calls': list(compiler.callees),
        'parallel': compiler.program['parallel'],
    }


def _explain_text(fn, max_iterations, *, loop_free=False, backend=None, reasons=None, specializations=()):
    ok, info = _describe(fn, max_iterations)
    name = getattr(fn, '__qualname__', fn.__name__)
    lines = []
    if loop_free:
        lines.append(f'{name}: runs as plain Python on purpose (no loop, so native code cannot be faster).')
        lines.append(
            '  Use @optimize(native_loop_free=True) to compile it anyway.'
            if ok
            else f'  It also could not be compiled: {info["reason"]}'
        )
    elif ok:
        lines.append(f'{name}: compiles to Rust.')
    else:
        lines.append(f'{name}: runs as plain Python. It cannot be compiled:')
        lines += ['  ' + line for line in info['reason'].splitlines()]
    if ok:
        params = ', '.join(f'{p}: {t}' for p, t in info['parameters'].items()) or 'none'
        lines.append(f'  parameters: {params}')
        lines.append(f'  returns: {info["returns"]}')
        if info['calls']:
            lines.append(f'  compiled together with: {", ".join(info["calls"])}')
        if info['parallel']:
            lines.append('  parallel: prange loop runs on several CPU cores')
        rust = toolchain.check_rust()
        if not rust['ok']:
            lines.append(f'  warning: {rust["problem"]} Calls run as plain Python until it is fixed:')
            lines += [f'    {step}' for step in rust['fix']]
    if specializations:
        lines.append(
            '  compiled for argument types: '
            + '; '.join('(' + ', '.join(_TYPE_NAMES.get(k, k) for k in spec) + ')' for spec in specializations)
        )
    if backend:
        lines.append(f'  current backend: {backend}')
    if reasons:
        lines.append('  fallbacks so far: ' + '; '.join(f'{r.splitlines()[0]} ({c}x)' for r, c in reasons.items()))
    return '\n'.join(lines)


def _plain_python(fn, max_iterations):
    """Zero-overhead result for loop-free functions: a real Python function, not a wrapper.

    Calls cost exactly what the original costs. Call counters are not tracked because
    there is no wrapper to count them.
    """
    clone = types.FunctionType(fn.__code__, fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__)
    clone.__kwdefaults__ = dict(fn.__kwdefaults__) if fn.__kwdefaults__ else None
    functools.update_wrapper(clone, fn)

    def report():
        return {
            'backend': 'python',
            'reason': LOOP_FREE_REASON,
            'compile_error': None,
            'native_calls': 0,
            'fallback_calls': 0,
            'calls': 0,
            'cache_hit': False,
            'cache_rebuilt': False,
            'compile_seconds': 0.0,
            'native_seconds': 0.0,
            'fallback_seconds': 0.0,
            'wrapper_seconds': 0.0,
            'compile_attempts': 0,
            'fallback_reasons': {},
            'fallback_rate': 0.0,
            'native_average_seconds': None,
            'fallback_average_seconds': None,
            'dispatch': 'python-direct',
            'call_counting': False,
            'toolchain_problem': False,
            'specializations': [],
            'long_calls': 0,
        }

    def prepare(*, background=False):
        return report()

    clone.python = fn
    clone.report = report
    clone.prepare = prepare
    clone.reset = report
    clone.rust_source = lambda: Compiler(fn, max_iterations or FAST_BUDGET).generate()
    clone.explain = lambda: _explain_text(fn, max_iterations, loop_free=True)
    return clone


P = ParamSpec('P')
R = TypeVar('R')


class OptimizedFunction(Protocol[P, R]):
    """What @optimize returns: callable like the original, plus PyRust helpers."""

    python: Callable[P, R]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...
    def report(self) -> dict[str, Any]: ...
    def prepare(self, *, background: bool = False) -> dict[str, Any]: ...
    def reset(self) -> dict[str, Any]: ...
    def explain(self) -> str: ...
    def rust_source(self) -> str: ...


class _Native:
    """One compiled specialization: the library, its entry point and how to call it."""

    __slots__ = (
        'lib',
        'entry',
        'kinds',
        'guards',
        'code_guards',
        'written',
        'returns',
        'out_types',
        'free',
        'has_arrays',
        'any_written',
        'random',
        'fast',
        'checks',
        'void',
    )

    def __init__(self, lib, description):
        self.lib = lib
        self.entry = lib.pyrust_run
        self.kinds = tuple(description.types[p] for p in description.params)
        self.guards = description.guards
        self.code_guards = description.code_guards
        self.written = tuple(p in description.written for p in description.params)
        self.any_written = any(self.written)
        self.has_arrays = any(k in ARRAY_INFO for k in self.kinds)
        self.random = description.program['random']
        self.returns = description.return_type
        argtypes = []
        for kind in self.kinds:
            if kind in ARRAY_INFO:
                argtypes += [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
            elif is_packed(kind):
                argtypes += [ctypes.c_void_p, ctypes.c_size_t]
            else:
                argtypes.append(_CTYPE[kind])
        self.has_arrays = self.has_arrays or any(is_packed(k) for k in self.kinds)
        rt = self.returns
        self.free = None
        if is_packed(rt):
            self.out_types = (ctypes.c_void_p, ctypes.c_size_t)
            self.free = lib.pyrust_free_u8
            self.free.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            self.free.restype = None
        elif rt in ARRAY_INFO:
            element = ARRAY_INFO[rt][0]
            self.out_types = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t)
            self.free = getattr(lib, f'pyrust_free_{element}')
            self.free.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            self.free.restype = None
        elif tuple_parts(rt):
            self.out_types = tuple(_CTYPE[p] for p in tuple_parts(rt))
        elif rt == 'none':
            self.out_types = ()
        else:
            self.out_types = (_CTYPE[rt],)
        argtypes += [ctypes.POINTER(t) for t in self.out_types]
        argtypes += [ctypes.c_uint64, ctypes.c_void_p]
        if self.random:
            argtypes += [ctypes.c_void_p, ctypes.c_void_p]
        self.entry.argtypes = argtypes
        self.entry.restype = ctypes.c_int32
        self.checks = bool(self.guards or self.code_guards)
        self.void = rt == 'none'
        self.fast = None
        if description.fast_eligible():
            # Scalars in, one scalar out: value and status come back together in a struct.
            fast = lib.pyrust_fast
            fast.argtypes = [*argtypes[: len(self.kinds)], ctypes.c_uint64]  # + iteration budget
            fast.restype = _FAST_RESULT[rt]
            self.fast = fast

    def result(self, outs):
        rt = self.returns
        if rt == 'none':
            return None
        if is_packed(rt):
            ptr, n = (o.value for o in outs)
            n = n or 0
            try:
                data = ctypes.string_at(ptr, n) if n else b''
            finally:
                self.free(ptr, n)
            return packing.unpack(rt, data)
        if rt == 'f32':
            import numpy as np

            return np.float32(outs[0].value)
        if rt in LIST_KINDS:
            element = ARRAY_INFO[rt][0]
            ptr, n, _, _ = (o.value for o in outs)
            n = n or 0
            try:
                return (_ELEMENT_CTYPE[element] * n).from_address(ptr)[:] if n else []
            finally:
                self.free(ptr, n)
        if rt in ARRAY_INFO:
            import numpy as np

            element, dims = ARRAY_INFO[rt]
            ptr, n, rows, cols = (o.value for o in outs)
            n = n or 0
            try:
                if n:
                    buffer = (_ELEMENT_CTYPE[element] * n).from_address(ptr)
                    data = np.frombuffer(buffer, dtype=_NUMPY_DTYPE[element]).copy()
                else:
                    data = np.zeros(0, dtype=_NUMPY_DTYPE[element])
            finally:
                self.free(ptr, n)
            return data.reshape((rows or 0, cols or 0)) if dims == 2 else data
        if tuple_parts(rt):
            return tuple(o.value for o in outs)
        if rt == 'char':
            return chr(outs[0].value)
        return outs[0].value


class _RandomState:
    """Python's shared random generator state, lent to one native call.

    Native code continues the exact Mersenne Twister sequence of the random module;
    the new state is written back only when the call succeeds, so a replay in Python
    starts from the same state.
    """

    __slots__ = ('state', 'words', 'gauss', 'pointers')

    def __init__(self):
        state = _random._inst.getstate()
        if state[0] != 3 or len(state[1]) != 625:
            raise Unsupported('unexpected random module state format')
        self.state = state
        self.words = (ctypes.c_uint32 * 625)(*state[1])
        self.gauss = (ctypes.c_double * 2)()
        self.restart()

    def restart(self):
        self.words[:] = self.state[1]
        gauss_next = self.state[2]
        self.gauss[0] = 0.0 if gauss_next is None else gauss_next
        self.gauss[1] = 0.0 if gauss_next is None else 1.0
        self.pointers = (ctypes.addressof(self.words), ctypes.addressof(self.gauss))
        return self.pointers

    def publish(self):
        _random._inst.setstate((3, tuple(self.words), self.gauss[0] if self.gauss[1] else None))


class _Failure:
    __slots__ = ('reason', 'toolchain')
    fast = None  # never takes the fast call path

    def __init__(self, reason, toolchain_problem):
        self.reason = reason
        self.toolchain = toolchain_problem


def _unsupported_value(value):
    if type(value) is int:
        return 'int does not fit in 64 bits'
    import sys

    np = sys.modules.get('numpy')
    if np is not None and isinstance(value, np.ndarray):
        return (
            f'expected a float64 or int64 NumPy array with 1 or 2 dimensions '
            f'(got {type(value).__name__} of {value.dtype}, {value.ndim}-D)'
        )
    if type(value) is list:
        return 'list elements must be all floats or all 64-bit ints'
    pd = sys.modules.get('pandas')
    if pd is not None and isinstance(value, pd.Series):
        return (
            'pandas Series must have a float64, int64, float32 or str dtype with no missing values '
            'and the default 0..n-1 index (use .reset_index(drop=True))'
        )
    pl = sys.modules.get('polars')
    if pl is not None and isinstance(value, pl.Series):
        return 'polars Series must hold floats, ints (up to 64-bit signed) or str, with no nulls'
    return f'{type(value).__name__} values are not compiled'


_SCALAR_KINDS = {float: 'f64', bool: 'bool'}
_I64_MIN = -9223372036854775808
_I64_MAX = 9223372036854775807


def _kinds_of(values):
    out = []
    for value in values:
        t = type(value)
        if t is int:
            out.append('i64' if _I64_MIN <= value <= _I64_MAX else None)
        else:
            out.append(_SCALAR_KINDS.get(t) or kind_of_value(value))
    return tuple(out)


@overload
def optimize(
    function: Callable[P, R],
    *,
    backend: str = ...,
    fallback: bool = ...,
    cache_dir: Optional[str] = ...,
    max_iterations: Optional[int] = ...,
    native_loop_free: bool = ...,
    warn_on_fallback: bool = ...,
    timing: bool = ...,
) -> OptimizedFunction[P, R]: ...
@overload
def optimize(
    function: None = None,
    *,
    backend: str = ...,
    fallback: bool = ...,
    cache_dir: Optional[str] = ...,
    max_iterations: Optional[int] = ...,
    native_loop_free: bool = ...,
    warn_on_fallback: bool = ...,
    timing: bool = ...,
) -> Callable[[Callable[P, R]], OptimizedFunction[P, R]]: ...
def optimize(
    function=None,
    *,
    backend='rust',
    fallback=True,
    cache_dir=None,
    max_iterations=None,
    native_loop_free=False,
    warn_on_fallback=True,
    timing=False,
):
    """Compile a numeric Python function to Rust, falling back to Python when needed.

    max_iterations: None (default) means no limit: calls that run long move to a
    background thread that Ctrl-C can stop. An int sets a hard budget of loop
    iterations and calls; a call that exceeds it is replayed in Python.

    native_loop_free: loop-free functions (no loop, no calls) run as plain Python by
    default, because native dispatch overhead is larger than their work.

    warn_on_fallback: emit a PyRustFallbackWarning the first time each distinct reason
    makes a call run as plain Python.

    timing: record per-call native/fallback timings in report() (small overhead).
    """
    if backend != 'rust':
        raise ValueError('Only backend="rust" is supported')
    if max_iterations is not None and (type(max_iterations) is not int or not 1 <= max_iterations < 2**64):
        raise ValueError('max_iterations must be None or a positive unsigned 64-bit integer')

    def decorate(fn):
        if not isinstance(fn, types.FunctionType):
            raise TypeError('Decorate an ordinary Python function')
        if not native_loop_free and _is_loop_free(fn):
            return _plain_python(fn, max_iterations)
        return _Optimized(
            fn,
            fallback=fallback,
            cache_dir=cache_dir,
            max_iterations=max_iterations,
            warn_on_fallback=warn_on_fallback,
            timing=timing,
        ).wrapper

    return decorate(function) if function is not None else decorate


class _Optimized:
    def __init__(self, fn, *, fallback, cache_dir, max_iterations, warn_on_fallback, timing):
        self.fn = fn
        self.fallback = fallback
        self.cache_dir = cache_dir
        self.max_iterations = max_iterations
        self.warn_on_fallback = warn_on_fallback
        self.timing = timing
        self.signature = inspect.signature(fn)
        self.n_params = len(self.signature.parameters)
        self.original = (fn.__code__, fn.__defaults__, fn.__kwdefaults__)
        self.compile_lock = threading.RLock()
        self.metrics_lock = threading.Lock()
        self.state = {
            'backend': 'pending',
            'reason': None,
            'compile_error': None,
            'native_calls': 0,
            'fallback_calls': 0,
            'calls': 0,
            'cache_hit': False,
            'cache_rebuilt': False,
            'compile_seconds': 0.0,
            'native_seconds': 0.0,
            'fallback_seconds': 0.0,
            'wrapper_seconds': 0.0,
            'compile_attempts': 0,
            'toolchain_problem': False,
            'long_calls': 0,
        }
        self.reasons = Counter()
        self.warned = set()
        self.specs = {}  # kinds tuple -> _Native or _Failure
        self.sources = {}  # kinds tuple -> Rust source
        self.worker = None
        try:
            self.default_kinds = tuple(default_kinds(fn).values())
        except Unsupported:
            self.default_kinds = None
        self.wrapper = self.build_wrapper()

    # ------------------------------------------------------------------ metrics
    def update(self, **values):
        with self.metrics_lock:
            self.state.update(values)

    def report(self):
        with self.metrics_lock:
            data = dict(self.state)
            data['fallback_reasons'] = dict(self.reasons)
            data['specializations'] = [list(k) for k, v in self.specs.items() if isinstance(v, _Native)]
        calls = data['calls']
        data['fallback_rate'] = data['fallback_calls'] / calls if calls else 0.0
        data['native_average_seconds'] = (
            data['native_seconds'] / data['native_calls'] if data['native_calls'] and self.timing else None
        )
        data['fallback_average_seconds'] = (
            data['fallback_seconds'] / data['fallback_calls'] if data['fallback_calls'] else None
        )
        data['dispatch'] = 'wrapper'
        data['call_counting'] = True
        return data

    # --------------------------------------------------------------- fallback
    def python_call(self, reason, args, kwargs, toolchain_problem=False):
        fn = self.fn
        with self.metrics_lock:
            self.state['reason'] = reason
            if toolchain_problem:
                first = reason not in _TOOLCHAIN_WARNED
                _TOOLCHAIN_WARNED.add(reason)
            else:
                first = reason not in self.warned
                self.warned.add(reason)
        if not self.fallback:
            raise OptimizationError(reason)
        if self.warn_on_fallback and first and reason != 'background compilation in progress':
            if toolchain_problem:
                warnings.warn(
                    f'{fn.__qualname__} ran as plain Python: {reason}\n'
                    '(Shown once: this applies to every @optimize function until Rust works.)',
                    RustToolchainWarning,
                    stacklevel=4,
                )
            else:
                warnings.warn(f'{fn.__qualname__} ran as plain Python: {reason}', PyRustFallbackWarning, stacklevel=4)
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            with self.metrics_lock:
                self.state['fallback_calls'] += 1
                self.state['fallback_seconds'] += time.perf_counter() - start
                self.reasons[reason] += 1

    # ---------------------------------------------------------------- compile
    def compile(self, kinds):
        """Compile (or load) the specialization for these argument kinds. Never raises."""
        with self.compile_lock:
            existing = self.specs.get(kinds)
            if existing is not None:
                return existing
            start = time.perf_counter()
            with self.metrics_lock:
                self.state['compile_attempts'] += 1
                self.state['backend'] = 'compiling'
            try:
                if len(self.specs) >= MAX_SPECIALIZATIONS:
                    raise Unsupported(f'more than {MAX_SPECIALIZATIONS} different argument type combinations')
                fn = self.fn
                description = Compiler(
                    fn, self.max_iterations or FAST_BUDGET, kinds=dict(zip(self.signature.parameters, kinds))
                )
                source = description.generate()
                self.sources[kinds] = source
                hit = rebuilt = False
                path = prebuilt.find(fn, source)
                if path is not None:
                    native = _Native(ctypes.CDLL(str(path)), description)
                    hit = True
                else:
                    compiler = toolchain.find_rustc()
                    if not compiler:
                        raise toolchain.not_found_error()
                    version = toolchain.rustc_version(compiler)
                    key = hashlib.sha256((source + version + platform.platform() + 'cabi-07').encode()).hexdigest()
                    root = cache_root(self.cache_dir)
                    with process_lock(root / (key + '.lock')):
                        path = cached_path(root, key)
                        hit = path is not None
                        rebuilt = (root / (key + '.json')).exists() and not hit
                        if path is None:
                            path = compile_library(root, key, compiler, source)
                        try:
                            native = _Native(ctypes.CDLL(str(path)), description)
                        except (OSError, AttributeError):
                            if not hit:
                                raise
                            # A valid hash does not guarantee loadability or a valid ABI symbol.
                            path = compile_library(root, key, compiler, source)
                            native = _Native(ctypes.CDLL(str(path)), description)
                            rebuilt, hit = True, False
                self.specs[kinds] = native
                self.update(
                    backend='rust',
                    reason=None,
                    compile_error=None,
                    cache_hit=hit,
                    cache_rebuilt=rebuilt,
                    toolchain_problem=False,
                )
                return native
            except Exception as exc:  # any compiler/loader failure means Python fallback
                failure = _Failure(str(exc), isinstance(exc, ToolchainError))
                self.specs[kinds] = failure
                self.update(
                    backend='python', reason=str(exc), compile_error=str(exc), toolchain_problem=failure.toolchain
                )
                return failure
            finally:
                with self.metrics_lock:
                    self.state['compile_seconds'] += time.perf_counter() - start

    def prepare(self, *, background=False):
        kinds = self.default_kinds
        if kinds is None:
            reason = _describe(self.fn, self.max_iterations)[1].get('reason', 'unsupported annotations')
            self.update(backend='python', reason=reason, compile_error=reason)
            if not self.fallback:
                raise OptimizationError(reason)
            return self.report()
        if background:
            with self.compile_lock:
                if kinds not in self.specs and (self.worker is None or not self.worker.is_alive()):
                    self.worker = threading.Thread(
                        target=self.compile, args=(kinds,), name='pyrust-compile', daemon=True
                    )
                    self.worker.start()
            return self.report()
        result = self.compile(kinds)
        if isinstance(result, _Failure) and not self.fallback:
            raise OptimizationError(result.reason)
        return self.report()

    def reset(self):
        active = self.worker
        if active is not None and active.is_alive():
            active.join()
        with self.compile_lock:
            self.specs.clear()
            self.sources.clear()
            self.worker = None
            self.update(
                backend='pending',
                reason=None,
                compile_error=None,
                cache_hit=False,
                cache_rebuilt=False,
                toolchain_problem=False,
            )
        return self.report()

    # ------------------------------------------------------------------- call
    def convert(self, values, native):
        """Native arguments, plus objects to keep alive and lists to copy back after success."""
        args = []
        keep = []
        copies = []
        lists = []
        ranges = [] if native.any_written else None
        for name, kind, value, writes in zip(self.signature.parameters, native.kinds, values, native.written):
            if kind in LIST_KINDS:
                # Lists are copied into a typed buffer; written lists get the new values back.
                if any(other is value and (writes or other_writes) for other, other_writes in lists):
                    raise Unsupported(f'{name}: the same list is passed twice and written')
                lists.append((value, writes))
                buffer = _array.array(_LIST_TYPECODE[kind], value)
                address, count = buffer.buffer_info()
                args += [address, count, 1]
                keep.append(buffer)
                if writes:
                    copies.append((value, buffer))
            elif kind in _SERIES_DTYPE:
                data = value.to_numpy()
                if data.dtype != _SERIES_DTYPE[kind]:
                    data = data.astype(_SERIES_DTYPE[kind])  # polars ints/float32: exact widening
                flags = data.flags
                if not (data.dtype.isnative and flags.c_contiguous and flags.aligned):
                    import numpy as np

                    data = np.ascontiguousarray(data)  # read-only, so a copy is fine
                address = data.ctypes.data
                args += [address, data.shape[0], 1]
                keep.append(data)
                if ranges is not None and data.size:
                    ranges.append((address, address + data.nbytes, False))
            elif kind == 'str':
                raw = value.encode('utf-32-le', 'surrogatepass')
                args += [ctypes.cast(ctypes.c_char_p(raw), ctypes.c_void_p).value, len(value), 1]
                keep.append(raw)
            elif kind == 'bytes':
                args += [ctypes.cast(ctypes.c_char_p(value), ctypes.c_void_p).value, len(value), 1]
            elif is_packed(kind):
                if kind == 'slist' and type(value) is not list:
                    value = value.to_list()  # pandas / polars Series of str
                data = packing.pack(kind, value)
                args += [ctypes.cast(ctypes.c_char_p(data), ctypes.c_void_p).value, len(data)]
                keep.append(data)
            elif kind in ARRAY_INFO:
                flags = value.flags
                if not (value.dtype.isnative and flags.c_contiguous and flags.aligned):
                    raise Unsupported(
                        f'{name}: expected aligned contiguous native {_NUMPY_DTYPE[ARRAY_INFO[kind][0]]} array'
                    )
                if writes and not flags.writeable:
                    raise Unsupported(f'{name}: array is read-only')
                address = value.ctypes.data
                if ARRAY_INFO[kind][1] == 1:
                    args += [address, value.shape[0], 1]
                else:
                    args += [address, value.shape[0], value.shape[1]]
                if ranges is not None and value.size:
                    ranges.append((address, address + value.nbytes, writes))
            else:
                args.append(value)
        if ranges and len(ranges) > 1:
            # A written array that shares memory with another argument would make undo
            # ambiguous, so those calls run in Python.
            for i, (a0, a1, wa) in enumerate(ranges):
                for b0, b1, wb in ranges[i + 1 :]:
                    if (wa or wb) and a0 < b1 and b0 < a1:
                        raise Unsupported('array arguments share memory with an array that is written')
        return args, keep, copies

    def run_long(self, native, args, refs, rng):
        """Re-run a call that exceeded the inline budget in a thread, with no budget and
        Ctrl-C support (the native loop checks a cancel flag)."""
        flag = ctypes.c_uint8(0)
        result = {}
        extra = rng.restart() if rng is not None else ()

        def runner():
            result['status'] = native.entry(*args, *refs, UNLIMITED, ctypes.addressof(flag), *extra)

        thread = threading.Thread(target=runner, name='pyrust-long-call', daemon=True)
        thread.start()
        try:
            while thread.is_alive():
                thread.join(0.05)
        except BaseException:
            flag.value = 1
            thread.join()
            raise
        with self.metrics_lock:
            self.state['long_calls'] += 1
        return result['status']

    def build_wrapper(self):
        fn = self.fn
        original = self.original
        n_params = self.n_params
        signature = self.signature
        specs = self.specs
        state = self.state
        metrics_lock = self.metrics_lock
        python_call = self.python_call
        timing = self.timing
        budget = self.max_iterations or FAST_BUDGET
        strict_budget = self.max_iterations is not None
        perf_counter = time.perf_counter
        byref = ctypes.byref

        def general(args, kwargs, first_status=None, known_kinds=None):
            start = perf_counter() if timing else 0.0
            native_elapsed = None
            try:
                if (
                    fn.__code__ is not original[0]
                    or fn.__defaults__ is not original[1]
                    or fn.__kwdefaults__ is not original[2]
                ):
                    return python_call('function code or defaults changed; create a new wrapper', args, kwargs)
                if kwargs or len(args) != n_params:
                    # Slow path for keywords/defaults; raises CPython's TypeError for bad calls.
                    bound = signature.bind(*args, **kwargs)
                    bound.apply_defaults()
                    values = tuple(bound.arguments.values())
                else:
                    values = args  # fast path: exact positional call
                kinds = known_kinds if known_kinds is not None else _kinds_of(values)
                native = specs.get(kinds)
                if native is None:
                    if None in kinds:
                        position = kinds.index(None)
                        bad = list(signature.parameters)[position]
                        value = values[position]
                        detail = _unsupported_value(value)
                        return python_call(f'{bad}: {detail}', args, kwargs)
                    worker = self.worker
                    if worker is not None and worker.is_alive():
                        return python_call('background compilation in progress', args, kwargs)
                    native = self.compile(kinds)
                if native.__class__ is _Failure:
                    return python_call(native.reason, args, kwargs, toolchain_problem=native.toolchain)
                for namespace, builtin_ns, name, expected in native.guards:
                    if namespace.get(name, builtin_ns.get(name)) is not expected:
                        return python_call(f'{name} has been rebound', args, kwargs)
                for callee, code, defaults in native.code_guards:
                    if callee.__code__ is not code or callee.__defaults__ is not defaults:
                        return python_call(f'code of called function {callee.__qualname__} changed', args, kwargs)
                copies = None
                if native.has_arrays:
                    try:
                        native_args, keep, copies = self.convert(values, native)
                    except Unsupported as exc:
                        return python_call(str(exc), args, kwargs)
                else:
                    native_args = values
                outs = [t() for t in native.out_types]
                refs = [byref(o) for o in outs]
                rng = None
                extra = ()
                if native.random:
                    try:
                        rng = _RandomState()
                    except Unsupported as exc:
                        return python_call(str(exc), args, kwargs)
                    extra = rng.pointers
                t0 = perf_counter() if timing else 0.0
                if first_status is None:
                    status = native.entry(*native_args, *refs, budget, None, *extra)
                else:
                    status = first_status  # the fast entry point already ran this call and failed
                if status == 4 and not strict_budget:
                    if copies:  # lists are not undone natively: start again from the originals
                        for target, buffer in copies:
                            buffer[:] = _array.array(buffer.typecode, target)
                    status = self.run_long(native, native_args, refs, rng)
                elapsed = perf_counter() - t0 if timing else 0.0
                if status:
                    with metrics_lock:
                        state['native_calls'] += 1
                        state['native_seconds'] += elapsed
                    if status == 7:
                        raise KeyboardInterrupt
                    return python_call(_STATUS.get(status, 'unknown native guard failure'), args, kwargs)
                native_elapsed = elapsed
                if copies:
                    for target, buffer in copies:
                        target[:] = buffer.tolist()
                if rng is not None:
                    rng.publish()
                return native.result(outs)
            finally:
                with metrics_lock:
                    state['calls'] += 1
                    if timing:
                        state['wrapper_seconds'] += perf_counter() - start
                    if native_elapsed is not None:
                        state['native_calls'] += 1
                        state['native_seconds'] += native_elapsed
                        state['reason'] = None
                        state['backend'] = 'rust'

        code0, defaults0, kwdefaults0 = original

        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            # Fast path: exact positional call of a scalar specialization (no out-parameter
            # objects, one ctypes call returning value and status together).
            if (
                not timing
                and not kwargs
                and len(args) == n_params
                and fn.__code__ is code0
                and fn.__defaults__ is defaults0
                and fn.__kwdefaults__ is kwdefaults0
            ):
                kinds = _kinds_of(args)
                native = specs.get(kinds)
                if native is not None and native.fast is not None:
                    if native.checks:
                        for namespace, builtin_ns, name, expected in native.guards:
                            if namespace.get(name, builtin_ns.get(name)) is not expected:
                                return general(args, kwargs, known_kinds=kinds)
                        for callee, code, defaults in native.code_guards:
                            if callee.__code__ is not code or callee.__defaults__ is not defaults:
                                return general(args, kwargs, known_kinds=kinds)
                    result = native.fast(*args, budget)
                    status = result.status
                    if status:
                        return general(args, kwargs, status, kinds)
                    with metrics_lock:
                        state['calls'] += 1
                        state['native_calls'] += 1
                        if state['reason'] is not None or state['backend'] != 'rust':
                            state['reason'] = None
                            state['backend'] = 'rust'
                    return None if native.void else result.value
                return general(args, kwargs, known_kinds=kinds)
            return general(args, kwargs)

        def rust_source(kinds=None):
            if kinds is None:
                if self.sources:
                    return next(iter(self.sources.values()))
                return Compiler(fn, budget).generate()
            return Compiler(fn, budget, kinds=dict(zip(signature.parameters, kinds))).generate()

        wrapped.python = fn
        wrapped.report = self.report
        wrapped.prepare = self.prepare
        wrapped.reset = self.reset
        wrapped.rust_source = rust_source
        wrapped.explain = lambda: _explain_text(
            fn,
            self.max_iterations,
            backend=self.report()['backend'],
            reasons=self.report()['fallback_reasons'],
            specializations=[k for k, v in specs.items() if isinstance(v, _Native)],
        )
        return wrapped
