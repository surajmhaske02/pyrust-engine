from .cache import prune_cache
from .compiler import F32Array, F32Array2, F64Array, F64Array2, I64Array, I64Array2, Unsupported, prange
from .runtime import (
    OptimizationError,
    OptimizedFunction,
    PyRustFallbackWarning,
    RustToolchainWarning,
    on_fallback,
    optimize,
    print_stats,
    reset_stats,
    stats,
)
from .toolchain import check_rust, doctor
from .vectorize import vectorize

optimize.vectorize = vectorize  # type: ignore[attr-defined]  # @optimize.vectorize works too

__all__ = [
    'optimize',
    'vectorize',
    'prange',
    'OptimizationError',
    'OptimizedFunction',
    'PyRustFallbackWarning',
    'RustToolchainWarning',
    'check_rust',
    'doctor',
    'Unsupported',
    'F64Array',
    'I64Array',
    'F64Array2',
    'I64Array2',
    'F32Array',
    'F32Array2',
    'prune_cache',
    'stats',
    'print_stats',
    'reset_stats',
    'on_fallback',
]
__version__ = '0.7.7'


def load_ipython_extension(ipython):
    """`%load_ext pyrust_engine` adds %pyrust_check and %pyrust_benchmark."""
    from .ipython import load

    load(ipython)
