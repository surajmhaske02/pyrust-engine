"""@vectorize: write a function for one value, run it over whole arrays in one native call.

    @vectorize
    def score(price: float, qty: int) -> float:
        return price * qty * (0.9 if qty > 10 else 1.0)

    score(prices, quantities)        # NumPy arrays, lists or pandas columns -> NumPy array
    score(df['price'], df['qty'])    # pandas columns -> pandas Series with the same index
    score(prices, 3)                 # plain numbers are used for every element

PyRust generates a loop that calls the function for each position and compiles both to
Rust together, so a tiny function no longer pays the cost of one native call per value.
Each element gets exactly the result the function gives in Python. Inputs PyRust cannot
compile (other dtypes, mixed lists, unsupported code) run the same loop in Python.
"""

import builtins
import functools
import itertools
import linecache
import sys
import threading

from .compiler import Compiler, kind_of_value, prange

_COUNTER = itertools.count()
_OUT = '_pyrust_out'  # the result array; fresh for every call, so it needs no undo information
_PYTHON = object()  # marks an argument only the Python loop can handle


def _numpy():
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - depends on the environment
        raise ImportError('vectorize needs NumPy: pip install "pyrust-engine[arrays]"') from None
    return np


class Vectorized:
    """A one-value function applied element by element over arrays, lists and pandas columns."""

    def __init__(self, function, parallel, options):
        self.python = getattr(function, 'python', function)
        functools.update_wrapper(self, self.python)
        self.parallel = parallel
        self.options = options
        self.arity = self.python.__code__.co_argcount
        self.kernels = {}  # which arguments are arrays -> compiled loop
        self.out_kinds = {}  # element kinds -> result kind ('f64', 'i64', 'bool') or None
        self.last = None
        self.lock = threading.Lock()

    # ------------------------------------------------------------------ inputs
    def column(self, value):
        """(numpy array, pandas index or None) for an array-like argument, None for a scalar.

        Returns (_PYTHON, value) for array-likes that only the Python loop can handle.
        """
        np = sys.modules.get('numpy')
        t = type(value)
        if t in (int, float, bool):
            return None
        if np is not None and t is np.ndarray:
            if value.ndim == 1 and value.dtype in (np.float64, np.int64):
                return np.ascontiguousarray(value), None
            return _PYTHON, value
        if t is list:
            kind = kind_of_value(value)
            np = _numpy()
            if kind == 'flist':
                return np.array(value, dtype=np.float64), None
            if kind == 'ilist':
                return np.array(value, dtype=np.int64), None
            return _PYTHON, value
        pd = sys.modules.get('pandas')
        if pd is not None and isinstance(value, pd.Series):
            data = value.to_numpy()
            if isinstance(value.dtype, np.dtype) and data.dtype in (np.float64, np.int64):
                return np.ascontiguousarray(data), value.index
            return _PYTHON, value
        return None

    def out_kind(self, kinds):
        """Result kind of the function for these element kinds, or None if it cannot compile."""
        if kinds not in self.out_kinds:
            try:
                compiler = Compiler(self.python, kinds=dict(zip(list(self.python.__code__.co_varnames), kinds)))
                compiler.generate()
                kind = compiler.return_type
                self.out_kinds[kinds] = kind if kind in ('f64', 'i64', 'bool') else None
            except Exception:
                self.out_kinds[kinds] = None
        return self.out_kinds[kinds]

    # ----------------------------------------------------------------- kernels
    def kernel(self, mask):
        with self.lock:
            existing = self.kernels.get(mask)
            if existing is not None:
                return existing
            from .runtime import optimize

            params = [f'x{k}' for k in range(self.arity)]
            call = ', '.join(f'x{k}[i]' if is_array else f'x{k}' for k, is_array in enumerate(mask))
            loop = 'prange' if self.parallel else 'range'
            name = '_pyrust_vectorized_' + self.python.__name__
            source = (
                f'def {name}({", ".join(params + [_OUT])}):\n'
                f'    for i in {loop}(len({_OUT})):\n'
                f'        {_OUT}[i] = _pyrust_function({call})\n'
            )
            filename = f'<pyrust-vectorize-{self.python.__qualname__}-{next(_COUNTER)}>'
            linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
            namespace = {
                '__builtins__': builtins,
                '__name__': '_pyrust_vectorize',
                '_pyrust_function': self.python,
                'prange': prange,
            }
            exec(compile(source, filename, 'exec'), namespace)
            options = {'warn_on_fallback': False, **self.options}
            compiled = optimize(namespace[name], **options)
            self.kernels[mask] = compiled
            return compiled

    # -------------------------------------------------------------------- call
    def __call__(self, *args):
        if len(args) != self.arity:
            return self.python(*args)  # raises Python's own TypeError
        columns = [self.column(a) for a in args]
        if all(c is None for c in columns):
            return self.python(*args)
        np = _numpy()
        lengths = {len(c[1] if c[0] is _PYTHON else c[0]) for c in columns if c is not None}
        if len(lengths) != 1:
            raise ValueError(f'{self.python.__name__}: array arguments have different lengths {sorted(lengths)}')
        n = lengths.pop()
        index = next((c[1] for c in columns if c is not None and c[0] is not _PYTHON and c[1] is not None), None)
        if index is None:
            pd = sys.modules.get('pandas')
            index = next((a.index for a in args if pd is not None and isinstance(a, pd.Series)), None)
        kinds = []
        for value, c in zip(args, columns):
            if c is None:
                kinds.append({int: 'i64', float: 'f64', bool: 'bool'}.get(type(value)))
            elif c[0] is _PYTHON:
                kinds.append(None)
            else:
                kinds.append('f64' if c[0].dtype == np.float64 else 'i64')
        result_kind = self.out_kind(tuple(kinds)) if None not in kinds else None
        if result_kind is None:
            out = self.python_loop(args, columns, n)
        else:
            out = np.empty(n, dtype=np.float64 if result_kind == 'f64' else np.int64)
            kernel = self.kernel(tuple(c is not None for c in columns))
            values = [a if c is None else c[0] for a, c in zip(args, columns)]
            kernel(*values, out)
            self.last = kernel
            if result_kind == 'bool':
                out = out.astype(bool)
        if index is not None:
            import pandas as pd

            return pd.Series(out, index=index)
        return out

    def python_loop(self, args, columns, n):
        np = _numpy()
        pick = []
        for value, c in zip(args, columns):
            if c is None:
                pick.append(lambda i, v=value: v)
            elif c[0] is _PYTHON:
                data = c[1].to_numpy() if hasattr(c[1], 'to_numpy') else c[1]
                pick.append(lambda i, d=data: d[i])
            else:
                pick.append(lambda i, d=c[0]: d[i])
        fn = self.python
        return np.array([fn(*(p(i) for p in pick)) for i in range(n)])

    # ----------------------------------------------------------------- helpers
    def report(self):
        """report() of the compiled loop used by the last call, or {} before the first call."""
        return self.last.report() if self.last is not None else {}

    def explain(self):
        from .runtime import _explain_text

        text = _explain_text(self.python, None)
        how = 'on several CPU cores' if self.parallel else 'in one native loop'
        return text + f'\n  vectorized: called for every element {how}'

    def rust_source(self):
        return self.last.rust_source() if self.last is not None else ''


def vectorize(function=None, *, parallel=False, **options):
    """Apply a one-value function to every element of arrays, lists or pandas columns.

    The loop and the function are compiled to Rust together. Arguments may mix arrays
    (all the same length) and plain numbers. Returns a NumPy array, or a pandas Series
    with the same index when an argument is a Series.

    parallel=True splits the elements across CPU cores (each element is independent,
    so the result is the same).
    Other keyword options are passed to @optimize (max_iterations, cache_dir, ...).
    """

    def decorate(fn):
        return Vectorized(fn, parallel, options)

    return decorate(function) if function is not None else decorate
