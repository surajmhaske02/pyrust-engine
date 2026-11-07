"""@vectorize: write a function for one value, run it over whole arrays in one native call.

    @vectorize
    def score(price: float, qty: int) -> float:
        return price * qty * (0.9 if qty > 10 else 1.0)

    score(prices, quantities)        # NumPy arrays, lists or pandas columns -> NumPy array
    score(df['price'], df['qty'])    # pandas columns -> pandas Series with the same index
    score(pl_df['price'], pl_df['qty'])  # polars columns -> polars Series with the same name
    score(prices, 3)                 # plain numbers are used for every element

Columns of float64, int64, float32, bool and str compile (bool values arrive as True/False);
the function may return a number, a bool or a str.

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

from .compiler import NUMPY2, Compiler, kind_of_value, prange

_COUNTER = itertools.count()
_OUT = '_pyrust_out'  # the result array; fresh for every call, so it needs no undo information
_RESULT_KINDS = ('f64', 'i64', 'bool', 'f32', 'strv')
_SCALAR_KINDS = {int: 'i64', float: 'f64', bool: 'bool', str: 'str'}


def _numpy():
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - depends on the environment
        raise ImportError('vectorize needs NumPy: pip install "pyrust-engine[arrays]"') from None
    return np


class _Column:
    """One array-like argument.

    data:  what the compiled loop receives (NumPy array or list of str), or None when only
           the Python loop can handle the values
    kind:  element kind passed to the function ('f64', 'i64', 'f32', 'bool', 'strv') or None
    items: what the Python loop indexes; its elements are exactly what the function gets
    """

    __slots__ = ('data', 'kind', 'items', 'index', 'name')

    def __init__(self, data, kind, items, index=None, name=None):
        self.data, self.kind, self.items, self.index, self.name = data, kind, items, index, name

    def __len__(self):
        return len(self.items)


def _from_numpy(np, value, index=None, name=None):
    if value.ndim != 1:
        return _Column(None, None, value, index, name)
    dtype = value.dtype
    if dtype == np.float64 or dtype == np.int64:
        value = np.ascontiguousarray(value)
        return _Column(value, 'f64' if dtype == np.float64 else 'i64', value, index, name)
    if dtype == np.float32 and NUMPY2():
        value = np.ascontiguousarray(value)
        return _Column(value, 'f32', value, index, name)
    if dtype == np.bool_:
        # bool columns arrive as Python bool (True/False), in compiled code and in Python.
        return _Column(value.astype(np.int64), 'bool', value.tolist(), index, name)
    return _Column(None, None, value, index, name)


def _from_list(np, value, index=None, name=None):
    kind = kind_of_value(value)
    if kind == 'flist':
        return _Column(np.array(value, dtype=np.float64), 'f64', value, index, name)
    if kind == 'ilist':
        return _Column(np.array(value, dtype=np.int64), 'i64', value, index, name)
    if kind == 'slist':
        return _Column(value, 'strv', value, index, name)
    if value and set(map(type, value)) == {bool}:
        return _Column(np.array(value, dtype=np.int64), 'bool', value, index, name)
    return _Column(None, None, value, index, name)


class Vectorized:
    """A one-value function applied element by element over arrays, lists and pandas/polars columns."""

    def __init__(self, function, parallel, options):
        self.python = getattr(function, 'python', function)
        functools.update_wrapper(self, self.python)
        self.parallel = parallel
        self.options = options
        self.arity = self.python.__code__.co_argcount
        self.kernels = {}  # (argument layout, result kind) -> compiled loop
        self.out_kinds = {}  # element kinds -> result kind or None
        self.last = None
        self.lock = threading.Lock()

    # ------------------------------------------------------------------ inputs
    def column(self, value):
        """_Column for an array-like argument, None for a plain value."""
        t = type(value)
        if t in _SCALAR_KINDS:
            return None
        np = sys.modules.get('numpy')
        if np is not None and t is np.ndarray:
            return _from_numpy(np, value)
        if t is list:
            return _from_list(_numpy(), value)
        pd = sys.modules.get('pandas')
        if pd is not None and isinstance(value, pd.Series):
            np = _numpy()
            dtype = value.dtype
            if isinstance(dtype, np.dtype) and dtype.kind != 'O':
                return _from_numpy(np, value.to_numpy(), value.index)
            if (getattr(dtype, 'kind', '') == 'O' or isinstance(dtype, pd.StringDtype)) and not value.hasnans:
                items = value.tolist()
                if set(map(type, items)) <= {str}:
                    return _Column(items, 'strv', items, value.index)
            return _Column(None, None, value.to_numpy(), value.index)
        pl = sys.modules.get('polars')
        if pl is not None and isinstance(value, pl.Series):
            np = _numpy()
            name = value.name
            if value.null_count():
                return _Column(None, None, value.to_list(), name=name)
            dtype = value.dtype
            if dtype == pl.String:
                items = value.to_list()
                return _Column(items, 'strv', items, name=name)
            if dtype == pl.Boolean:
                items = value.to_list()
                return _Column(np.array(items, dtype=np.int64), 'bool', items, name=name)
            if dtype in (pl.Float64, pl.Int64, pl.Float32):
                return _from_numpy(np, value.to_numpy(), name=name)
            if dtype in (pl.Int32, pl.Int16, pl.Int8, pl.UInt32, pl.UInt16, pl.UInt8):
                return _from_numpy(np, value.to_numpy().astype(np.int64), name=name)
            return _Column(None, None, value.to_list(), name=name)
        return None

    def out_kind(self, kinds):
        """Result kind of the function for these element kinds, or None if it cannot compile."""
        if kinds not in self.out_kinds:
            try:
                # Compiled on its own, a str parameter has kind 'str' (in the loop it receives 'strv').
                own = ['str' if kind == 'strv' else kind for kind in kinds]
                compiler = Compiler(self.python, kinds=dict(zip(list(self.python.__code__.co_varnames), own)))
                compiler.generate()
                kind = compiler.return_type
                self.out_kinds[kinds] = kind if kind in _RESULT_KINDS else None
            except Exception:
                self.out_kinds[kinds] = None
        return self.out_kinds[kinds]

    # ----------------------------------------------------------------- kernels
    def kernel(self, layout, result_kind):
        """Compiled loop. layout: per argument None (plain value), 'array' or 'bool' (0/1 ints read as bool)."""
        key = (layout, result_kind == 'strv')
        with self.lock:
            existing = self.kernels.get(key)
            if existing is not None:
                return existing
            from .runtime import optimize

            params = [f'x{k}' for k in range(self.arity)]
            items = []
            for k, how in enumerate(layout):
                items.append(f'x{k}' if how is None else f'bool(x{k}[i])' if how == 'bool' else f'x{k}[i]')
            call = ', '.join(items)
            name = '_pyrust_vectorized_' + self.python.__name__
            if result_kind == 'strv':
                # Strings are collected in a list (returned as one packed buffer).
                source = (
                    f'def {name}({", ".join(params + ["_pyrust_n"])}):\n'
                    f'    out = []\n'
                    f'    for i in range(_pyrust_n):\n'
                    f'        out.append(_pyrust_function({call}))\n'
                    f'    return out\n'
                )
            else:
                loop = 'prange' if self.parallel else 'range'
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
            self.kernels[key] = compiled
            return compiled

    # -------------------------------------------------------------------- call
    def __call__(self, *args):
        if len(args) != self.arity:
            return self.python(*args)  # raises Python's own TypeError
        columns = [self.column(a) for a in args]
        if all(c is None for c in columns):
            return self.python(*args)
        np = _numpy()
        lengths = {len(c) for c in columns if c is not None}
        if len(lengths) != 1:
            raise ValueError(f'{self.python.__name__}: array arguments have different lengths {sorted(lengths)}')
        n = lengths.pop()
        kinds = []
        for value, c in zip(args, columns):
            kinds.append(_SCALAR_KINDS.get(type(value)) if c is None else c.kind)
        result_kind = self.out_kind(tuple(kinds)) if None not in kinds else None
        if result_kind is None:
            out = self.python_loop(args, columns, n)
        else:
            layout = tuple(None if c is None else 'bool' if c.kind == 'bool' else 'array' for c in columns)
            kernel = self.kernel(layout, result_kind)
            values = [a if c is None else c.data for a, c in zip(args, columns)]
            if result_kind == 'strv':
                out = np.array(kernel(*values, n))
            else:
                dtype = {'f64': np.float64, 'f32': np.float32}.get(result_kind, np.int64)
                out = np.empty(n, dtype=dtype)
                kernel(*values, out)
                if result_kind == 'bool':
                    out = out.astype(bool)
            self.last = kernel
        return self.wrap(out, args, columns)

    def wrap(self, out, args, columns):
        """pandas Series (same index) or polars Series (same name) when an argument was one."""
        index = next((c.index for c in columns if c is not None and c.index is not None), None)
        if index is not None:
            import pandas as pd

            return pd.Series(out, index=index)
        pl = sys.modules.get('polars')
        if pl is not None:
            source = next((a for a in args if isinstance(a, pl.Series)), None)
            if source is not None:
                return pl.Series(source.name, out)
        return out

    def python_loop(self, args, columns, n):
        np = _numpy()
        pick = []
        for value, c in zip(args, columns):
            if c is None:
                pick.append(lambda i, v=value: v)
            else:
                pick.append(lambda i, d=c.items: d[i])
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
    """Apply a one-value function to every element of arrays, lists or pandas/polars columns.

    The loop and the function are compiled to Rust together. Arguments may mix arrays
    (all the same length) and plain values. Returns a NumPy array, a pandas Series with
    the same index when an argument is a pandas Series, or a polars Series with the same
    name when an argument is a polars Series (use it in pl.col('x').map_batches(f)).

    parallel=True splits the elements across CPU cores (each element is independent,
    so the result is the same).
    Other keyword options are passed to @optimize (max_iterations, cache_dir, ...).
    """

    def decorate(fn):
        return Vectorized(fn, parallel, options)

    return decorate(function) if function is not None else decorate
