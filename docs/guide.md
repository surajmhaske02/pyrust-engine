# Guide: what compiles

PyRust compiles a numeric subset of Python. Anything outside it still runs, as plain Python, and `fn.explain()` tells you which line stopped compilation and how to change it.

## Values

| Type | Notes |
|---|---|
| `int` | 64-bit, overflow-checked. Results that don't fit replay in Python's big integers. |
| `float` | IEEE double, no fast-math: same results as CPython, including `-0.0`, NaN and infinities. |
| `bool` | `True`/`False`, comparisons, `and`/`or`/`not`; bools in arithmetic act as 0/1 like Python. |
| NumPy arrays | 1-D and 2-D `float64` / `int64`, C-contiguous. |
| tuples | for unpacking (`a, b = b, a + b`) and returning several values. |

Annotations are optional. A function is compiled once for each combination of argument types it is called with (`int`, `float`, `bool`, array kinds).

## Statements and expressions

- Arithmetic: `+ - * / // % **`, `& | ^ ~ << >>`, comparisons including chained `a < b < c`.
- Control flow: `if`/`elif`/`else`, `for` over `range(...)`, over an array (`for x in a`), `enumerate(a)`, `prange(...)`, `while`, `break`, `continue`, early `return`, `x if c else y`.
- Builtins: `abs`, `min`, `max`, `sum(array)`, `round(x)`, `divmod`, `pow`, `int`, `float`, `bool`, `len`.
- `math`: `sqrt exp log log2 log10 sin cos tan asin acos atan atan2 sinh cosh tanh fabs copysign fmod floor ceil trunc isnan isinf isfinite pi e tau inf nan`.
- Module-level `int`/`float`/`bool` constants.
- Calls to your own functions, with keyword and default arguments, including recursion (annotate the return type: `-> int`).

## Arrays

```python
@optimize
def normalize(values: F64Array) -> None:
    total = sum(values)
    for i in range(len(values)):
        values[i] /= total

@optimize
def outer(a, b):
    out = np.zeros((len(a), len(b)))
    for i in range(len(a)):
        for j in range(len(b)):
            out[i, j] = a[i] * b[j]
    return out
```

Writes go straight into your array, with an undo log: if the call fails halfway, the writes are undone and the call replays in Python, so you see exactly what plain Python would do.

## Parallel loops

```python
from pyrust_engine import optimize, prange

@optimize
def transform(values, out) -> int:
    positive = 0
    for i in prange(len(values)):
        out[i] = math.sqrt(abs(values[i])) * 2.0
        if values[i] > 0:
            positive += 1          # int totals are allowed: they add up exactly in any order
    return positive
```

Rules that keep results identical to Python: each iteration writes only its own element (`out[i]`), arrays it writes are read only at `[i]`, and the only shared updates are `+=` on int totals. Float totals are rejected because their result depends on the order of additions.

## What runs as Python

Lists, dicts, sets, strings, classes and methods, closures, lambdas, generators, `try`/`with`, `print()`, array slicing, `round(x, ndigits)`, `math.hypot`. Each comes with a hint in `explain()`.

!!! note "NumPy scalars"
    In plain Python, `a[i]` and `for x in a` give NumPy scalars (`np.float64`). PyRust returns plain `float`/`int` with the same value. Comparisons between ints larger than 2**53 and floats replay in Python, because NumPy and Python compare them differently.
