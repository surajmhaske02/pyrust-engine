# Guide: what compiles

PyRust compiles a numeric subset of Python. Anything outside it still runs, as plain Python, and `fn.explain()` tells you which line stopped compilation and how to change it.

## Values

| Type | Notes |
|---|---|
| `int` | 64-bit, overflow-checked. Results that don't fit replay in Python's big integers. |
| `float` | IEEE double, no fast-math: same results as CPython, including `-0.0`, NaN and infinities. |
| `bool` | `True`/`False`, comparisons, `and`/`or`/`not`; bools in arithmetic act as 0/1 like Python. |
| NumPy arrays | 1-D and 2-D `float64`, `int64` and `float32` (float32 needs NumPy 2). |
| lists | of floats, ints or strings: `[]`, `append`, `pop`, `extend`, `sort`, slicing, comprehensions, `sorted()`, `zip()`. |
| `str`, `bytes` | strings are values: slicing, `split`, `strip`, `find`, `replace`, `join`, `lower`, `+`, f-strings, `int(s)`, `float(s)`. `bytes` are read-only. |
| dicts and sets | `dict[str or int, int/float/bool/str]` and `set[str or int]`: get/set/`in`/`del`, loops over `.items()`, set `& | - ^`. |
| pandas Series | `float64` / `int64` columns with the default `0..n-1` index, read-only: `s[i]`, `s.iloc[i]`, `len(s)`, `for x in s`. |
| tuples | for unpacking (`a, b = b, a + b`) and returning several values. |

Annotations are optional. A function is compiled once for each combination of argument types it is called with.

## Statements and expressions

- Arithmetic: `+ - * / // % **`, `& | ^ ~ << >>`, comparisons including chained `a < b < c`.
- Control flow: `if`/`elif`/`else`, `for` over `range(...)`, over an array (`for x in a`), `enumerate(a)`, `prange(...)`, `while`, `break`, `continue`, early `return`, `x if c else y`.
- Builtins: `abs`, `min`, `max` (also of a list, and `max(d, key=d.get)`), `sum`, `round(x)`, `divmod`, `pow`, `int`, `float`, `bool`, `len`, `ord`, `chr`, `str`, `list`, `set`, `dict`, `sorted`, `zip`, `reversed(range(...))`.
- Module-level `frozenset`s and tuples of strings, and `str` constants (for example a stop-word set).
- `random`: `random()`, `uniform(a, b)`, `randint(a, b)`, `randrange(stop)`, `randrange(start, stop)`, `gauss(mu, sigma)`, `choice(seq)`.
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

## float32

```python
import numpy as np
from pyrust_engine import optimize

@optimize
def dot(a, b):                         # float32 arrays, e.g. embeddings
    total = np.float32(0.0)
    for i in range(len(a)):
        total += a[i] * b[i]           # float32 * float32 stays float32, like NumPy
    return total                       # an np.float32, as in Python
```

PyRust follows NumPy 2's rules exactly. float32 op float32 is float32. A Python number (a literal, an `int`/`float` argument, `len(...)`, a `math` result) is converted to float32 first. Mixing float32 with `np.float64` values from a float64 array isn't compiled, because NumPy decides that result type from where the value came from. Write `float(x)` to compute in float64, or `np.float32(x)` to stay in float32. Division by zero replays in Python, where NumPy returns `inf` with a warning.

## Lists

```python
@optimize
def running_max(values: list[int]) -> list[int]:
    out = [0] * len(values)
    best = values[0]
    for i, x in enumerate(values):
        best = max(best, x)
        out[i] = best
    return out                      # a normal Python list
```

Lists hold one type: floats, ints or strings. Local lists can be built with `[]` and `append`, `[0.0] * n`, literals like `[1, 2, 3]`, comprehensions (`[x * x for x in values if x > 0]`), slicing and `sorted()`, and changed with `append`, `pop`, `extend`, `sort(reverse=...)`, `reverse`, `clear` and `del lst[i]`. `index`, `count`, `in`, `min`, `max`, `+` and `zip` work too, and they can be returned.

Lists **passed in** are copied for the call:

- **Lists of numbers** can be written (`lst[i] = x`, `lst.sort()`), and the new values are copied back into the same list object after the call succeeds. Each element keeps its type, so a list of floats only accepts float values (`lst[i] = float(x)`).
- **Lists of strings** are read-only.
- **No list passed in can grow or shrink.** Build a new one and return it.

`sum()` of a list of floats isn't compiled, because Python 3.12+ adds Python floats with an extra-precise algorithm; write the loop yourself.

Two names for one list (`b = a`) aren't compiled, because Python would share the object. Write `b = list(a)` for a copy. Lists cost one copy per call; for very large numeric data called often, NumPy arrays avoid it.

## Strings and bytes

```python
@optimize
def edit_distance(a: str, b: str) -> int:
    prev = [0] * (len(b) + 1)
    cur = [0] * (len(b) + 1)
    for j in range(len(b) + 1):
        prev[j] = j
    for i in range(1, len(a) + 1):
        cur[0] = i
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        for j in range(len(b) + 1):
            prev[j] = cur[j]
    return prev[len(b)]

@optimize
def vowels(text: str) -> int:
    count = 0
    for ch in text:
        if ch.lower() in "aeiou":
            count += 1
    return count
```

Strings work with any Unicode, including emoji.

- **Reading:** `len(s)`, `s[i]` (negative too), slicing `s[a:b]`, `for ch in s`, `enumerate(s)`, comparisons (`==`, `<`, … by code point, like Python), and `in` (a character or a substring).
- **Methods:** `find`, `rfind`, `index`, `count` (with start/end), `startswith`, `endswith`, `split()` and `split(sep)`, `strip`, `lstrip`, `rstrip` (with or without characters), `replace`, `join`, `lower`, `upper`, and `isdigit`, `isalpha`, `isalnum`, `isspace` on characters and whole strings.
- **Building:** `+`, `s * n`, `s += t` (grows in place like CPython), f-strings with ints, bools and strings, and `str(int)`.
- **Parsing:** `int(s)` and `float(s)`.

These follow CPython's rules exactly, including whitespace for `split()`/`strip()` and the index rules for `find`. `lower`/`upper` and the `is...` tests are exact for ASCII. For other characters the call replays in Python, so results still match. Float formatting (`f"{x:.2f}"`, `str(1.5)`) isn't compiled: format floats outside the function.

`bytes` work like a read-only list of ints: `b[i]` and `for x in b` give ints.

## Dicts and sets

```python
STOP = frozenset({'the', 'a', 'of'})

@optimize
def word_counts(text: str) -> dict[str, int]:
    counts = {}
    for word in text.lower().split():
        if word not in STOP:
            counts[word] = counts.get(word, 0) + 1
    return counts                               # a normal dict, in Python's insertion order

@optimize
def jaccard(a: str, b: str) -> float:
    sa = set(a.split())
    sb = set(b.split())
    return len(sa & sb) / len(sa | sb) if sa | sb else 1.0
```

**Dicts** have str or int keys, and int, float, bool or str values (one type each).

- **Compiled:** `d[k]`, `d[k] = v`, `d[k] += v`, `d.get(k, default)`, `d.pop(k[, default])`, `d.setdefault(k, v)`, `del d[k]`, `k in d`, `len(d)`, `d.clear()`, loops over `d`, `d.keys()`, `d.values()` and `d.items()`, `sorted(d)`, `max(d, key=d.get)` and `dict(d)`.
- **Order:** dicts keep insertion order exactly like Python.
- **Changing size inside a loop:** adding or removing keys while looping over the dict replays in Python, which raises the same `RuntimeError`.

**Sets** have str or int elements.

- **Compiled:** `set()`, `{a, b}`, `set(words)`, `add`, `discard`, `remove`, `update`, `clear`, `in`, `len`, `&`, `|`, `-`, `^`, `sorted(s)`, and returning a set.
- **Not compiled:** looping over a set, because Python's set order can't be reproduced. Loop over `sorted(s)` instead.

**General rules:**

- Lists of strings, dicts and sets **passed in** are read-only in compiled code; build a new one and return it.
- A helper function you call can fill a dict or list you made: `helper(words, counts)`.
- Module-level constant collections must be `frozenset`s or tuples, because a global `set`/`dict` could change between calls.

## pandas

```python
@optimize
def mean_gap(prices) -> float:          # pass a column: df["price"]
    total = 0.0
    for i in range(1, len(prices)):
        total += abs(prices[i] - prices[i - 1])
    return total / (len(prices) - 1)
```

A `float64`, `int64` or `float32` Series with the default index (`0..n-1`) is passed without copying. In Python, `s[i]` looks up the **label** `i`; with the default index that is position `i`, so `s[-1]` raises `KeyError` exactly as in Python. Use `s.iloc[i]` for positions, including negative ones. A Series with another index runs as Python with a hint (use `.reset_index(drop=True)`). Series are read-only in compiled code; write results into a NumPy array or list. For whole columns, `vectorize` (below) is usually simpler.

**Text columns (pandas 3).** pandas 3 stores text in the `str` dtype. A Series of strings with no missing values compiles too (pandas 2 `object` columns of strings as well):

```python
@optimize
def count_valid(emails) -> int:         # df["email"]
    n = 0
    for e in emails:
        if e.find('@') > 0 and e.endswith('.com'):
            n += 1
    return n
```

`s[i]`, `len(s)`, loops over `s`, and `sorted`/`list`/`set`/`min`/`max`/`str.join` of `s` compile. Operators on the whole Series (`s + s`, `'x' in s`, `s == 'a'`) mean something different for a Series (element by element, or the index), so those functions run as Python, with a hint to pass `s.tolist()` instead. Columns with missing values (`NaN`, `None`) run as Python.

## polars

A polars Series without nulls is accepted where a list is: floats (`Float64`, `Float32`) and ints (`Int64` and smaller, `UInt32` and smaller) are read without copying where possible, and `String` columns work like a list of str. Like a list, `s[-1]` is the last value. Every value is exactly what Python sees when it reads the Series (Python `float`, `int`, `str`).

For per-row functions over number columns, use `@vectorize` (below) instead of `map_elements` (the numbers are read without copying). For text columns, polars' own `map_elements` is currently about as fast or faster:

```python
import polars as pl

df = df.with_columns(pl.col("email").map_batches(domain_length).alias("domain_len"))
```

## Random numbers

```python
import random

@optimize
def monte_carlo_pi(n: int) -> float:
    inside = 0
    for _ in range(n):
        x = random.random()
        y = random.random()
        if x * x + y * y <= 1.0:
            inside += 1
    return 4.0 * inside / n

random.seed(42)
monte_carlo_pi(1_000_000)               # the same answer plain Python gives after seed(42)
```

PyRust copies the state of Python's own random generator into the native call, runs the same Mersenne Twister algorithm, and writes the new state back when the call succeeds. So `random.seed()` works, results are identical to plain Python, and the next `random.random()` after the call continues the same sequence. Supported: `random()`, `uniform`, `randint`, `randrange` (one or two arguments), `gauss` and `choice`, called as `random.x()` or imported with `from random import x`. Not inside `prange` (Python draws numbers in order). Separate `random.Random()` instances and `numpy.random` aren't compiled.

!!! note "Threads"
    The generator state is copied in and written back, so another thread drawing random numbers from the shared generator during a compiled call has its draws overwritten. Give each thread its own work, or draw numbers on one thread.

## vectorize: one-value functions over whole columns

```python
from pyrust_engine import vectorize

@vectorize
def score(price: float, qty: int) -> float:
    return price * qty * (0.9 if qty > 10 else 1.0)

score(prices, quantities)          # NumPy arrays or lists -> NumPy array
score(df["price"], df["qty"])      # pandas columns -> Series with the same index
score(pl_df["price"], pl_df["qty"])  # polars columns -> polars Series with the same name
score(prices, 3)                   # plain numbers are reused for every element

@vectorize
def domain_length(email: str) -> int:
    at = email.find("@")
    return len(email) - at - 1 if at >= 0 else 0

domain_length(df["email"])         # str column (pandas 3 'str', object, polars String, list of str)

@vectorize
def clean(text: str) -> str:       # str results too
    return text.strip().lower()
```

A tiny function called once per value from Python is dominated by call overhead, even natively. `vectorize` generates the loop, compiles it together with your function, and runs everything in one native call. It replaces `df.apply(lambda row: ...)`, `Series.map` and list comprehensions over columns: about 20× faster on number and bool columns. On text columns the gain is small (0.9–2× against pandas `map`), because each Python string has to be converted. Every element gets exactly the value your function returns in Python. Array arguments must have the same length. `@vectorize(parallel=True)` splits the elements across CPU cores.

| Column type | Compiled | The function receives |
|---|---|---|
| `float64`, `int64` (NumPy, pandas, polars, lists) | yes | the values |
| `float32` (NumPy 2) | yes | `numpy.float32` values |
| `bool` | yes | `True` / `False` |
| `str` (pandas 3 `str`, `object` of str, polars `String`, list of str) | yes | `str` |
| polars `Int32`, `Int16`, `Int8`, `UInt32`, ... | yes | `int` |
| missing values, dates, categories, mixed lists | no: same loop in Python | the values |

Results can be numbers, bools or strings (a NumPy array, or a Series as above). Also available as `@optimize.vectorize`.

## Web apps and threads

A compiled call runs without holding Python's global interpreter lock, so CPU-heavy work in a web endpoint can run in a worker thread while the event loop keeps serving requests:

```python
import asyncio
from fastapi import FastAPI

app = FastAPI()

@app.get("/similarity")
async def similarity(a: str, b: str):
    distance = await asyncio.to_thread(edit_distance, a, b)   # compiled with @optimize
    return {"distance": distance}
```

Several threads can run compiled functions at the same time on different cores. For servers without Rust, build the libraries ahead of time (see [Deployment](deployment.md)).

**Free-threaded Python (3.14t).** On the free-threaded build there is no GIL at all, so the Python code between compiled calls runs in parallel too. PyRust loads no C extension, so it never turns the GIL back on; CI runs the full test suite on 3.14t and checks that the GIL stays off. `python -m pyrust_engine doctor` says which build you are on. `pyrust_engine.ai.batch(fn, items)` runs any function over many items on a thread pool.

**Call cost.** Since 0.7.3 a call with only `int`/`float`/`bool` arguments and a number, bool or `None` result uses a shorter native entry point that returns the value and the status together, about 40% less per call than 0.7.2 (2.8 µs instead of 4.5 µs on the Linux CI machine). Functions without a loop still run as plain Python by default, because even a short native call costs more than a few additions.

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

Classes and methods, closures, lambdas, generators, `try`/`with`, `print()`, float formatting, NumPy slices (they are views), mixed-type lists/dicts, float dict keys, loops over sets, global mutable `set`/`dict`/`list`, `round(x, ndigits)`, `math.hypot`, `numpy.random`. Each comes with a hint in `explain()`, or use `python -m pyrust_engine suggest file.py`.

!!! note "NumPy scalars"
    In plain Python, `a[i]` and `for x in a` give NumPy scalars (`np.float64`). PyRust returns plain `float`/`int` with the same value. Comparisons between ints larger than 2**53 and floats replay in Python, because NumPy and Python compare them differently.
