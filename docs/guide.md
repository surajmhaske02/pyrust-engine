# Guide: what compiles

PyRust compiles a numeric subset of Python. Anything outside it still runs, as plain Python, and `fn.explain()` tells you which line stopped compilation and how to change it.

## Values

| Type | Notes |
|---|---|
| `int` | 64-bit, overflow-checked. Results that don't fit replay in Python's big integers. |
| `float` | IEEE double, no fast-math: same results as CPython, including `-0.0`, NaN and infinities. |
| `bool` | `True`/`False`, comparisons, `and`/`or`/`not`; bools in arithmetic act as 0/1 like Python. |
| NumPy arrays | 1-D and 2-D `float64` / `int64`. |
| lists | `list[float]` or `list[int]` arguments: read, loop over, write (`lst[i] = x`). New lists with `[0.0] * n` can be filled and returned. |
| `str`, `bytes` | read-only: `len(s)`, `s[i]`, `for ch in s`, comparisons, `ch in "aeiou"`, `ord`, `chr`, `ch.isdigit()` and friends. |
| pandas Series | `float64` / `int64` columns with the default `0..n-1` index, read-only: `s[i]`, `s.iloc[i]`, `len(s)`, `for x in s`. |
| tuples | for unpacking (`a, b = b, a + b`) and returning several values. |

Annotations are optional. A function is compiled once for each combination of argument types it is called with.

## Statements and expressions

- Arithmetic: `+ - * / // % **`, `& | ^ ~ << >>`, comparisons including chained `a < b < c`.
- Control flow: `if`/`elif`/`else`, `for` over `range(...)`, over an array (`for x in a`), `enumerate(a)`, `prange(...)`, `while`, `break`, `continue`, early `return`, `x if c else y`.
- Builtins: `abs`, `min`, `max`, `sum(array)`, `round(x)`, `divmod`, `pow`, `int`, `float`, `bool`, `len`, `ord`, `chr`.
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

A list is copied into native memory for the call (a list of floats or a list of ints, not mixed). If the function writes into a list you passed, the new values are copied back into **the same list object** after the call succeeds. Each element keeps its type, so a list of floats only accepts float values (`lst[i] = float(x)`). `sum()` of a list of floats isn't compiled: Python 3.12+ adds Python floats with an extra-precise algorithm, so write the loop yourself.

Lists cost one copy per call. For very large data called often, NumPy arrays avoid the copy.

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

Strings are read character by character (any Unicode, including emoji). Supported: `len(s)`, `s[i]` (negative too), `for ch in s`, `enumerate(s)`, `==` `<` and the other comparisons between characters, `ch in "abc"` and `ch in s`, `ord(ch)`, `chr(i)`, and `ch.isdigit() isalpha() isalnum() isspace() isupper() islower() lower() upper()`. The character methods are exact for ASCII; any other character makes the call replay in Python, so results still match. A function can return one character (`-> str`). Building new strings (`+`, f-strings, `join`, `split`) isn't compiled.

`bytes` work like a read-only list of ints: `b[i]` and `for x in b` give ints.

## pandas

```python
@optimize
def mean_gap(prices) -> float:          # pass a column: df["price"]
    total = 0.0
    for i in range(1, len(prices)):
        total += abs(prices[i] - prices[i - 1])
    return total / (len(prices) - 1)
```

A `float64` or `int64` Series with the default index (`0..n-1`) is passed without copying. In Python, `s[i]` looks up the **label** `i`; with the default index that is position `i`, so `s[-1]` raises `KeyError` exactly as in Python. Use `s.iloc[i]` for positions, including negative ones. A Series with another index runs as Python with a hint (use `.reset_index(drop=True)`). Series are read-only in compiled code; write results into a NumPy array or list. For whole columns, `vectorize` (below) is usually simpler.

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
score(prices, 3)                   # plain numbers are reused for every element
```

A tiny function called once per value from Python is dominated by call overhead, even natively. `vectorize` generates the loop, compiles it together with your function, and runs everything in one native call. It replaces `df.apply(lambda row: ...)` and list comprehensions over columns. Every element gets exactly the value your function returns in Python. Array arguments must have the same length. `@vectorize(parallel=True)` splits the elements across CPU cores. Inputs that can't be compiled (other dtypes, mixed lists) run the same loop in Python. Also available as `@optimize.vectorize`.

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

Dicts, sets, list methods (`append`, ...), list literals other than `[value] * n`, building or formatting strings, classes and methods, closures, lambdas, generators, `try`/`with`, `print()`, slicing, `round(x, ndigits)`, `math.hypot`, `numpy.random`. Each comes with a hint in `explain()`.

!!! note "NumPy scalars"
    In plain Python, `a[i]` and `for x in a` give NumPy scalars (`np.float64`). PyRust returns plain `float`/`int` with the same value. Comparisons between ints larger than 2**53 and floats replay in Python, because NumPy and Python compare them differently.
