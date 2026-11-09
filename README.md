# PyRust Engine

[![Tests](https://github.com/surajmhaske02/pyrust-engine/actions/workflows/test.yml/badge.svg)](https://github.com/surajmhaske02/pyrust-engine/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/pyrust-engine)](https://pypi.org/project/pyrust-engine/)
![Python](https://img.shields.io/badge/python-3.10%20to%203.15-blue)
[![Docs](https://img.shields.io/badge/docs-website-blue)](https://surajmhaske02.github.io/pyrust-engine/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/surajmhaske02/pyrust-engine/blob/main/LICENSE)
![Status](https://img.shields.io/badge/status-experimental-orange)

**Make your Python code faster by adding one line.**

Put `@optimize` above a function. PyRust turns that function into Rust, compiles it and runs it as native machine code. You keep writing normal Python, and **your answers never change**: anything PyRust can't run safely runs as normal Python instead.

- ✍️ **Write plain Python.** No new language, no build step, no rewrite. Type annotations are optional.
- ⚡ **Get real speed on CPU-heavy code:** arithmetic, `math` functions, bit operations, helper functions, recursion and NumPy arrays run about **6× to 110× faster** in our tests on Linux, macOS and Windows.
- 🧰 **Use everyday Python:** lists, strings, dicts, sets, float32 embeddings, pandas columns (pandas 3 text columns too), polars Series, `random`, and methods that read and update `self.x` compile too, with the same results as plain Python.
- 🐼 **Speed up small functions over whole columns** with `@vectorize`: a faster replacement for `df.apply`, `Series.map` and polars `map_elements` (about 20× faster on number columns, 3–4× on text columns).
- 🤖 **Built for AI apps and agents:** ready-made `pyrust_engine.ai` (BM25 with add/remove/save/load, a drop-in `BM25Okapi` for `rank_bm25`, cosine, binary and int8 top-k with filters, chunking, metrics, batches), `suggest`/`apply` with JSON output, and an MCP server for coding assistants.
- 🧶 **Ready for free-threaded Python 3.14t:** tested in CI, and PyRust never turns the GIL back on.
- 🧵 **Use every CPU core** with `prange()` for loops whose iterations are independent.
- 📊 **See what runs natively:** `pyrust_engine.print_stats()`, `on_fallback()` hooks and logging for production.
- 🛟 **Never get a wrong answer.** Big integers, math errors, Ctrl-C and anything unsupported fall back to normal Python with the exact same result or exception.
- 🔍 **See what's happening.** `fn.explain()` and a one-time warning say which line can't be sped up, why, and how to rewrite it.

📖 [Documentation](https://surajmhaske02.github.io/pyrust-engine/) · 🚀 [Getting started](https://surajmhaske02.github.io/pyrust-engine/getting-started/) · 🧩 [Examples](https://surajmhaske02.github.io/pyrust-engine/examples/) · 📓 [Notebook](https://github.com/surajmhaske02/pyrust-engine/blob/main/examples/pyrust_vs_python.ipynb) · 📚 [API reference](https://github.com/surajmhaske02/pyrust-engine/blob/main/docs/API.md) · 🤝 [Contributing](https://github.com/surajmhaske02/pyrust-engine/blob/main/CONTRIBUTING.md) · 📝 [Changelog](https://github.com/surajmhaske02/pyrust-engine/blob/main/CHANGELOG.md)

```python
import math
import numpy as np
from pyrust_engine import optimize, prange

def distance(x, y):
    return math.sqrt(x * x + y * y)           # plain helper, compiled together with its caller

@optimize
def total_distance(n):
    total = 0.0
    for i in range(n):
        total += distance(float(i), float(n - i))
    return total

@optimize
def mean_and_spread(values):                   # NumPy array in, two values out
    mean = sum(values) / len(values)
    spread = 0.0
    for x in values:
        spread = max(spread, abs(x - mean))
    return mean, spread

@optimize
def transform(values, out) -> int:             # all CPU cores; result identical to Python
    positive = 0
    for i in prange(len(values)):
        out[i] = math.sqrt(abs(values[i])) * 2.0
        if values[i] > 0:
            positive += 1
    return positive

total_distance(300_000)                        # ~64x faster than plain Python on Linux in our benchmark
print(total_distance.explain())                # compiles? if not: which line, why, and a hint
```

Everyday Python and AI work (0.7.1 and 0.7.2):

```python
import random
from pyrust_engine import optimize, vectorize

@optimize
def edit_distance(a: str, b: str) -> int:      # strings and lists
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
def monte_carlo_pi(n: int) -> float:           # same numbers as plain Python after random.seed()
    inside = 0
    for _ in range(n):
        x, y = random.random(), random.random()
        if x * x + y * y <= 1.0:
            inside += 1
    return 4.0 * inside / n

@vectorize
def score(price: float, qty: int) -> float:    # write it for one row...
    return price * qty * (0.9 if qty > 10 else 1.0)

df["score"] = score(df["price"], df["qty"])    # ...run it over whole pandas columns in one native loop

@optimize
def word_counts(text: str) -> dict[str, int]:  # dicts, sets, str.split, f-strings
    counts = {}
    for word in text.lower().split():
        counts[word] = counts.get(word, 0) + 1
    return counts

from pyrust_engine.ai import BM25, cosine_top_k, chunk_text, token_f1
ids, scores = cosine_top_k(query_vec, embeddings, k=5)     # float32 embedding search on all cores
hits = BM25(chunk_text(document, 800, 100)).search("fast python", k=3)
```

Classes and methods (0.7.5):

```python
from dataclasses import dataclass
from pyrust_engine import optimize

@dataclass
class Particle:
    x: float
    vx: float
    steps: int = 0

    @optimize
    def advance(self, dt: float, n: int) -> None:   # 7.8–12× faster in CI
        for _ in range(n):
            self.x += self.vx * dt                  # attributes are written back after the call
            self.steps += 1
```

New in 0.7.3:

```python
from pyrust_engine.ai import BM25Okapi, binarize, hamming_top_k, tokenize_many

bm25 = BM25Okapi(tokenized_corpus)             # was: from rank_bm25 import BM25Okapi (same scores)
bm25.add(more_docs); bm25.remove([3]); bm25.save("index.npz")

ids, dist = hamming_top_k(binarize(query_vec), binarize(vectors), 10, mask=tenant == "acme")
tokens = tokenize_many(documents)              # all CPU cores, same result as one by one

@vectorize
def domain(email: str) -> str:                 # text columns in, text out
    return email[email.find("@") + 1:].lower()

df["domain"] = domain(df["email"])             # pandas 3 str column
pl_df.with_columns(pl.col("email").map_batches(domain))   # polars
```

For coding agents: `python -m pyrust_engine suggest file.py --json`, `apply file.py --write`, and `claude mcp add pyrust -- python -m pyrust_engine mcp`. See [AI and agents](https://surajmhaske02.github.io/pyrust-engine/ai/) and [`AGENTS.md`](https://github.com/surajmhaske02/pyrust-engine/blob/main/AGENTS.md).

> **Status: experimental.** Not a general Python compiler and not production-hardened. 272 tests pass in CI on Linux, macOS and Windows with Python 3.10 to 3.15, and on free-threaded Python 3.14t and 3.15t.

## Comparison at a glance

| | Baseline | **PyRust 0.7.5** |
|---|---|---|
| Integer loop (500k) | 43.1 ms · 1× | **1.19 ms · 36×** |
| Float recurrence (300k) | 19.9 ms · 1× | **1.82 ms · 11.0×** |
| Array kernel, read (100k float64) | 12.1 ms · 1× | **0.125 ms · 96×** |
| Write into an output array (100k) | 42.6 ms · 1× | **0.435 ms · 98×** |
| Recursive `fib(25)` | 9.70 ms · 1× | **0.736 ms · 13.2×** |
| `math.sqrt` / `sin` / `log` loop (300k) | 49.2 ms · 1× | **5.11 ms · 9.6×** |
| Helper function called in a loop (300k) | 56.4 ms · 1× | **0.840 ms · 67×** |
| Bit-mixing hash with `^ & << >> **` (300k) | 90.7 ms · 1× | **0.748 ms · 121×** |
| A method updating `self.x`, `self.vx`… (20k steps) | 2.12 ms · 1× | **0.176 ms · 12.1×** |
| 1,000 calls on a 32-element NumPy array | 3.91 ms · 1× | **0.440 ms · 8.9×** |
| Edit distance of two strings (288 chars each) | 13.7 ms · 1× | **0.937 ms · 14.6×** |
| Monte Carlo with `random.random()` (200k) | 30.7 ms · 1× | **4.05 ms · 7.6×** |
| Tiny function over 100k rows: `@vectorize` vs a Python loop | 19.1 ms · 1× | **0.737 ms · 26×** |
| `@vectorize` over a 200k-row text column vs pandas `Series.map` | 96.5 ms (pandas) | **20.1 ms · 4.8×** |
| Variance of a Python list (100k floats, copy included) | 7.51 ms · 1× | **0.950 ms · 7.9×** (0.7.7: lists read directly) |
| Tokenize 1.1M characters (`ai.tokenize`) | 60.9 ms · 1× | **20.9 ms · 2.9×** |
| Word count with a dict (160k words) | 22.7 ms · 1× | **16.9 ms · 1.35×** |
| Edit distance of two 500-char strings (`ai.levenshtein`) | 53.6 ms · 1× | **3.06 ms · 17.5×** |
| float32 dot product (100k, NumPy 2 rules) | 24.9 ms · 1× | **0.214 ms · 117×** |
| BM25 query over 1,977 documents | 26.2 ms · 1× | **0.052 ms · 508×** |
| `BM25Okapi` query, 5,000 documents, vs the **`rank_bm25`** package | 1.98 ms (rank_bm25) | **0.068 ms · 29×** (same scores) |
| Binary-code top-10 over 200k × 384 bits vs **NumPy** | 13.4 ms (NumPy) | **4.19 ms · 3.2×** |
| int8 top-10 over 100k × 384 vs **NumPy** (int32 copy + matmul) | 39.3 ms (NumPy) | **40.0 ms · 0.98×** (no 4× copy) |
| Cosine top-10 over 50k×384 float32 vs **NumPy BLAS** | 19.3 ms (NumPy) | **23.1 ms · 0.84×** |
| JSON → array pipeline | 9.13 ms · 1× | **6.67 ms · 1.37×** |
| Native call of a tiny function, per call | 0.059 µs (Python) | **0.25 µs** (0.7.3: 2.8 µs) |
| Tiny `add(a, b)` per call | ≈1× | **≈1×** (runs as plain Python on purpose; use `@vectorize` for columns) |
| Big integers / overflow / math errors | Exact | **Exact (replays in Python)** |
| Unsupported code | Runs | **Runs in Python, with the line and a hint** |
| Runtime dependencies | None | **None** (NumPy for arrays; Rust only to compile) |
| First call (compile) | None | 0.1–0.5 s (about 0.005 s from the disk cache) |

<sub>"time · speedup", measured by CI for 0.7.5. Rows marked "vs NumPy" or "vs `rank_bm25`" compare with those libraries, not with plain Python. BM25 is compared with a straightforward pure-Python BM25 that scans the documents for each query; cosine top-k is compared with vectorized NumPy, which is already native code (PyRust is about even there). Runner: GitHub Actions `ubuntu-latest` runner (x86_64, Python 3.13, Rust 1.99). Shared CI machines are noisy; treat the numbers as approximate. Raw data: [`benchmarks/`](https://github.com/surajmhaske02/pyrust-engine/tree/main/benchmarks).</sub>

![Speedup versus plain Python](https://cdn.jsdelivr.net/gh/surajmhaske02/pyrust-engine@main/docs/benchmark-speedup.svg)

### Speed on every platform

Speedup over plain Python, same CI run (0.7.5), Python 3.13 and Rust 1.99:

| Workload | Linux x86_64 | macOS Apple Silicon | Windows x86_64 |
|---|---:|---:|---:|
| Integer loop | 36× | 53× | 36× |
| Float recurrence | 11.0× | 7.4× | 9.2× |
| Array kernel (read) | 96× | 81× | 97× |
| Write into an output array | 98× | 78× | 63× |
| Recursive `fib(25)` | 13.2× | 15.0× | 15.8× |
| `math` functions loop | 9.6× | 13.7× | 15.0× |
| Helper function in a loop | 67× | 110× | 65× |
| Bit-mixing hash | 121× | 83× | 122× |
| Method updating `self.x` | 12.1× | 7.8× | 10.0× |
| 1,000 calls on a small array | 8.9× | 6.1× | 6.3× |
| Text column vs pandas `map` | 4.8× | 4.2× | 3.8× |
| Native call of a tiny function, per call | 0.25 µs | 0.18 µs | 0.25 µs |

Float recurrences and `math`-heavy loops gain less because the floating-point work itself costs the same in both; PyRust removes the interpreter overhead around it. A comparison with Numba is on the [benchmarks page](https://surajmhaske02.github.io/pyrust-engine/benchmarks/).

## What compiles

| | |
|---|---|
| **Values** | `int` (64-bit, overflow-checked), `float`, `bool`, tuples (unpacking and multiple returns), NumPy `float64`/`int64`/`float32` arrays (1-D and 2-D), lists of floats, ints or strings, `str`, `bytes`, `dict`, `set`, pandas Series (`float64`/`int64`/`float32`/`str`), polars Series |
| **Operators** | `+ - * / // % **`, `& \| ^ ~ << >>`, comparisons incl. chained and exact int-vs-float |
| **Control flow** | `if`/`elif`/`else`, `for` over `range`, an array, `enumerate` or `prange`, `while`, `break`, `continue`, `return`, `x if c else y` |
| **Builtins** | `abs min max sum round divmod pow int float bool len ord chr str list set dict sorted zip`, `int.bit_count()` |
| **Strings** | slicing, `split strip find rfind count startswith endswith replace join lower upper`, `+`, `+=`, f-strings, `int(s)`, `float(s)`, `in` |
| **Lists** | `[]`, `append pop extend sort reverse index count`, literals, comprehensions, slicing, `sorted()`; lists you pass are written back |
| **dict / set** | `d[k]`, `get pop setdefault del in`, loops over `items()`, `max(d, key=d.get)`; sets `add discard in & \| - ^` |
| **`random`** | `random uniform randint randrange gauss choice`, exactly Python's sequence |
| **pandas / polars** | Series of numbers or str: `s[i]`, `s.iloc[i]`, `len(s)`, `for x in s`; `@vectorize` over whole columns (numbers, bools, str) |
| **`math`** | `sqrt exp log log2 log10 sin cos tan asin acos atan atan2 sinh cosh tanh fabs copysign fmod floor ceil trunc isnan isinf isfinite pi e tau inf nan` |
| **Arrays** | `a[i]`, `a[i, j]`, `a.shape`, `len(a)`, `for x in a`, writes (`a[i] = v`, `+=`), new arrays with `np.zeros/empty/ones(_like)` that can be returned, read-only views `a[lo:hi]` and `m[i, :]`, `int32`/`uint8`/`bool` arrays, `np.sqrt/floor/ceil/abs/square/isnan` |
| **Classes** | methods and functions reading `obj.x` and assigning number attributes (`self.x += v`), methods calling methods (`self.helper()`); dataclasses, `__slots__`, class attributes |
| **Exceptions** | `try`/`except`/`else`/`finally`, `raise`, `assert` (an exception replays the call in Python, which handles it exactly) |
| **Functions** | calls to your own functions (keyword and default arguments, recursion with a `-> int`/`-> float` annotation), module-level constants |

Everything else (`print`, `with`, float formatting, closures, column slices, …) still runs, as plain Python, and `explain()` gives a hint. Full details: [Guide](https://surajmhaske02.github.io/pyrust-engine/guide/).

## What makes Python slow, and what PyRust does about it

| Why Python is slow | Example | PyRust 0.7 |
|---|---|---|
| Every operation is interpreted one bytecode at a time | loops | ✅ Compiled to native machine code |
| Every number is a heap object with type checks | `total += i * i` | ✅ Plain 64-bit ints, floats and bools in CPU registers |
| Function calls are expensive | helpers in a loop, recursion | ✅ Helpers and recursion compiled into the same native code |
| Math functions go through Python objects | `math.sqrt(x)`, `x ** 0.5` | ✅ Native calls, same results |
| Bit manipulation creates new int objects | hashing | ✅ Native bit operations |
| Indexing arrays goes through NumPy objects | `a[i]`, `out[i] = x` | ✅ Reads and writes array memory directly |
| The GIL keeps one thread busy | CPU-bound loops | ✅ `prange()` runs independent iterations on all cores; `ai.batch` runs many items on threads; free-threaded 3.14t supported |
| Crossing into native code has a fixed cost | tiny functions called millions of times | ✅ `@vectorize` runs them over whole arrays in one call; ✅ since 0.7.4 CPython calls the native code directly: about 0.2 µs per call (was 2–4.5 µs) |
| Python values are generic objects | lists, text processing, counting | ✅ Lists, strings, dicts and sets as native data, with Python's exact rules |
| Random numbers go through Python calls | simulations, Monte Carlo | ✅ Python's own generator algorithm, natively, same sequence |
| Objects and classes are dynamic | methods, dataclasses | ✅ since 0.7.5 `self.x` reads and assignments compile (numbers, str, NumPy arrays); ❌ method calls on objects, JSON documents |
| I/O waits | network, disk, database | ❌ Not a CPU problem: use `asyncio` or threads |

## Version history

| Release | What it added |
|---|---|
| **0.7.7 (current)** | Methods calling methods, `try`/`except`/`raise`/`assert`, NumPy views and `int32`/`uint8`/`bool` arrays, `np.sqrt` and friends, direct calls with lists and `str`/list results, `stats()` and logging |
| 0.7.6 | `suggest`/`apply` and the MCP server handle class methods; MCP 2025-11-25 (tool hints, structured results); free-threaded 3.15t in CI; refreshed docs and benchmarks |
| 0.7.5 | Methods and objects: `self.x` reads and number assignments (dataclasses, `__slots__`, plain classes); direct calls with NumPy arrays, `str` and `bytes` |
| 0.7.4 | CPython calls compiled functions directly (about 0.2 µs instead of 2–3 µs per call); text columns and text batches 2–4× faster |
| 0.7.3 | pandas 3 text and polars columns, drop-in `BM25Okapi` for `rank_bm25`, index add/remove/save/load, binary and int8 search with filters, batches, free-threaded Python |
| 0.7.2 | float32, string values, growable lists, dicts and sets, the `pyrust_engine.ai` library, `suggest`/`apply`, MCP server |
| 0.7.1 | Lists, `str`/`bytes`, pandas Series, the `random` module, `@vectorize` |
| 0.7.0 | bools, tuples, 2-D and new arrays, `prange`, per-type specialization, unlimited long calls with Ctrl-C, undo log for array writes, ahead-of-time `build`, Jupyter magics |
| 0.5–0.6 | ints, floats, `float64`/`int64` arrays, `math`, helper calls, recursion, `doctor` |

Details for every release: [Changelog](https://github.com/surajmhaske02/pyrust-engine/blob/main/CHANGELOG.md).

## Install

| Needed | Version | Why |
|---|---|---|
| **Python** | 3.10 to 3.15 | runs PyRust |
| **Rust** (`rustc`) | any recent stable | compiles your functions (not needed for [prebuilt](#deploy-without-rust) libraries) |
| **C linker** | system default | Rust uses it to build the library |
| NumPy | optional | arrays |

| OS | Install Rust |
|---|---|
| **macOS** | `xcode-select --install`, then `brew install rustup && rustup default stable` |
| **Linux** | `sudo apt install build-essential`, then `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` |
| **Windows** | [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with "Desktop development with C++", then `winget install Rustlang.Rustup` |

```sh
pip install "pyrust-engine[arrays]"
python -m pyrust_engine doctor          # checks Python, Rust, the linker, NumPy, and compiles a test function
```

Without Rust, everything still runs correctly as plain Python, and PyRust shows **one** warning with the exact fix (including for half-installed toolchains, missing linkers and Jupyter's shorter `PATH`). See [Troubleshooting](https://surajmhaske02.github.io/pyrust-engine/troubleshooting/). On macOS with Homebrew Python, install into a virtual environment first: `python3 -m venv ~/pyrust-env && source ~/pyrust-env/bin/activate`.

## Usage highlights

### Arrays: read, write, create

```python
@optimize
def outer(a, b):
    out = np.zeros((len(a), len(b)))
    for i in range(len(a)):
        for j in range(len(b)):
            out[i, j] = a[i] * b[j]
    return out                                 # a new NumPy array
```

Writes into arrays you pass in go straight to memory with an undo log: if the call fails halfway, the writes are undone and the call replays in Python, so the array ends up exactly as plain Python would leave it.

### Recursion and multiple return values

```python
@optimize
def fib(n: int) -> int:                        # recursion needs the return annotation
    return n if n < 2 else fib(n - 1) + fib(n - 2)

@optimize
def fib_pair(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a, b
```

### Parallel loops

`prange()` is `range()` in plain Python. Under PyRust it splits iterations across CPU cores, but only for patterns whose result is exactly the sequential one: each iteration writes its own element (`out[i] = ...`), and the only shared updates are `+=` on int totals. Anything else is rejected with an explanation (float totals depend on the order of additions).

### Diagnostics

```python
fn.explain()        # compiles or not; the line, reason and a hint; types it was compiled for
fn.report()         # calls, native/fallback counts, fallback reasons, specializations, long calls
fn.rust_source()    # the generated Rust
```

```text
PyRustFallbackWarning: simulate ran as plain Python: simulate, line 14: unsupported expression: `[x, y]`
  hint: List literals are compiled only as [value] * n. Pass lists of numbers as arguments, or create one with [0.0] * n and fill it in a loop.
```

In Jupyter: `%load_ext pyrust_engine`, then `%pyrust_check fn` and `%pyrust_benchmark fn(args)`.

### Lists, strings, pandas and random numbers

```python
@optimize
def double_all(values: list[int]) -> None:
    for i in range(len(values)):
        values[i] *= 2                         # written back into the same list after the call

@optimize
def count_digits(text: str) -> int:
    count = 0
    for ch in text:
        if ch.isdigit():
            count += 1
    return count

@optimize
def mean_gap(prices) -> float:                 # a pandas column: mean_gap(df["price"])
    total = 0.0
    for i in range(1, len(prices)):
        total += abs(prices[i] - prices[i - 1])
    return total / (len(prices) - 1)
```

`random.random()`, `uniform`, `randint`, `randrange`, `gauss` and `choice` run Python's own Mersenne Twister algorithm natively: after `random.seed(42)` you get exactly the numbers plain Python gives, and the sequence continues correctly afterwards.

### Web apps and threads

Compiled calls don't hold the GIL, so a CPU-heavy step in a FastAPI/Django/Flask endpoint can run in a worker thread (`await asyncio.to_thread(fn, ...)`) without blocking other requests. Ship [prebuilt](#deploy-without-rust) libraries so servers don't need Rust. On free-threaded Python 3.14t, plain Python threads run in parallel as well; PyRust is tested there and keeps the GIL off.

### Deploy without Rust

```sh
python -m pyrust_engine build mypackage/kernels.py     # writes mypackage/__pyrust__/ (+ manifest.json)
```

Ship the `__pyrust__/` folder with your package. Machines without Rust load the prebuilt libraries (matched by platform and generated code, verified by SHA-256). See [Deployment](https://surajmhaske02.github.io/pyrust-engine/deployment/).

### Options

```python
@optimize(fallback=False)            # raise OptimizationError instead of running as Python
@optimize(max_iterations=10**9)      # hard limit; default None = unlimited, Ctrl-C works
@optimize(native_loop_free=True)     # compile tiny loop-free functions too
@optimize(warn_on_fallback=False)    # silence PyRustFallbackWarning
@optimize(timing=True)               # record native/fallback timings in report()
```

## Why it exists

Most "make Python fast" tools ask you to give something up: a new interpreter, a build step, a different integer model, or a heavy dependency. PyRust takes a narrower bet:

- **Correctness first.** Integers stay arbitrary-precision; overflow, division by zero, bounds, math domain errors and anything unexpected stop the native call, undo its array writes and **replay the original Python function**, so you get exactly Python's result or exception.
- **Opt-in, per function.** Decorate the hot function, leave everything else alone.
- **Light footprint.** Pure-Python package with no runtime dependencies. Uses `rustc` directly through `ctypes`: no PyO3, Maturin or crates.

| | **PyRust** | Numba | Cython | PyPy |
|---|---|---|---|---|
| How you use it | decorator | decorator | separate build step | different interpreter |
| Integer overflow | exact (replays in Python) | wraps at 64 bits | C rules for C types | exact |
| Unsupported code | runs as Python, with a hint | compile error | compiles via the C API | runs everything |
| Plain helpers called inside | compiled together | must be `@njit` too | yes | yes |
| Runtime dependencies | none | numba + llvmlite | none after build | whole runtime |
| Maturity | experimental | mature | mature | mature |

## Limitations

- **A subset of Python.** Methods compile through attribute reads, number assignments (`self.x += 1`) and calls to other plain methods; static/class methods and properties, closures, generators, `with`, float formatting, column slices or `print` in compiled code run as Python. Code inside `try` is fast only while it doesn't raise: an exception replays the whole call in Python.
- **Containers hold one type** (a list of floats, a dict of str to int, ...). Lists, dicts and sets passed in are copied: lists of numbers are written back, lists of strings, dicts and sets are read-only. For simple work on big lists the copy still costs something (variance of 100k floats: 4–8×, NumPy arrays: much more). Looping over a set isn't compiled (no reproducible order); `sorted(s)` is.
- **Strings:** `lower`/`upper`/`is...` are native for ASCII; other characters replay in Python (same result, slower).
- **float32** needs NumPy 2; mixing float32 with values from float64 arrays runs as Python.
- **`random`:** only the shared generator's functions listed above; not inside `prange`, and don't draw random numbers from another thread during a compiled call.
- **`min`/`max` over mixed int and float**, `round(x, ndigits)` and `math.hypot` aren't compiled (Python's results depend on details native code can't match exactly).
- **NumPy scalars:** in Python, `a[i]` gives `np.float64`; PyRust returns a plain `float` with the same value.
- **Recursion** needs a return annotation and runs natively up to 300 levels deep.
- **Per-call overhead:** about 0.2–0.4 µs for calls with numbers, NumPy arrays, `str` or `bytes` returning a number, bool or `None` (CPython calls the native code directly); 1–3 µs otherwise (lists, dicts, returned strings or arrays, objects; via `ctypes`). A plain Python call costs about 0.05 µs, so loop-free functions still run as Python by default.
- **First call compiles** (0.1–0.5 s, cached on disk). Use `fn.prepare()` or ship [prebuilt](#deploy-without-rust) libraries.
- **Arrays:** 1-D and 2-D, C-contiguous `float64` / `int64` / `float32`. pandas Series need a `float64`/`int64`/`float32`/`str` dtype, no missing values and the default index; polars Series need no nulls. Series of str only support `s[i]`, `len`, loops and `sorted/list/set/min/max/join` (Series operators work element by element and run as Python).
- **`batch`/`tokenize_many` with the GIL** only run the native part in parallel; converting text in and out still takes turns (`tokenize_many`: 1.7–1.9×).
- **Text columns in `@vectorize`** gain 3.4–4.5× over pandas `map` and 1.5–2.2× over polars `map_elements`; number and bool columns gain 20× or more.
- **`prange`** only accepts independent iterations (see above).
- **Experimental.** Not production-hardened and not a sandbox: only run trusted code.

## Correctness evidence

- **272 automated tests**, passing on Linux, macOS and Windows with Python 3.10, 3.12, 3.13, 3.14 and 3.15 (15 CI jobs) and free-threaded Python 3.14t and 3.15t (6 jobs), plus lint (ruff), type checks (mypy) and a package build check on every push.
- **Edge cases** for every operation: NaN, ±0.0, ±inf, overflow, math domain errors, huge exponents and shifts, int/float comparisons around 2**53 and 2**63, writes that fail halfway, overlapping and read-only arrays, recursion depth, Ctrl-C on long calls, prebuilt libraries without Rust.
- **Generated-program fuzzing:** random programs using the supported operators and `math` functions are compiled and compared with CPython on all three operating systems, including which exception is raised.
- **Benchmark tracking:** every CI job compares its speedups with the last release and flags drops in the job summary.

Details: [`VALIDATION.md`](https://github.com/surajmhaske02/pyrust-engine/blob/main/VALIDATION.md).

## How it works

```
your function ─► AST check, per argument types ─► Rust source (checked int ops, exact float rules)
   ─► rustc (cached on disk, or a prebuilt library) ─► direct CPython call (numbers) or ctypes call
   ─► status 0: return the result
   ─► status ≠ 0 (overflow, ÷0, bounds, math error, panic, ...): undo array writes, replay in Python
   ─► more than 50M steps: re-run in a background thread with no limit (Ctrl-C stops it)
```

Generated Rust never contains raw user text: identifiers are mapped to internal names and only allowed AST nodes are translated.

## Safety

- Run **trusted code only**. Compiled libraries run inside your process; this is not a sandbox.
- Keep the cache private (it must be owned by you and not group/world-writable on POSIX).
- Don't change arrays, function code or globals from another thread while a call is running.

## Project layout

```
pyrust_engine/   compiler.py (Python AST → Rust), runtime.py (decorator, dispatch), cache.py,
                 compiler_text.py + rust_text.py (strings, lists, dicts, sets), kinds.py, packing.py,
                 vectorize.py (@vectorize), ai.py (AI library), advisor.py (suggest/apply), mcp.py (MCP server),
                 toolchain.py (Rust detection, doctor), prebuilt.py (ahead-of-time builds),
                 ipython.py (Jupyter magics), profiling.py, __main__.py (CLI), py.typed
tests/           test_engine.py, test_v2.py … test_v73.py, test_ai.py, test_tools.py, test_toolchain.py, test_fuzz.py
examples/        pyrust_vs_python.ipynb, rag_retriever.py, fastapi_app.py, eval_metrics.py, docker/,
                 benchmark_suite.py, workloads.py
benchmarks/      raw results per release and platform, compare.py (regression report)
docs/            documentation website (MkDocs), API.md, benchmark chart
```

## Roadmap

1. **Direct calls for more types:** lists, dicts, objects and returned strings or arrays still go through `ctypes` (1–3 µs).
2. **More of classes:** objects inside objects (`self.pos.x`), lists of objects, assigning new lists to attributes.
3. **More arrays:** arithmetic on `int32`/`uint8` values with NumPy's wrap-around rules, 3-D arrays, column slices `m[:, j]`.
4. **LSH buckets for `minhash`** (near-duplicates in near-linear time), MMR and rank fusion helpers.
5. **Faster dense array writes** (an optimised path for functions that fill whole arrays).
6. **Benchmarks on ARM Linux**, and against Cython and PyPy.

## License

[MIT](https://github.com/surajmhaske02/pyrust-engine/blob/main/LICENSE) © 2026 Suraj Mhaske
