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
- 🧰 **Use everyday Python values:** lists, strings, `bytes`, pandas columns and the `random` module compile too, with the same results as plain Python.
- 🐼 **Speed up small functions over whole columns** with `@vectorize`: a faster replacement for `df.apply` and list comprehensions.
- 🧵 **Use every CPU core** with `prange()` for loops whose iterations are independent.
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

More everyday Python (new in 0.7.1):

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
```

> **Status: experimental.** Not a general Python compiler and not production-hardened. 168 tests pass in CI on Linux, macOS and Windows with Python 3.10 to 3.15.

## Comparison at a glance

| | Plain Python | **PyRust 0.7** |
|---|---|---|
| Integer loop (500k) | 34.8 ms · 1× | **1.12 ms · 31×** |
| Float recurrence (300k) | 21.4 ms · 1× | **1.56 ms · 14×** |
| Array kernel, read (100k float64) | 10.3 ms · 1× | **0.122 ms · 85×** |
| Write into an output array (100k) | 33.8 ms · 1× | **0.380 ms · 89×** |
| Recursive `fib(25)` | 7.26 ms · 1× | **0.596 ms · 12×** |
| `math.sqrt` / `sin` / `log` loop (300k) | 41.0 ms · 1× | **3.74 ms · 11×** |
| Helper function called in a loop (300k) | 47.6 ms · 1× | **0.745 ms · 64×** |
| Bit-mixing hash with `^ & << >> **` (300k) | 74.4 ms · 1× | **0.662 ms · 112×** |
| Edit distance of two strings (288 chars each) · *0.7.1* | 11.2 ms · 1× | **0.758 ms · 15×** |
| Monte Carlo with `random.random()` (200k) · *0.7.1* | 24.9 ms · 1× | **3.38 ms · 7.4×** |
| Tiny function over 100k rows: `@vectorize` vs a Python loop · *0.7.1* | 13.4 ms · 1× | **0.653 ms · 20×** |
| Variance of a Python list (100k floats, copy included) · *0.7.1* | 6.07 ms · 1× | **3.12 ms · 1.9×** |
| Tiny `add(a, b)` per call | ≈1× | **≈1×** (runs as plain Python on purpose; use `@vectorize` for columns) |
| JSON → array pipeline | 7.75 ms · 1× | **5.39 ms · 1.44×** |
| Big integers / overflow / math errors | Exact | **Exact (replays in Python)** |
| Unsupported code | Runs | **Runs in Python, with the line and a hint** |
| Runtime dependencies | None | **None** (NumPy for arrays; Rust only to compile) |
| First call (compile) | None | 0.1–0.5 s (about 0.005 s from the disk cache) |

<sub>"time · speedup vs Python", measured by CI on a GitHub Actions `ubuntu-latest` runner (x86_64, Python 3.13, Rust 1.99). Shared CI machines are noisy; treat the numbers as approximate. Raw data: [`benchmarks/`](https://github.com/surajmhaske02/pyrust-engine/tree/main/benchmarks).</sub>

![Speedup versus plain Python](https://cdn.jsdelivr.net/gh/surajmhaske02/pyrust-engine@main/docs/benchmark-speedup.svg)

### Speed on every platform

Speedup over plain Python, same CI run, Python 3.13 and Rust 1.99:

| Workload | Linux x86_64 | macOS Apple Silicon | Windows x86_64 |
|---|---:|---:|---:|
| Integer loop | 31× | 39× | 29× |
| Float recurrence | 14× | 5.7× | 7.8× |
| Array kernel (read) | 85× | 42× | 86× |
| Write into an output array | 89× | 73× | 44× |
| Recursive `fib(25)` | 12× | 13× | 12× |
| `math` functions loop | 11× | 14× | 15× |
| Helper function in a loop | 64× | 91× | 83× |
| Bit-mixing hash | 112× | 72× | 104× |
| Tiny `add(a, b)` (runs as Python) | ≈1× | ≈1× | ≈1× |

Float recurrences and `math`-heavy loops gain less because the floating-point work itself costs the same in both; PyRust removes the interpreter overhead around it. A comparison with Numba is on the [benchmarks page](https://surajmhaske02.github.io/pyrust-engine/benchmarks/).

## What compiles

| | |
|---|---|
| **Values** | `int` (64-bit, overflow-checked), `float`, `bool`, tuples (unpacking and multiple returns), NumPy `float64`/`int64` arrays (1-D and 2-D), lists of floats or ints, `str`, `bytes`, pandas `float64`/`int64` Series |
| **Operators** | `+ - * / // % **`, `& \| ^ ~ << >>`, comparisons incl. chained and exact int-vs-float |
| **Control flow** | `if`/`elif`/`else`, `for` over `range`, an array, `enumerate` or `prange`, `while`, `break`, `continue`, `return`, `x if c else y` |
| **Builtins** | `abs min max sum round divmod pow int float bool len ord chr` |
| **Strings** | `s[i]`, `for ch in s`, comparisons, `ch in "aeiou"`, `ch.isdigit()` / `isalpha` / `isspace` / `lower()` / … (read-only) |
| **`random`** | `random uniform randint randrange gauss choice`, exactly Python's sequence |
| **Lists** | read, loop, write (copied back into your list), new lists with `[0.0] * n` that can be returned |
| **pandas** | Series `s[i]`, `s.iloc[i]`, `len(s)`, `for x in s`; `@vectorize` over whole columns |
| **`math`** | `sqrt exp log log2 log10 sin cos tan asin acos atan atan2 sinh cosh tanh fabs copysign fmod floor ceil trunc isnan isinf isfinite pi e tau inf nan` |
| **Arrays** | `a[i]`, `a[i, j]`, `a.shape`, `len(a)`, `for x in a`, writes (`a[i] = v`, `+=`), new arrays with `np.zeros/empty/ones(_like)` that can be returned |
| **Functions** | calls to your own functions (keyword and default arguments, recursion with a `-> int`/`-> float` annotation), module-level constants |

Everything else (dicts, sets, building strings, list methods, classes, `print`, `try`, slicing, …) still runs, as plain Python, and `explain()` gives a hint. Full details: [Guide](https://surajmhaske02.github.io/pyrust-engine/guide/).

## What makes Python slow, and what PyRust does about it

| Why Python is slow | Example | PyRust 0.7 |
|---|---|---|
| Every operation is interpreted one bytecode at a time | loops | ✅ Compiled to native machine code |
| Every number is a heap object with type checks | `total += i * i` | ✅ Plain 64-bit ints, floats and bools in CPU registers |
| Function calls are expensive | helpers in a loop, recursion | ✅ Helpers and recursion compiled into the same native code |
| Math functions go through Python objects | `math.sqrt(x)`, `x ** 0.5` | ✅ Native calls, same results |
| Bit manipulation creates new int objects | hashing | ✅ Native bit operations |
| Indexing arrays goes through NumPy objects | `a[i]`, `out[i] = x` | ✅ Reads and writes array memory directly |
| The GIL keeps one thread busy | CPU-bound loops | ✅ `prange()` runs independent iterations on all cores |
| Crossing into native code has a fixed cost | tiny functions called millions of times | ✅ `@vectorize` runs them over whole arrays in one call; ⚠️ single calls still cost 2–3 µs |
| Python values are generic objects | lists of numbers, text processing | ✅ Lists, `str`, `bytes` and pandas columns read as native memory |
| Random numbers go through Python calls | simulations, Monte Carlo | ✅ Python's own generator algorithm, natively, same sequence |
| Dicts, sets and objects are dynamic | JSON, classes, grouping | ❌ Not compiled. Use the right library, PyPy or CPython's JIT |
| I/O waits | network, disk, database | ❌ Not a CPU problem: use `asyncio` or threads |

## Version comparison

| | 0.5 | 0.6 | **0.7** |
|---|---|---|---|
| Values | `int`, `float`, `float64` arrays | + `int64` arrays | **+ `bool`, tuples, 2-D arrays, new arrays** |
| Loops | `range`, `while` | same | **+ `for x in a`, `enumerate`, `prange` (multi-core)** |
| Builtins | `abs min max int pow`, `math.*` | same | **+ `sum round divmod bool`, `atan2 copysign fmod`** |
| Functions | helper calls | + recursion | **+ keyword/default arguments, helpers may write arrays** |
| Argument types | from annotations | same | **compiled per type actually passed; annotations optional** |
| Long calls | 10M-step budget, then slow replay | same | **no limit, background thread, Ctrl-C works** |
| Array writes | — | full copy per call | **undo log (no copy for sparse writes)** |
| Crashes in generated code | abort | abort | **caught, replayed in Python** |
| Deployment | needs Rust | + `doctor`, clear toolchain errors | **+ `build` ahead of time: no Rust needed at run time** |
| Developer tools | `explain()`, warnings, `check` | + type hints, API docs | **+ hints in messages, Jupyter magics, docs website** |
| Tests / CI | 89 / 3 OSes × 3 Pythons | 128 | **150 / 3 OSes × Python 3.10–3.15, lint, coverage, benchmark tracking** |

**0.7.1 (current)** adds lists, `str`/`bytes`, pandas Series, the `random` module and `@vectorize`. Older releases: [Changelog](https://github.com/surajmhaske02/pyrust-engine/blob/main/CHANGELOG.md).

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

### Web apps

Compiled calls don't hold the GIL, so a CPU-heavy step in a FastAPI/Django/Flask endpoint can run in a worker thread (`await asyncio.to_thread(fn, ...)`) without blocking other requests. Ship [prebuilt](#deploy-without-rust) libraries so servers don't need Rust.

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

- **A subset of Python.** No dicts, sets, list methods (`append`), building strings, classes, methods, closures, generators, `try`/`with`, slicing or `print` in compiled code (they run as Python).
- **Lists** are copied for each call. For simple work on big lists the copy dominates (variance of 100k floats: 1.9×); NumPy arrays avoid it. Strings, `bytes` and pandas Series are read-only.
- **`random`:** only the shared generator's functions listed above; not inside `prange`, and don't draw random numbers from another thread during a compiled call.
- **`min`/`max` over mixed int and float**, `round(x, ndigits)` and `math.hypot` aren't compiled (Python's results depend on details native code can't match exactly).
- **NumPy scalars:** in Python, `a[i]` gives `np.float64`; PyRust returns a plain `float` with the same value.
- **Recursion** needs a return annotation and runs natively up to 300 levels deep.
- **Per-call overhead** of about 2–3 µs for native calls (via `ctypes`). Loop-free functions skip it; loops with tiny inputs don't.
- **First call compiles** (0.1–0.5 s, cached on disk). Use `fn.prepare()` or ship [prebuilt](#deploy-without-rust) libraries.
- **Arrays:** 1-D and 2-D, C-contiguous `float64` / `int64` only. pandas Series need a `float64`/`int64` dtype and the default index.
- **`prange`** only accepts independent iterations (see above).
- **Experimental.** Not production-hardened and not a sandbox: only run trusted code.

## Correctness evidence

- **168 automated tests**, passing on Linux, macOS and Windows with Python 3.10, 3.12, 3.13, 3.14 and 3.15 (15 CI jobs), plus lint (ruff), type checks (mypy) and a package build check on every push.
- **Edge cases** for every operation: NaN, ±0.0, ±inf, overflow, math domain errors, huge exponents and shifts, int/float comparisons around 2**53 and 2**63, writes that fail halfway, overlapping and read-only arrays, recursion depth, Ctrl-C on long calls, prebuilt libraries without Rust.
- **Generated-program fuzzing:** random programs using the supported operators and `math` functions are compiled and compared with CPython on all three operating systems, including which exception is raised.
- **Benchmark tracking:** every CI job compares its speedups with the last release and flags drops in the job summary.

Details: [`VALIDATION.md`](https://github.com/surajmhaske02/pyrust-engine/blob/main/VALIDATION.md).

## How it works

```
your function ─► AST check, per argument types ─► Rust source (checked int ops, exact float rules)
   ─► rustc (cached on disk, or a prebuilt library) ─► ctypes call
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
                 vectorize.py (@vectorize), toolchain.py (Rust detection, doctor), prebuilt.py (ahead-of-time builds),
                 ipython.py (Jupyter magics), profiling.py, __main__.py (CLI), py.typed
tests/           test_engine.py, test_v2.py … test_v71.py, test_toolchain.py, test_fuzz.py
examples/        pyrust_vs_python.ipynb, benchmark_suite.py, workloads.py
benchmarks/      raw results per release and platform, compare.py (regression report)
docs/            documentation website (MkDocs), API.md, benchmark chart
```

## Roadmap

1. **Faster calls:** a CPython extension entry point instead of `ctypes`, to bring the 2–3 µs per native call well under 1 µs.
2. **Dicts and sets** of numbers inside compiled functions (counting, grouping, memoization).
3. **More array types:** `float32`, `int32`, `uint8`, `bool`, 3-D arrays and slices like `a[i, :]`.
4. **Faster dense array writes** (an optimised path for functions that fill whole arrays).
5. **Benchmarks on ARM Linux**, and against Cython and PyPy.

## License

[MIT](https://github.com/surajmhaske02/pyrust-engine/blob/main/LICENSE) © 2026 Suraj Mhaske
