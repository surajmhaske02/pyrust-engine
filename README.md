# PyRust Engine

[![Tests](https://github.com/surajmhaske02/pyrust-engine/actions/workflows/test.yml/badge.svg)](https://github.com/surajmhaske02/pyrust-engine/actions/workflows/test.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.12%20%7C%203.13-blue)
![Status](https://img.shields.io/badge/status-experimental-orange)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/pyrust-engine)](https://pypi.org/project/pyrust-engine/)

**Make your Python code faster by adding one line.**

Put `@optimize` above a function. PyRust turns that function into Rust, compiles it and runs it as native machine code. You keep writing normal Python, and **your answers never change**: anything PyRust can't run safely runs as normal Python instead.

- ✍️ **Write plain Python.** No new language, no build step, no rewrite.
- ⚡ **Get real speed on calculation-heavy code:** math functions, arithmetic, bit operations, helper functions, NumPy arrays and repeated work run about **5× to 180× faster** in our tests on Linux, macOS and Windows.
- 🛟 **Never get a wrong answer.** Big integers, math errors and anything unsupported fall back to normal Python with the exact same result.
- 🔍 **See what's happening.** `fn.explain()` and a one-time warning tell you which line can't be sped up, and why.

📖 [API reference](docs/API.md) · 📓 [Try it in a notebook](examples/pyrust_vs_python.ipynb) · 🤝 [Contributing](CONTRIBUTING.md) · 📝 [Changelog](CHANGELOG.md)

Good fits: scientific and engineering formulas, finance and statistics, simulations, games and physics, signal processing, hashing and checksums, and number-heavy data processing.

```python
import math
from pyrust_engine import optimize

def distance(x: float, y: float) -> float:
    return math.sqrt(x * x + y * y)          # helpers are compiled together with their caller

@optimize
def total_distance(n: int) -> float:
    total = 0.0
    for i in range(n):
        total += distance(float(i), float(n - i))
    return total

@optimize
def checksum(n: int) -> int:
    h = 2166136261
    for i in range(n):
        h = ((h ^ i) * 16777619) & (2 ** 32 - 1)   # integer math and bit operations
    return h

total_distance(300_000)          # first call compiles (~0.1 s), then runs as native code
total_distance(300_000)          # ~68x faster than plain Python in our benchmark
checksum(300_000)                # same answer as plain Python, at native speed
print(total_distance.explain())  # does it compile? if not, which line and why
```

> **Status: experimental prototype.** Not a general Python compiler and not production-hardened. Tests pass in CI on Linux, Windows and macOS (Python 3.10, 3.12, 3.13), and benchmarks are published for all three.

## Comparison at a glance

| | Plain Python | **PyRust 0.6** |
|---|---|---|
| Integer loop (500k) | 43.0 ms · 1× | **1.20 ms · 35.9×** |
| Float recurrence (300k) | 21.8 ms · 1× | **1.82 ms · 12.0×** |
| Array kernel, read (100k float64) | 12.9 ms · 1× | **0.174 ms · 73.8×** |
| Write into an output array (100k) · *new in 0.6* | 44.1 ms · 1× | **0.247 ms · 179×** |
| Recursive `fib(25)` · *new in 0.6* | 9.71 ms · 1× | **0.642 ms · 15.1×** |
| `math.sqrt` / `sin` / `log` loop (300k) | 52.2 ms · 1× | **4.92 ms · 10.6×** |
| Helper function called in a loop (300k) | 57.7 ms · 1× | **0.848 ms · 68.1×** |
| Bit-mixing hash with `^ & << >> **` (300k) | 94.8 ms · 1× | **0.754 ms · 126×** |
| Tiny `add(a, b)` per call | 0.069 µs · 1× | **0.059 µs · ≈1×** (runs as plain Python) |
| JSON → array pipeline | 9.20 ms · 1× | **6.72 ms · 1.37×** |
| Big integers / overflow / math errors | Exact | **Exact (replays in Python)** |
| Unsupported code | Runs | **Runs in Python, with a warning that says which line** |
| Extra runtime dependencies | None | **None** (NumPy for arrays) |
| First call (compile) | None | 0.10–0.16 s (0.02 s from disk cache) |

<sub>"time · speedup vs Python", measured by CI on a GitHub Actions `ubuntu-latest` runner (x86_64, Python 3.13, Rust 1.99). Rows marked *new in 0.6* ran as plain Python (1×) in 0.5 and earlier. macOS and Windows numbers are under [Speed on every platform](#speed-on-every-platform).</sub>

## Version comparison

| | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | **0.6 (current)** |
|---|---|---|---|---|---|---|
| Types | `int` | + `float`, `float64` arrays | same | same | same | **+ `int64` arrays** |
| Arithmetic | `+ - *` | + `// % /` | + unary `+` | same | + `**`, `& \| ^ ~ << >>`, float `//` `%` | same |
| Builtins and `math` | — | `len`, `float` | same | same | + `abs min max int pow`, `math.*` | same |
| Calling other functions | — | — | — | — | Yes | **+ recursion** |
| Arrays | — | read | read | read | read | **read and write** |
| Comparisons | single | + `and/or/not` | + chained, truthiness | same | same | **+ exact int vs float** |
| Other syntax | `for`, `if` | + `while`, `break`, `continue` | + `elif` | same | + ternary, `while True`, globals | **+ functions returning `None`** |
| Tiny `add(a, b)` per call | 3.3 µs | 5.3 µs | 2.0 µs | ≈ Python | ≈ Python | ≈ Python |
| Developer tools | `report()` | + `prepare`, `reset`, profiler | same | + `native_loop_free` | + `explain()`, warnings, `check` | **+ type hints, API docs** |
| Automated tests | 12 | 34 | 47 | 56 | 89 | **112** |
| CI platforms passing | — | — | Linux | all three | all three | all three **+ package build** |
| Published benchmarks | Linux | Linux | Linux | Linux | Linux | **Linux, macOS, Windows** |

---

## What makes Python slow, and what PyRust does about it

Python is slow for a few specific reasons. PyRust removes some of them for the code it compiles. Others need a different tool, and it's better to know which is which.

| Why Python is slow | Example | PyRust 0.6 |
|---|---|---|
| Every operation is interpreted one bytecode at a time | `for` / `while` loops | ✅ Compiled to native machine code |
| Every number is a heap object with a type check | `total += i * i` | ✅ Plain 64-bit ints and floats in CPU registers |
| Function calls are expensive | a helper called inside a loop, recursion | ✅ Helpers and recursive functions are compiled into the same native code (0.5, 0.6) |
| Math functions go through Python objects | `math.sqrt(x)`, `x ** 0.5` | ✅ Native calls, same results (0.5) |
| Bit manipulation creates new int objects | hashing, `x ^ (x >> 13)` | ✅ Native bit operations (0.5) |
| Indexing arrays goes through NumPy objects | `float(a[i])`, `out[i] = x` in a loop | ✅ Reads and writes array memory directly (writes in 0.6) |
| Crossing into native code has a fixed cost | tiny functions called millions of times | ⚠️ ~2 µs per native call; tiny loop-free functions skip it |
| The GIL stops threads running Python in parallel | CPU-bound threads | ⚠️ Native calls release the GIL (ctypes), but there are no parallel loops yet |
| Lists, dicts, strings and objects are dynamic | text processing, JSON, classes | ❌ Not compiled. Use the right library, PyPy or CPython's JIT |
| I/O waits (network, disk, database) | web requests | ❌ Not a CPU problem: use `asyncio` or threads |

## Why it exists

Most "make Python fast" tools ask you to give something up: a new interpreter, a build step, a different integer model, or a heavy dependency. PyRust takes a narrower bet:

- **Correctness first.** Integers stay arbitrary-precision. If a 64-bit operation would overflow, divide by zero, index out of bounds or hit an invalid `range` step, the native call stops and **the original Python function is replayed**, so you get exactly Python's result or Python's exception. Nothing wraps silently.
- **Opt-in, per function.** Decorate the hot loop, leave everything else alone. Unsupported code, inputs or environments run as normal Python and the reason is recorded.
- **Light footprint.** Pure-Python package with no runtime dependencies (NumPy only for array kernels). Uses `rustc` directly through `ctypes`: no PyO3, Maturin, crates or C compiler toolchain beyond Rust's own linker.

## How it compares

### Speed

![Speedup versus plain Python](docs/benchmark-speedup.svg)

| Workload | Python | **PyRust 0.6** | PyRust vs Python |
|---|---:|---:|---:|
| Integer loop (`//`, `%`, branches; n = 500k) | 43.0 ms | **1.20 ms** | **35.9× faster** |
| Float recurrence (n = 300k) | 21.8 ms | **1.82 ms** | **12.0× faster** |
| Array kernel, sum of squares (100k float64) | 12.9 ms | **0.174 ms** | **73.8× faster** |
| Smoothing filter writing an output array (100k) | 44.1 ms | **0.247 ms** | **179× faster** |
| Recursive `fib(25)` | 9.71 ms | **0.642 ms** | **15.1× faster** |
| `math.sqrt(x) * math.sin(x) + math.log(x)` loop (n = 300k) | 52.2 ms | **4.92 ms** | **10.6× faster** |
| Helper `distance(x, y)` called inside a loop (n = 300k) | 57.7 ms | **0.848 ms** | **68.1× faster** |
| Bit-mixing hash with `^ & << >> **` (n = 300k) | 94.8 ms | **0.754 ms** | **126× faster** |
| Tiny function `add(a, b)`, per call (default) | 0.069 µs | **0.059 µs** | **same speed** (plain Python) |
| Tiny function `add(a, b)`, forced native | 0.068 µs | 4.00 µs | 59× slower |
| JSON → NumPy → kernel → JSON pipeline | 9.20 ms | **6.72 ms** | **1.37× faster** |

<sub>Median of 7 warm runs, wrapper and argument conversion included. Measured by CI on a GitHub Actions `ubuntu-latest` runner: Linux x86_64, CPython 3.13.18, Rust 1.99.1, NumPy 2.5.4. Shared CI machines are noisy, so treat these as approximate. Raw samples: [`benchmarks/results-v0.6.0-ci-linux.json`](benchmarks/results-v0.6.0-ci-linux.json); older runs are in [`benchmarks/`](benchmarks/). Workloads are in [`examples/workloads.py`](examples/workloads.py). Reproduce with `python -m examples.benchmark_suite`.</sub>

### Speed on every platform

Speedup over plain Python, same CI run, Python 3.13 and Rust 1.99 on each GitHub Actions runner:

| Workload | Linux x86_64 | macOS Apple Silicon (arm64) | Windows x86_64 |
|---|---:|---:|---:|
| Integer loop | 35.9× | 39.5× | 37.5× |
| Float recurrence | 12.0× | 5.3× | 8.9× |
| Array kernel (read) | 73.8× | 37.8× | 69.7× |
| Write into an output array | 179× | 128× | 152× |
| Recursive `fib(25)` | 15.1× | 17.0× | 19.9× |
| `math` functions loop | 10.6× | 15.3× | 15.5× |
| Helper function in a loop | 68.1× | 87.2× | 61.6× |
| Bit-mixing hash | 126× | 60.0× | 131× |
| Tiny `add(a, b)` (runs as Python) | ≈1× | ≈1× | ≈1× |
| JSON → array pipeline | 1.37× | 1.35× | 1.31× |

<sub>Raw samples: [Linux](benchmarks/results-v0.6.0-ci-linux.json), [macOS](benchmarks/results-v0.6.0-ci-macos.json), [Windows](benchmarks/results-v0.6.0-ci-windows.json). Differences between platforms mostly reflect how fast plain Python is on each machine.</sub>

**How to read this, honestly:**

- PyRust gives **large speedups on number-crunching code** on all three platforms: 36–40× on integer loops, 60–131× on bit operations and 128–179× when writing arrays.
- **New in 0.6:** writing into arrays and recursion now compile. In 0.5 those rows ran as plain Python (1×).
- Math-heavy code gains less (10–15×), because `sqrt`, `sin` and `log` cost the same in both; PyRust only removes the overhead around them. Float recurrences gain 5–12× for the same reason.
- **Tiny functions don't get slower.** Loop-free functions run as plain Python (details in [Tiny functions](#tiny-functions-the-problem-and-the-fix)). A function *with* a loop but a very small input still pays about 2–4 µs per call.
- **End-to-end gains shrink** when parsing and conversion dominate: the JSON pipeline only improved about 1.35×.

### Startup cost

| Workload | Linux: first call incl. compile | Linux: from disk cache | macOS: compile / cache | Windows: compile / cache |
|---|---:|---:|---:|---:|
| Integer loop | 0.113 s | 0.021 s | 0.196 s / 0.027 s | 0.366 s / 0.051 s |
| Recursive `fib` | 0.098 s | 0.020 s | 0.134 s / 0.026 s | 0.296 s / 0.048 s |
| Array write | 0.124 s | 0.020 s | 0.176 s / 0.029 s | 0.334 s / 0.051 s |

PyRust caches compiled libraries on disk across processes. Use `fn.prepare()` (or `prepare(background=True)`) before latency-sensitive work.

### Design trade-offs

This table compares how each tool is designed. It is not a benchmark.

| | **PyRust** | Numba | Cython | PyPy | Codon |
|---|---|---|---|---|---|
| How you use it | Decorator on a function | Decorator on a function | Separate build step (`.pyx` or annotated `.py`) | Different Python interpreter | Separate compiler |
| Integer overflow | **Exact Python result** (replays in Python) | Wraps at 64 bits | Python ints exact; C-typed ints follow C rules (optional overflow checks) | Exact | Wraps at 64 bits |
| Unsupported code | Runs as normal Python, reason recorded | Compile error | Compiles via the C API (slower paths) | Runs everything | Compile error |
| Scope | Numeric subset: ints, floats, `math`, helper calls, read-only arrays | Large NumPy-oriented subset | Full Python | Full Python | Large Python-like subset |
| Runtime dependencies | None (NumPy for arrays) | numba + llvmlite | None after build | Whole runtime | Codon runtime |
| Toolchain needed | `rustc` | none extra | C compiler | none extra | Codon |
| Maturity | Experimental | Mature | Mature | Mature | Active project |

**Pick PyRust** when you want Python-exact results (including big integers and math errors), a safety net that never changes your answers and no compiled dependency stack, and your hot spots are numeric loops, math and helper functions. **Pick Numba or NumPy** for maximum speed on array maths. **Pick Cython** when you need to compile large or object-heavy code. **Pick PyPy** to speed up a whole pure-Python application.

## Tiny functions: the problem and the fix

**The problem (0.3.0 and earlier).** `add(a, b)` took 2.08 µs through PyRust against 0.034 µs in plain Python: 61× slower. The Rust code isn't the slow part. The add itself takes about a nanosecond. The cost is the trip from Python into native code on every call:

| Step per call | What happens |
|---|---|
| Wrapper checks | Code/defaults unchanged? Builtins not rebound? Library loaded? |
| Argument checks | Each argument must be an exact `int` that fits 64 bits, or an exact `float` |
| `ctypes` conversion | Python objects → C values, plus an output buffer for the result |
| Native call | Status code back, then the result is read out of the buffer |
| Metrics | Timers and counters updated under a lock |

Together that is about 2 µs. A loop of 500,000 iterations easily pays it back. A single addition never can.

**The fix (0.4.0).** At decoration time, PyRust checks the function's AST. A function with **no `for` or `while` loop** does a fixed, tiny amount of work, so native code can't win. For those functions `@optimize` returns a plain Python function (a clone of the original with `report()`, `prepare()`, `reset()` and `rust_source()` attached). There is no wrapper, so a call costs exactly what the original costs, and results are the original's by construction. Since 0.5, a loop-free function that calls another function (for example `return helper_sum(n) + 1`) is compiled normally, because the call can hide real work.

```python
@optimize
def add(a, b):
    return a + b

add(2, 3)                    # plain Python speed
add.report()['dispatch']     # 'python-direct'
add.report()['reason']       # 'loop-free function: a native call costs more than ...'

@optimize(native_loop_free=True)   # force Rust anyway (e.g. to test the native path)
def add_native(a, b):
    return a + b
```

Trade-off: a loop-free function has no wrapper, so `report()` can't count its calls (`call_counting` is `False`).

**What still costs ~2 µs.** A function that contains a loop but is called with tiny inputs (say `n = 3`), or a tiny function called millions of times from a Python loop. Best practice: decorate the function that **contains the hot loop**. Since 0.5, small helpers it calls are compiled into the same native code, so they cost nothing extra. The remaining fix is on the [roadmap](#roadmap): a CPython extension call path instead of `ctypes`.

## Limitations

- **Numeric subset only.** `int` (64-bit, overflow-checked), `float`, 1-D `float64` / `int64` arrays, `math` functions and calls to other numeric functions (including recursion). No lists, dicts, strings, tuples, classes or generators: those run as plain Python.
- **`min` / `max` over mixed int and float values** aren't compiled, because Python returns the winning argument with its own type.
- **Recursion** needs a return annotation (`-> int` / `-> float`) and runs natively up to 300 levels deep; deeper calls replay in Python.
- **Per-call overhead of ~2–4 µs** for native calls. Loop-free functions avoid it; loops with tiny inputs don't.
- **First call compiles** (about 0.1–0.4 s per function depending on the OS, 0.02–0.05 s from the disk cache). Use `fn.prepare()` ahead of time.
- **Arrays:** only 1-D, C-contiguous `float64` / `int64`. Writing an array copies it once per call (copy-on-write, so failures can be replayed exactly). Only the decorated function itself can write arrays, not helpers it calls.
- **Needs `rustc` on `PATH`** on every machine that should run natively (no prebuilt wheels).
- **Native loops can't be interrupted** by Ctrl-C until they finish or hit `max_iterations`.
- **No parallel loops.** One call uses one CPU core.
- **Benchmarks come from GitHub's CI machines** (Linux x86_64, macOS arm64, Windows x86_64), which are shared and noisy. There are no numbers for ARM Linux yet.
- **Experimental.** Not production-hardened, not a sandbox: only run trusted code.

## Correctness evidence

- **112 automated tests**, passing in CI on Linux, Windows and macOS with Python 3.10, 3.12 and 3.13, covering semantics, recursion (depth and budget limits), array writes (failure halfway through, overlapping and read-only arrays), `int64` arrays, exact int/float comparisons at the 2**53 and 2**63 boundaries, every 0.5 operator and `math` function at its edge cases (NaN, ±0.0, ±inf, overflow, domain errors, huge exponents and shifts), calls between functions, rebinding of globals and helpers, overflow replay, exceptions, NaN, ±0.0, infinities, array layouts, bounds, cache corruption and recovery, cross-process locking, background compilation, threads and profiling.
- **Generated-program fuzzing:** 4,000 random programs (2,000 integer + 2,000 float, using `while`, `break`, `continue`, `elif`, `//`, `%`, `/`), **32,000 calls compared with CPython, 0 mismatches**. Floats are compared by `repr`, so NaN and signed zero must match exactly. Every generated program must compile natively, so the test can't pass by quietly falling back to Python.
- **0.5 fuzzing:** a second generator adds `**`, `& | ^ ~ << >>`, `abs`, `min`, `max`, `pow`, ternaries, float `//` `%` `**`, `int()`, `math.floor/ceil` and nine `math` functions. Its programs run on all three operating systems in CI and must match CPython exactly, including which exception is raised.
- **Fuzzer sensitivity:** deliberately planted bugs (wrong modulo sign; `a / b` rewritten as `a * (1/b)`) were caught immediately.

Details and limitations: [`VALIDATION.md`](VALIDATION.md).

## Install

Requires **Python 3.10+** and **Rust** (`rustc` on `PATH`, from <https://rustup.rs>).

```sh
pip install pyrust-engine              # core, no dependencies
pip install "pyrust-engine[arrays]"    # + NumPy, for F64Array / I64Array
```

From source, with the tests and benchmarks:

```sh
git clone https://github.com/surajmhaske02/pyrust-engine.git
cd pyrust-engine
python -m pip install ".[arrays]"          # NumPy for array kernels
python -m unittest discover -s tests -v    # 112 tests
python -m examples.benchmark_suite         # add ".[benchmark]" for Numba comparison
```

Without Rust, everything still works in plain Python; `report()` says why.

## Usage

### Integer and float kernels

Save functions in a `.py` file (source must be inspectable). Annotations pick native types: unannotated or `int` means signed 64-bit integer; `float` means f64.

```python
from pyrust_engine import optimize

@optimize
def recurrence(n: int, value: float) -> float:
    for i in range(n):
        value = (value * 1.000001 + 0.0001) / 1.000002
    return value
```

Arguments must be exact `int` (fitting 64 bits) or exact `float`. Anything else, including `bool`, int subclasses or an int passed to a `float` parameter, runs in Python. A local variable keeps one type: start a float accumulator at `0.0`, not `0`.

### NumPy arrays: read and write

```python
import numpy as np
from pyrust_engine import optimize, F64Array, I64Array

@optimize
def energy(values: F64Array) -> float:          # read
    total = 0.0
    for i in range(len(values)):
        x = float(values[i])
        total += x * x
    return total

@optimize
def smooth(values: F64Array, out: F64Array):    # write into an output array (0.6)
    n = len(values)
    for i in range(n):
        left = float(values[i - 1]) if i > 0 else float(values[i])
        right = float(values[i + 1]) if i < n - 1 else float(values[i])
        out[i] = (left + 2.0 * float(values[i]) + right) / 4.0

@optimize
def histogram(labels: I64Array, counts: I64Array, bins: int):   # int64 arrays (0.6)
    for i in range(len(labels)):
        counts[int(labels[i]) % bins] += 1

data = np.random.default_rng(0).standard_normal(100_000)
out = np.empty_like(data)
energy(data); smooth(data, out)
```

Accepted arrays: exact `ndarray`, 1-D, C-contiguous, aligned, native-endian `float64` (`F64Array`) or `int64` (`I64Array`). Others run in Python. Read with `float(a[i])` / `int(a[i])`; write with `a[i] = value`, `a[i] += value`, `-=`, `*=` (and `/=` for floats). Negative indices work; out-of-bounds access raises Python's `IndexError` via replay.

Writes go to a private copy that is copied back only when the whole native call succeeds, so if anything fails halfway, Python re-runs the function on your original data and you see exactly what plain Python would do. Read-only arrays, and written arrays that share memory with another argument, run as plain Python.

### Recursion

```python
@optimize
def fib(n: int) -> int:          # the return annotation is required for recursion
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

fib(30)                          # ~15-20x faster than plain Python in our benchmarks
```

Mutual recursion works too. Native recursion goes up to 300 levels deep, and each call counts toward `max_iterations`; deeper or longer runs replay in Python with the same result (including `RecursionError`).

### Supported syntax

| Construct | Native support |
|---|---|
| Parameters | `int` (default), `float`, `F64Array`, `I64Array` |
| Integer ops | `+ - *`, unary `- + ~`, `//` and `%` with Python floor semantics, `**` (non-negative exponent), `& \| ^ << >>`; all overflow-checked. `int / int` gives a float like Python |
| Float ops | `+ - * / // % **`, unary `- +`, using CPython's exact rules for signed zero, NaN, infinities and errors; mixed int/float promotes like Python; no fast-math |
| Builtins | `abs`, `min(a, b, ...)`, `max(a, b, ...)`, `int(x)`, `float(x)`, `pow(a, b)`, `len(array)` |
| `math` | `sqrt exp log log2 log10 sin cos tan asin acos atan sinh cosh tanh fabs`, `floor ceil trunc` (return int), `isnan isinf isfinite` (in conditions), `pi e tau inf nan`. Works as `math.sqrt(x)` or `from math import sqrt` |
| Function calls | Calls to other plain or `@optimize` functions with int/float/array arguments, compiled into the same library. Recursion and mutual recursion with a `-> int` / `-> float` annotation |
| Globals | Module-level `int` / `float` constants (if one is reassigned later, calls fall back to Python) |
| Loops | `for` over `range(stop)`, `range(start, stop)`, `range(start, stop, step)` (negative steps too); `while`; `break`; `continue` |
| Branches | `if` / `elif` / `else`, early `return` |
| Conditions | Comparisons incl. chained `a < b < c`; exact int vs float (`i < x`); `and`, `or`, `not`; truthiness `if n:`; `while True`; `a if cond else b` |
| Locals | Assignment, annotated assignment, `+= -= *= //= %= /=` |
| Arrays | `float(a[i])`, `int(a[i])`, `a[i] = v`, `a[i] += v` (`-= *= /=`), `len(a)`; negative indices; bounds-checked |
| Returns | One scalar type (`int` or `float`), or no value (`None`) |

These run as normal Python instead: lists, dicts, strings, tuples, closures, generators, classes, keyword arguments in calls. Variables first assigned inside a loop aren't treated as assigned after it, which is conservative by design. When something isn't supported, `fn.explain()` and the fallback warning give the line and the code.

### Control and diagnostics

```python
fn.prepare()                    # compile now
fn.prepare(background=True)     # compile in a thread; calls use Python meanwhile
fn.reset()                      # clear compile state and retry
fn.report()                     # calls, native/fallback counts, timings, fallback_reasons
fn.rust_source()                # inspect the generated Rust
fn.explain()                    # human-readable: compiles or not, which line and why (0.5)
fn.python(...)                  # call the original function

@optimize(fallback=False)            # raise OptimizationError instead of falling back (good for benchmarks)
@optimize(max_iterations=10**9)      # native loop budget (default 10,000,000)
@optimize(native_loop_free=True)     # compile loop-free functions too (default: run them as plain Python)
@optimize(warn_on_fallback=False)    # silence PyRustFallbackWarning (default: warn once per reason)
```

When a call runs as plain Python, PyRust warns once per reason, for example:

```text
PyRustFallbackWarning: simulate ran as plain Python: simulate, line 14: unsupported expression: `[x, y]`
```

`fn.explain()` prints the same information on demand:

```text
simulate: compiles to Rust.
  parameters: n: int, dt: float
  returns: float
  compiled together with: step, energy
```

> **Iteration budget:** a call that exceeds `max_iterations` discards its native work and **replays the whole call in Python**, losing the speedup. With 12M iterations: Python 0.450 s, default budget 0.384 s, `max_iterations=10**9` 0.015 s. Raise it for known large loops; watch `report()['fallback_reasons']`.

### Find candidates in existing code

```sh
python -m pyrust_engine check path/to/module.py               # which functions compile, and why not (0.5)
python -m pyrust_engine profile path/to/script.py arg1 arg2   # runs the script once (trusted code only)
python -m pyrust_engine cache --days 30                       # list old cache entries (add --delete)
```

`check` loads the file (without running its `if __name__ == "__main__":` block) and explains every function in it; add `--json` for tools. The profiler ranks functions in the entry script by time and reports which ones fit the native subset. Neither rewrites your code.

## How it works

```
Python source ─► AST check & type inference ─► Rust source (checked i64 / f64 ops)
      ─► rustc --crate-type cdylib (cached, hashed, locked) ─► ctypes call
      ─► status 0: return native result
      ─► status ≠ 0 (overflow, ÷0, bounds, budget, no return): replay original Python
```

The supported subset has no side effects, so replaying a call is always safe. Generated Rust never contains raw user text: identifiers are mapped to internal names and only allowed AST nodes are translated.

## Safety

- Run **trusted code only**. Compiled libraries run inside your process; this is not a sandbox.
- Keep the cache private (it must be owned by you and not group/world-writable on POSIX).
- Long native loops can't be interrupted by Ctrl-C until they finish or hit the iteration budget.
- Don't mutate arrays, function code or builtins concurrently with a running call.

## Project layout

```
pyrust_engine/   compiler.py (AST → Rust), runtime.py (decorator, dispatch, metrics, typing),
                 cache.py (disk cache, locks), profiling.py, __main__.py (CLI), py.typed
tests/           test_engine.py, test_v2.py … test_v6.py, test_fuzz.py (generated programs)
examples/        pyrust_vs_python.ipynb (try it), benchmark_suite.py, workloads.py
benchmarks/      raw benchmark results (0.2.0 – 0.6.0; 0.6.0 for Linux, macOS and Windows)
docs/            API.md (reference), benchmark chart and the script that generates it
.github/         CI (tests on 3 OSes + package build), PyPI publish workflow, issue templates
```

Regenerate the chart after a new benchmark run:

```sh
python -m examples.benchmark_suite --output benchmarks/my-results.json
python docs/make_chart.py benchmarks/my-results.json docs/benchmark-speedup.svg
```

## What's missing for developers (v0.6)

Done in 0.6: array writes, `int64` arrays, recursion, exact int/float comparisons, type hints (`py.typed`), the [API reference](docs/API.md), the [contributor guide](CONTRIBUTING.md) with issue templates, an updated [notebook](examples/pyrust_vs_python.ipynb), GitHub Releases, benchmarks for macOS and Windows, and the [PyPI release](https://pypi.org/project/pyrust-engine/). These gaps remain:

| Missing | Why it matters to a developer |
|---|---|
| **Needs Rust installed on every machine** | PyRust compiles *your* functions, so `rustc` is needed wherever they should run natively. Without it, code still runs as plain Python. |
| **Lists, tuples, dicts and strings** | Code that builds lists or returns several values still runs as plain Python. |
| **2-D arrays and other dtypes** | Only 1-D `float64` / `int64` arrays are compiled. |
| **Call overhead** | Each native call costs about 2–4 µs, which matters for small inputs called very often. |

## Roadmap

What needs improving, most impactful first:

1. **Lower native call overhead (2–4 µs → well under 1 µs).** Replace the `ctypes` call with a small CPython extension entry point, and generate a specialised argument checker per function.
2. **Typed lists and tuples** (`list[float]` arguments, local tuples, multiple return values).
3. **2-D arrays** and `float32` / `int32` / `bool` element types.
4. **Parallel loops** (`prange`) that use every CPU core, since native code already releases the GIL.
5. **Adaptive dispatch.** Measure native vs Python time for the first calls and route small inputs (e.g. `n = 3`) to Python automatically.
6. **Benchmarks on ARM Linux**, and against Cython, PyPy and Codon.

See [`CHANGELOG.md`](CHANGELOG.md) for release history.

## License

[MIT](LICENSE) © 2026 Suraj Mhaske
