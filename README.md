# PyRust Engine

[![Tests](https://github.com/surajmhaske02/pyrust-engine/actions/workflows/test.yml/badge.svg)](https://github.com/surajmhaske02/pyrust-engine/actions/workflows/test.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.12%20%7C%203.13-blue)
![Status](https://img.shields.io/badge/status-experimental-orange)

**Add one decorator, and PyRust compiles a numeric Python function to native Rust. Results stay identical to Python: anything it can't run exactly falls back to Python.**

```python
from pyrust_engine import optimize

@optimize
def weighted_sum(n: int) -> int:
    total = 0
    for i in range(n):
        if i % 7 == 0:
            total += i // 3
        else:
            total += i % 11
    return total

weighted_sum(500_000)        # first call compiles (~0.2 s), then runs natively
weighted_sum(500_000)        # ~27x faster than plain Python on our benchmark
weighted_sum.report()        # native/fallback counts, timings, fallback reasons
```

> **Status: experimental prototype.** Not a general Python compiler and not production-hardened. Tested on Linux only so far; Windows and macOS run in CI.

## Comparison at a glance

| | Plain Python | **PyRust 0.3** | Numba | NumPy (vectorised) |
|---|---|---|---|---|
| Integer loop (500k) | 25.5 ms · 1× | **0.93 ms · 27.5×** | 0.59 ms · 43.4× | n/a |
| Float recurrence (300k) | 7.21 ms · 1× | **1.51 ms · 4.8×** | 1.75 ms · 4.1× | n/a |
| Array kernel (100k float64) | 5.91 ms · 1× | **0.144 ms · 40.9×** | 0.056 ms · 105× | 0.066 ms · 90× |
| Tiny `add(a, b)` per call | 0.034 µs · 1× | 2.08 µs · 0.016× | 0.11 µs · 0.30× | n/a |
| JSON → array pipeline | 5.93 ms · 1× | **4.40 ms · 1.35×** | not measured | n/a |
| Big integers / overflow | Exact | **Exact (replays in Python)** | Wraps at 64 bits | Wraps at 64 bits |
| Unsupported code | Runs | **Runs in Python, reason logged** | Compile error | n/a |
| Extra runtime dependencies | None | **None** (NumPy for arrays) | numba, llvmlite | numpy |
| First call (compile) | None | 0.15–0.18 s (0.05 s from disk cache) | 0.03–0.49 s | None |

<sub>Speed numbers are "time · speedup vs Python". Full method, raw data and caveats under [How it compares](#how-it-compares).</sub>

## Version comparison

| | 0.1.0 | 0.2.0 | **0.3.0 (current)** |
|---|---|---|---|
| Types | `int` | `int`, `float`, NumPy `float64` arrays | `int`, `float`, NumPy `float64` arrays |
| Arithmetic | `+ - *` | `+ - * // % /` | `+ - * // % /`, unary `+` |
| Loops | `range(stop)`, `range(start, stop)` | all `range` forms, `while`, `break`, `continue` | same |
| Branches | `if` / `else` | `if` / `else` | `if` / `elif` / `else` with early returns, `break`/`continue` in branches |
| Conditions | single comparison | single comparison, `and` / `or` / `not` | chained `a < b < c`, truthiness, float vs int literal |
| Final `return` required | Yes | Yes | No (missing return replays in Python) |
| Per-call overhead (`add(a, b)`) | 3.3 µs | 5.3 µs | **2.0 µs** |
| Cache | per-user disk cache | hashed, cross-process locked, self-repairing | same |
| Tools | `report()` | + `prepare`, `reset`, metrics, profiler CLI, cache pruning | same |
| Automated tests | 12 | 34 | **47** |
| Random testing | 100 rounds of fixed kernels | 12,000+ comparisons on fixed templates | **+ 32,000 generated-program comparisons, 0 mismatches** |

---

## Why it exists

Most "make Python fast" tools ask you to give something up: a new interpreter, a build step, a different integer model, or a heavy dependency. PyRust takes a narrower bet:

- **Correctness first.** Integers stay arbitrary-precision. If a 64-bit operation would overflow, divide by zero, index out of bounds or hit an invalid `range` step, the native call stops and **the original Python function is replayed**, so you get exactly Python's result or Python's exception. Nothing wraps silently.
- **Opt-in, per function.** Decorate the hot loop, leave everything else alone. Unsupported code, inputs or environments run as normal Python and the reason is recorded.
- **Light footprint.** Pure-Python package with no runtime dependencies (NumPy only for array kernels). Uses `rustc` directly through `ctypes`: no PyO3, Maturin, crates or C compiler toolchain beyond Rust's own linker.

## How it compares

### Speed

![Speedup versus plain Python](docs/benchmark-speedup.svg)

| Workload | Python | **PyRust** | Numba | NumPy (vectorised) | PyRust vs Python |
|---|---:|---:|---:|---:|---:|
| Integer loop (`//`, `%`, branches; n = 500k) | 25.5 ms | **0.93 ms** | 0.59 ms | — | **27.5× faster** |
| Float recurrence (n = 300k) | 7.21 ms | **1.51 ms** | 1.75 ms | — | **4.8× faster** |
| Array kernel, sum of squares (100k float64) | 5.91 ms | **0.144 ms** | 0.056 ms | 0.066 ms | **40.9× faster** |
| Tiny function `add(a, b)`, per call | 0.034 µs | 2.08 µs | 0.11 µs | — | **61× slower** |
| JSON → NumPy → kernel → JSON pipeline | 5.93 ms | **4.40 ms** | — | — | **1.35× faster** |

<sub>Median of 7 warm runs, wrapper and argument conversion included. Linux x86_64, CPython 3.13.18, Rust 1.97.0, NumPy 2.5.4, Numba 0.68.1 (`fastmath=False`). Raw samples: [`benchmarks/results-v0.3.0.json`](benchmarks/results-v0.3.0.json). Reproduce with `python -m examples.benchmark_suite`.</sub>

**How to read this, honestly:**

- PyRust gives **large speedups on loop-heavy numeric code** and beat Numba on the float recurrence in this run.
- **Numba is faster** on the integer and array kernels, and **vectorised NumPy** is the better tool when the problem is a whole-array expression.
- **Tiny functions get slower.** Each native call costs about 2 µs of dispatch. Decorate functions that do real work per call.
- **End-to-end gains shrink** when parsing and conversion dominate: the JSON pipeline only improved 1.35×.

### Startup cost

| Workload | PyRust: first call incl. Rust compile | PyRust: first call from disk cache | Numba: first call incl. JIT |
|---|---:|---:|---:|
| Integer loop | 0.175 s | 0.047 s | 0.488 s |
| Float recurrence | 0.183 s | 0.062 s | 0.044 s |
| Array kernel | 0.179 s | 0.049 s | 0.095 s |

PyRust caches compiled libraries on disk across processes. Use `fn.prepare()` (or `prepare(background=True)`) before latency-sensitive work.

### Design trade-offs

Speed was only measured against Numba and NumPy. The rows for Cython, PyPy and Codon describe how each tool is designed, not benchmark results.

| | **PyRust** | Numba | Cython | PyPy | Codon |
|---|---|---|---|---|---|
| How you use it | Decorator on a function | Decorator on a function | Separate build step (`.pyx` or annotated `.py`) | Different Python interpreter | Separate compiler |
| Integer overflow | **Exact Python result** (replays in Python) | Wraps at 64 bits | Python ints exact; C-typed ints follow C rules (optional overflow checks) | Exact | Wraps at 64 bits |
| Unsupported code | Runs as normal Python, reason recorded | Compile error | Compiles via the C API (slower paths) | Runs everything | Compile error |
| Scope | Small numeric subset | Large NumPy-oriented subset | Full Python | Full Python | Large Python-like subset |
| Runtime dependencies | None (NumPy for arrays) | numba + llvmlite | None after build | Whole runtime | Codon runtime |
| Toolchain needed | `rustc` | none extra | C compiler | none extra | Codon |
| Maturity | Experimental | Mature | Mature | Mature | Active project |

**Pick PyRust** when you want Python-exact integer semantics, a silent safety net and no compiled dependency stack, and your hot spots are plain numeric loops. **Pick Numba or NumPy** for maximum speed on array maths. **Pick Cython** when you need to compile large or object-heavy code. **Pick PyPy** to speed up a whole pure-Python application.

## Correctness evidence

- **47 automated tests** covering semantics, overflow replay, exceptions, NaN, ±0.0, infinities, array layouts, bounds, cache corruption and recovery, cross-process locking, background compilation, threads and profiling.
- **Generated-program fuzzing:** 4,000 random programs (2,000 integer + 2,000 float, using `while`, `break`, `continue`, `elif`, `//`, `%`, `/`), **32,000 calls compared with CPython, 0 mismatches**. Floats are compared by `repr`, so NaN and signed zero must match exactly. Every generated program must compile natively, so the test can't pass by quietly falling back to Python.
- **Fuzzer sensitivity:** deliberately planted bugs (wrong modulo sign; `a / b` rewritten as `a * (1/b)`) were caught immediately.

Details and limitations: [`VALIDATION.md`](VALIDATION.md).

## Install

Requires **Python 3.10+** and **Rust** (`rustc` on `PATH`, from <https://rustup.rs>). Not yet on PyPI.

```sh
git clone https://github.com/surajmhaske02/pyrust-engine.git
cd pyrust-engine
python -m pip install ".[arrays]"          # NumPy for array kernels
python -m unittest discover -s tests -v    # 47 tests
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

### NumPy arrays without copying

```python
import numpy as np
from pyrust_engine import optimize, F64Array

@optimize
def energy(values: F64Array) -> float:
    total = 0.0
    for i in range(len(values)):
        x = float(values[i])
        total += x * x
    return total

energy(np.arange(100_000, dtype=np.float64))
```

Accepted arrays: exact `ndarray`, 1-D, C-contiguous, aligned, native-endian `float64`. Others fall back without copying. Kernels only read arrays. Use `float(values[i])` for element reads; negative indices work and out-of-bounds reads raise Python's `IndexError` via replay.

### Supported syntax

| Construct | Native support |
|---|---|
| Parameters | `int` (default), `float`, `F64Array` |
| Integer ops | `+ - *`, unary `- +`, `//` and `%` with Python floor semantics; overflow-checked |
| Float ops | `+ - * /`, unary `- +`; mixed int/float promotes like Python; no fast-math |
| Loops | `for` over `range(stop)`, `range(start, stop)`, `range(start, stop, step)` (negative steps too); `while`; `break`; `continue` |
| Branches | `if` / `elif` / `else`, early `return` |
| Conditions | Comparisons incl. chained `a < b < c`; float vs int literal (`x > 0`); `and`, `or`, `not`; truthiness `if n:` |
| Locals | Assignment, annotated assignment, `+= -= *= //= %= /=` |
| Calls | `len(array)`, `float(x)`, `float(array[i])` |

These run as normal Python instead: lists, dicts, strings, other calls, `**`, recursion, closures, generators, classes, array writes. Variables first assigned inside a loop aren't treated as assigned after it, which is conservative by design.

### Control and diagnostics

```python
fn.prepare()                    # compile now
fn.prepare(background=True)     # compile in a thread; calls use Python meanwhile
fn.reset()                      # clear compile state and retry
fn.report()                     # calls, native/fallback counts, timings, fallback_reasons
fn.rust_source()                # inspect the generated Rust
fn.python(...)                  # call the original function

@optimize(fallback=False)            # raise OptimizationError instead of falling back (good for benchmarks)
@optimize(max_iterations=10**9)      # native loop budget (default 10,000,000)
```

> **Iteration budget:** a call that exceeds `max_iterations` discards its native work and **replays the whole call in Python**, losing the speedup. With 12M iterations: Python 0.450 s, default budget 0.384 s, `max_iterations=10**9` 0.015 s. Raise it for known large loops; watch `report()['fallback_reasons']`.

### Find candidates in existing code

```sh
python -m pyrust_engine profile path/to/script.py arg1 arg2   # runs the script once (trusted code only)
python -m pyrust_engine cache --days 30                       # list old cache entries (add --delete)
```

The profiler ranks functions in the entry script by time and reports which ones fit the native subset. It never rewrites your code.

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
pyrust_engine/   compiler.py (AST → Rust), runtime.py (decorator, dispatch, metrics),
                 cache.py (disk cache, locks), profiling.py, __main__.py (CLI)
tests/           test_engine.py, test_v2.py, test_v3.py, test_fuzz.py (generated programs)
examples/        benchmark_suite.py (Python / PyRust / Numba / NumPy), benchmark.py
benchmarks/      raw benchmark results (0.2.0 and 0.3.0 runs)
docs/            benchmark chart and the script that generates it
```

Regenerate the chart after a new benchmark run:

```sh
python -m examples.benchmark_suite --output benchmarks/my-results.json
python docs/make_chart.py benchmarks/my-results.json docs/benchmark-speedup.svg
```

## Roadmap

- Results from Windows/macOS CI and ARM hardware
- Lower call overhead (a CPython extension instead of `ctypes`)
- Array outputs and writes; more array element types
- Calls between optimized functions
- Benchmarks against Cython, PyPy and Codon
- Generated-program fuzzing for array kernels

See [`CHANGELOG.md`](CHANGELOG.md) for release history.

## License

No license has been chosen yet. Until one is added, all rights are reserved by the author. Add a `LICENSE` file before inviting outside use or contributions.
