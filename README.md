# PyRust Engine 0.3

A typed Python-to-Rust middleware prototype. Decorate supported computational functions with `@optimize`; the engine compiles them into native Rust libraries and keeps Python fallback for unsupported inputs, syntax, and native guard failures.

**This is not a general Python compiler or a production-ready replacement for Python.** Profiling finds candidates; it does not rewrite entire applications. Native acceleration is useful for substantial computation. Small functions can become much slower because of dispatch overhead.

## Install and test

Requires Python 3.10+ and `rustc` with Rust 2021 support on PATH. Install Rust from https://rustup.rs. Windows also needs the appropriate linker/build tools. Scalar kernels need no third-party Python runtime package. Array support requires NumPy.

```sh
python -m pip install ".[arrays]"
python -m unittest discover -s tests -v
python -m examples.benchmark_suite --output benchmarks/my-results.json
```

For optional Numba comparisons:

```sh
python -m pip install ".[benchmark]"
python -m examples.benchmark_suite --output benchmarks/my-results.json
```

These are **local installation commands**. The package has not been published to PyPI.

## Integer and float kernels

Save functions in a `.py` file so their source can be inspected.

```python
from pyrust_engine import optimize

@optimize
# Unannotated parameters default to the integer specialization.
def calculate(n: int) -> int:
    total = 0
    for i in range(n):
        total += i * i
    return total

print(calculate(1_000_000))
print(calculate.report())

@optimize
def recurrence(n: int, value: float) -> float:
    for i in range(n):
        value = (value * 1.000001 + 0.0001) / 1.000002
    return value

print(recurrence(100_000, 1.0))
```

Annotations select native parameter types. `int` accepts exact built-in Python integers fitting signed 64 bits; `float` accepts exact built-in Python floats. Passing an int to a float parameter falls back rather than silently coercing it. Unsupported annotations also fall back. Boolean values and integer subclasses do not enter integer kernels.

All paths must return the same inferred scalar type. Return annotations do not coerce results. Local variables cannot change type: initialize a float accumulator as `0.0`, not `0`.

## Array kernels with no input copy

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

values = np.arange(100_000, dtype=np.float64)
print(energy(values))
```

Native array inputs must be exact NumPy `ndarray` objects, one-dimensional, aligned, C-contiguous, and native-endian `float64`. Both writable and read-only arrays are accepted, but kernels only **read** them. Strided arrays, different dtypes and ndarray subclasses fall back without conversion or copying. Array outputs and array writes are not implemented.

Use **`float(values[i])`** for element reads. This explicitly gives Python float arithmetic on the Python path and avoids silently changing NumPy scalar warnings/error settings. Plain `values[i]` expressions are outside the subset. Negative indices work; bounds errors replay in Python to preserve its exception.

Keep inputs alive and do not mutate, resize or free their buffers concurrently with a native call. The wrapper retains argument references, but cannot prevent another thread from mutating an array. ctypes releases the GIL.

## Supported syntax

| Construct | Native support |
|---|---|
| Scalar parameters | `int`, `float`; unannotated defaults to `int` |
| Arrays | `F64Array`, `len(a)`, `float(a[i])`; read-only operations |
| Integer operations | `+`, `-`, `*`, unary `-`/`+`, `//`, `%`, checked overflow |
| Float operations | `+`, `-`, `*`, unary `-`/`+`, `/`; no fast-math flags |
| Mixed int/float arithmetic | `+`, `-`, `*`, `/` when at least one operand is float |
| Conversion | Explicit built-in `float(scalar)` |
| Conditions | Comparisons of same-type scalars, including chained `a < b < c`; float vs integer literal with \|literal\| ≤ 2**53 (`if x > 0:`); `and`, `or`, `not`; truthiness `if n:` / `if x:` |
| Loops | `range(stop)`, `range(start, stop[, step])`, including negative steps; `while` |
| Control flow | Nested loops, `if`/`elif`/`else`, `break`, `continue`, `pass` |
| Local variables | Ordinary/annotated assignment and supported augmented operations |
| Return | Scalar returns of one type; early returns allowed; a path that falls off the end replays in Python (returning `None`, as the original does) |

Not supported: integer `/`, float `//` or `%`, powers, comparisons between int and float variables, general calls, lists, dictionaries, strings, classes, array writes, array outputs, recursion, closures, generators, async, loop `else`, exceptions in native source, or arbitrary dynamic Python. These use Python fallback.

Definite-assignment analysis is deliberately conservative. It understands branches that always `return`, `break` or `continue` (so `elif` chains with early returns compile), but variables first assigned inside a loop body are not considered assigned after the loop, and such functions fall back.

Integer `//` and `%` preserve Python's negative-number semantics. Intermediate integer overflow falls back even if a later operation would bring the result back into range. Floats use normal f64 operations without reassociation/fast-math; tests cover NaN, infinities and signed zero. Exact NaN payloads and bitwise reproducibility across hardware are not guaranteed.

## Prepare, background compilation and retry

```python
calculate.prepare()                  # Blocking precompilation, before serving requests
calculate.prepare(background=True)   # Starts preparation in a daemon thread
calculate.reset()                    # Clear wrapper compilation state; retain metrics/cache
calculate.prepare()                  # Retry after fixing compiler/cache environment
```

During background preparation, normal calls execute Python instead of waiting. Background work ends if the process exits. `reset()` waits for active background preparation before clearing state; concurrent calls already holding a native library can finish safely. Use a new wrapper if function code or defaults were changed. Mutation of code, builtins, defaults or input buffers concurrently with execution is outside the supported contract.

Default fallback preserves the original Python return value/exception. `fallback=False` raises `OptimizationError` on rejection or native guard failure and is useful for benchmarks. It also raises while background compilation is pending.

Every native loop iteration consumes a configurable shared budget (`max_iterations=10_000_000` by default). Exceeding it falls back to Python; this limits time spent in an uninterruptible native loop, but is not a hard wall-clock timeout. It does not make an infinite Python function terminate.

**Budget exhaustion replays the whole call in Python, so it loses the speedup.** Measured with a 12,000,000-iteration loop: plain Python 0.450 s, default budget 0.384 s (native work discarded, then Python replay), `max_iterations=10**9` 0.015 s. Raise `max_iterations` for functions with known large loops, and watch `report()['fallback_reasons']` for `native iteration budget exceeded`.

## Profile existing code

```sh
python -m pyrust_engine profile path/to/your_script.py arg1 arg2
```

This **executes the trusted script once**, ranks functions in that entry script by self time, and reports which retained top-level functions match the compiler subset. Script output goes to stderr; the profile is JSON on stdout. Use a representative workload. Decorate suitable functions after reviewing the report. Imported-module functions, closures, and whole-application automatic rewriting/selection are not implemented. Compatibility is static; actual runtime inputs still undergo guards.

## Metrics and cache management

`fn.report()` includes:

- `calls`, `native_calls` (native attempts, including guard failures), `fallback_calls`, and `fallback_rate`.
- Reasons counted separately, plus the latest reason and a persistent compilation error field.
- Compilation/preparation time, native execution time, Python fallback time, and whole-wrapper time.
- Average native/fallback timings, cache hit/rebuild status and compilation attempts.

Native timing excludes argument conversion; whole-wrapper timing includes it and compilation. Times can overlap across threads and are not application wall time. Failed argument binding still counts as a wrapper call. Counter updates are synchronized; reports are diagnostic snapshots, not a transactional audit log.

Cache entries use compiler/platform/source keys, SHA-256 content checks, OS cross-process locks and atomic publication. Invalid metadata/binary hashes trigger a rebuild. Missing symbols and load failures attempt one rebuild for cached entries, then fall back. The cache must be private; on POSIX it must be owned by the current user and not group/world writable. Hashes detect corruption, not a malicious cache owner.

```sh
python -m pyrust_engine cache --days 30             # Dry-run list
python -m pyrust_engine cache --days 30 --delete    # Remove old entries
```

Or call `prune_cache(path, older_than_days=30, dry_run=True)`. Age is based on metadata creation/rebuild time, not last access. Lock files are retained to avoid cross-process races. Windows may retain in-use DLLs; deletion errors are reported. A cache is a performance optimization, not required data.

## Call overhead

Each call pays for argument checks, metrics and a `ctypes` dispatch. Plain positional calls take a fast path that skips `inspect.Signature.bind`; keyword calls and calls that rely on defaults take the slower binding path. Measured on Linux with `add(a, b)`: plain Python ~40 ns, positional ~2.0 µs, keyword ~4.3 µs (it was ~5.3 µs for every call in 0.2.0). Decorate functions that do substantial work per call.

## Evaluation

See **VALIDATION.md**, **benchmarks/results.json** (0.2.0 run) and **benchmarks/results-v0.3.0.json** (0.3.0 run) for measured results, full timing samples and limitations. The updated suite compares Python, PyRust, Numba (if installed), vectorized NumPy for the array task, and a complete JSON/array pipeline. Cold compilation and cache startup are reported separately. The original single benchmark is retained in `examples/benchmark.py`.

Do not use the largest speedup as an application-wide claim. Numba/NumPy can beat this prototype. Tiny functions can be dramatically slower under the wrapper. Benchmark your workload before adopting it.

CI is configured for Linux/macOS/Windows and Python 3.10/3.12/3.13, but only locally executed environments listed in VALIDATION.md are verified. Remote CI has not run.

## Architecture and trust boundary

Python source → typed AST validation → Rust `Result<T, i32>` code → rustc cdylib → ctypes dispatch. Native status codes trigger Python replay for integer overflow, zero division/range step, bounds failure or loop-budget exhaustion. Supported native code has no external writes, so replay does not duplicate external side effects.

No Rust crates or PyO3/Maturin are needed. Compiled libraries run in-process. Run only trusted Python; this is not a sandbox for user uploads. Debuggers/tracers, monkey-patching races, native memory faults and externally corrupted libraries are not covered by Python fallback. No formal equivalence proof or production security review has been performed.

## Push to GitHub

This folder is already a Git repository with history (0.1.0 → 0.2.0 → 0.3.0; the earlier standalone review is on branch `review-fixes-v0.2.0`). From the extracted folder, with Git and GitHub CLI installed:

```sh
gh auth login
gh repo create pyrust-engine --private --source=. --remote=origin --push
git push origin --all
```

For an existing empty remote, use `git remote add origin YOUR_REPOSITORY_URL` and `git push -u origin main`. Select a license before publishing publicly; none has been chosen on your behalf.

## Remaining work

Real Windows/macOS CI execution; production load testing; richer type inference; safe array output kernels; imported-module hotspot analysis; profiler-guided automatic dispatch decisions; benchmarks against Cython, PyPy and Codon; generated-program fuzzing for array kernels; lower-overhead dispatch (CPython extension instead of ctypes). These are not implemented or claimed as completed.

## Testing

```sh
python -m unittest discover -s tests -v
PYRUST_FUZZ_PROGRAMS=500 PYRUST_FUZZ_SEED=7 python -m unittest tests.test_fuzz   # larger generated-program run
```

`tests/test_fuzz.py` generates random integer and float programs in the supported subset (including `while`, `break`, `continue`, `elif`, `//`, `%`, `/`) and requires identical results or exception types natively and in CPython. Every generated program must compile natively, so the fuzz test cannot pass by silently falling back.
