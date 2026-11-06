# 0.4.0

Fixes the tiny-function slowdown and gets CI green on all three platforms.

- Loop-free functions (no `for`/`while`) now run as plain Python by default. Native
  dispatch through `ctypes` costs about 2 µs per call, more than such functions do, so
  `add(a, b)` was 61× slower than Python in 0.3.0. `@optimize` now returns a clone of
  the original function with `report()`, `prepare()`, `reset()` and `rust_source()`
  attached, so calls cost the same as plain Python. `report()` shows
  `dispatch: 'python-direct'` and the reason. `native_loop_free=True` restores native
  compilation.
- `report()` gains `dispatch` and `call_counting` keys.
- Windows CI: test temporary directories now ignore cleanup errors, because a DLL loaded
  by the test process can't be deleted. The corrupt-binary test is skipped on Windows,
  because a loaded DLL can't be replaced either.
- macOS CI: `rustc` output isn't byte-identical between builds there, so the
  corrupt-binary test now checks the library named by the cache metadata.
- Benchmark suite adds a forced-native tiny-function row; chart footer reads the
  environment from the results file.
- 9 new tests (56 total). README adds a tiny-function explanation, a limitations
  section and a prioritised roadmap.

# 0.3.0

Merges the independent review of 0.1.0 into the 0.2.0 feature release.

- Faster call path: exact positional calls skip `inspect.Signature.bind`; the compiled
  library is published as one immutable snapshot read without a lock; one metrics lock
  per call instead of four. `add(a, b)`: 5.3 µs → 2.0 µs per call (keyword calls 4.3 µs).
- Definite-assignment analysis understands branches that always `return`, `break` or
  `continue`: `elif` chains with early returns and similar loops now compile natively.
- A final `return` is no longer required: a path that falls off the end replays in
  Python, which returns `None` exactly as the original function.
- Chained comparisons (`a < b < c`), integer/float truthiness (`if n:`), float vs integer
  literal comparisons with |literal| ≤ 2**53 (`if x > 0:`), unary `+`.
- Any unexpected exception during compilation or loading now falls back to Python
  instead of escaping to the caller.
- New generated-program differential fuzzer (`tests/test_fuzz.py`) for integer and
  float programs, including `while`/`break`/`continue`; floats compared by `repr`.
- 13 new tests (47 total). README documents call overhead and iteration-budget cost.
- CI: installs a Rust toolchain explicitly and runs a larger fuzz pass.
- README rewritten around comparison tables (vs Python, Numba, NumPy, Cython, PyPy,
  Codon, and across versions) with a generated benchmark chart (`docs/`).

# 0.2.0

- Typed float64 scalar and zero-copy read-only NumPy array kernels.
- Python-compatible signed integer floor division/modulo.
- Stepped/negative ranges, while loops, break/continue and boolean conditions.
- Native iteration budget and checked array bounds.
- Explicit precompile, background preparation and reset/retry.
- Cross-process cache locks, content hashes, atomic publication and recovery.
- Dry-run-first cache pruning.
- Synchronized counters, timing metrics and fallback reason counts.
- Entry-script hotspot profiler and CLI.
- Expanded randomized, exception, cache, concurrency and array-layout tests.
- Multi-workload cold/warm benchmarks including Numba/NumPy and a JSON pipeline.
- Platform CI matrix configured; remote jobs have not been executed.

# 0.1.0

Initial integer-only decorator and Rust compilation prototype.
