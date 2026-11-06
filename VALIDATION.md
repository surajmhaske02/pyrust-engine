# Validation

## v0.2.0 — independent review, 6 November 2026

Executed on Linux x86_64, CPython 3.13.18, Rust 1.97.0 (stable). Windows and macOS were **not** run locally; the GitHub Actions matrix covers them once the repository is pushed.

### Re-check of the v0.1.0 claims

- Original 12 tests: **all passed** on this newer Python/Rust combination.
- Original benchmark: warm speedup reproduced (72.9–74.9× here vs 26.8× originally; the kernel is vectorisable, so the figure is sensitive to compiler version and CPU).

### Differential fuzzing (new)

Random programs in the supported subset are generated and each is run natively and in CPython with edge-case inputs (0, ±1, 2**40, ±2**62, i64 min/max). Results, or exception types, must match.

| Run | Programs | Calls compared | Compiled natively | Mismatches |
|---|---:|---:|---:|---:|
| v0.1.0 grammar (`+ - *`, `range` 1–2 args, `if/else`) | 500 | ~3,000 | 500 | **0** |
| v0.2.0 grammar (adds `// %`, `range` step, `elif`, `and/or/not`, chained compare, annotations), seeds 1–8 | 2,000 | 16,000 | 2,000 | **0** |

Fuzzer sensitivity check: planting a wrong floor-division rounding rule in the generated Rust made the 40-program default fuzz test fail immediately; restoring it made it pass.

### Findings from the review (fixed in v0.2.0)

1. **Per-call overhead** — every call went through `inspect.Signature.bind`: 3.3 µs per call vs 58 ns for plain Python on `return a + b`. A positional fast path brings this to ~1.05 µs; the remainder is `ctypes` dispatch (documented).
2. **`elif` chains with early returns fell back to Python** — definite-assignment analysis intersected the variables of every branch, including branches that always return. Now return-aware.
3. **Annotated locals (`total: int = 0`) fell back to Python.** Now supported.
4. **Unexpected internal exceptions** (anything other than the listed types) escaped to the caller instead of falling back. Now caught and reported as `Internal compiler error`.
5. **Thread-start race** — the "already compiled" flag was set before compilation finished; it is now published last with double-checked locking. Covered by an 8-thread first-call test.

### New native features (all fuzzed)

`//` and `%` with Python floor semantics; `range(start, stop, step)` including negative steps and overflow at the end of the range; `and`/`or`/`not`, chained comparisons and integer truthiness in conditions; `elif`; `pass`; unary `+`.

### Test suite

`python -m unittest discover -s tests -v`: **21 tests passed** with Rust available (incl. 40-program differential fuzz). With `rustc` removed from `PATH`: 11 passed, 10 native-only tests skipped.

### Benchmarks (`python -m examples.benchmark`, cold cache, median of 5 warm runs)

| Kernel | n | Python | Native (warm) | Speedup | First call incl. compile |
|---|---:|---:|---:|---:|---:|
| `branch_sum` (vectorisable) | 1,000,000 | 34.4 ms | 0.47 ms | 72.9× | 141 ms |
| `lcg_mix` (serial dependency, `%` and `//`) | 1,000,000 | 160.8 ms | 4.29 ms | 37.5× | 189 ms |
| `tiny(a, b) = a + b` per call | — | 29 ns | 1,052 ns | **0.03× (slower)** | — |

These are micro-benchmarks on one machine. They are not a promise for other functions or whole applications.

### Still untested

Windows and macOS (CI configured, not run locally), real application workloads, long-running or concurrent production use, and comparisons against Numba, Cython, PyPy and Codon.

## v0.1.0 — original author's validation (summary)

Linux x86_64, CPython 3.12.14, Rust 1.75.0: 12 tests passed; `branch_sum` 1,000,000 iterations: Python 67.07 ms, native 2.50 ms (26.8×), first call 877 ms including compilation.
