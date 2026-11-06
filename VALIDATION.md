# Validation

## 0.4.0 to 0.6.0: CI on three operating systems, 6 November 2026

From 0.4.0 on, every push runs the full test suite in GitHub Actions on **Linux, Windows and macOS**, each with **Python 3.10, 3.12 and 3.13** (9 jobs), with Rust stable. Each job also runs a larger generated-program fuzz pass (`PYRUST_FUZZ_PROGRAMS=150`) and the benchmark suite, and uploads the benchmark results as an artifact.

| Release | Tests | Result | What was added to the evidence |
|---|---:|---|---|
| 0.4.0 | 56 | 9/9 jobs passed | Loop-free functions run as plain Python; fixed two test-only failures on Windows (loaded DLLs can't be deleted or replaced) and macOS (Rust builds aren't byte-identical) |
| 0.5.0 | 89 | 9/9 jobs passed | Edge-case tests for every new operator and `math` function; extended fuzzer with `**`, bit operations, `abs`/`min`/`max`, ternaries, float `//` `%` `**`, `int()`, `math.floor/ceil` and nine `math` functions, compared with CPython on all three operating systems |
| 0.6.0 | see CI | see CI | Recursion (depth and budget limits), array writes with copy-on-write (failure mid-call, overlapping arrays, read-only arrays), `I64Array`, exact int/float comparisons at 2**53 and 2**63 boundaries, package build check |

Notes:

- `math` functions are compared bit-for-bit with CPython on each operating system. Both call the platform's C math library, which is why results match; on a platform where they didn't, the fuzzer would fail.
- Benchmark numbers in the README come from the `ubuntu-latest` CI job. Shared CI machines are noisy, so treat them as approximate.

## 0.3.0 — merge of the 0.2.0 release and the independent review, 6 November 2026

Executed on Linux x86_64, CPython 3.13.18, Rust 1.97.0, NumPy 2.5.4, Numba 0.68.1. Windows and macOS were **not** run; the CI matrix covers them once pushed.

### Re-check of 0.2.0 as delivered

- All **34** original 0.2.0 tests passed unchanged on this newer Python/Rust combination.
- The 0.2.0 benchmark results reproduced in shape (table below).

### Tests after the merge

- `python -m unittest discover -s tests -v`: **47 tests passed**, 0 failures, 0 skips (34 original + 11 in `test_v3.py` + 2 generated-program fuzz tests).
- With `rustc` removed from `PATH`: 14 passed, 33 native-only tests skipped (fallback path).

### Generated-program differential fuzzing (new)

0.2.0's randomized tests used fixed function templates. `tests/test_fuzz.py` instead generates whole random programs in the supported subset (assignments, annotated assignments, `+ - * // % /`, unary minus, `for` over all `range` forms, `while`, `break`, `continue`, `if/elif/else`, `and/or/not`, chained comparisons, truthiness, early returns) and requires identical results (floats compared by `repr`, so NaN and -0.0 are exact) or identical exception types. Every program must compile natively.

| Run | Integer programs | Float programs | Calls compared | Mismatches |
|---|---:|---:|---:|---:|
| Seeds 1–8, 250 programs each per kind | 2,000 | 2,000 | 32,000 | **0** |

Sensitivity check: planting a wrong modulo sign rule made the integer fuzz fail immediately; replacing `a / b` with `a * (1.0 / b)` made the float fuzz fail immediately. Restoring the code made both pass.

### Findings fixed in 0.3.0

1. **Per-call overhead** — every call ran `inspect.Signature.bind`, took four locks and read the clock four times: 5.3 µs per `add(a, b)` call (plain Python 40 ns). Now 2.0 µs positional, 4.3 µs keyword.
2. **`elif` chains with early returns, and branches ending in `break`/`continue`, fell back to Python** because definite assignment intersected all branches. Now exit-aware.
3. **A final `return` was required** even when every path returned (e.g. `if/else` both returning). Now any path that falls off the end replays in Python and returns `None`.
4. **Common conditions fell back**: chained comparisons, `if n:`, and `if x > 0:` with a float `x`. Now native (integer literals only up to 2**53, where conversion is exact; `2**53 + 1` is still rejected and tested).
5. **Some unexpected exceptions escaped** from compilation instead of falling back (only a fixed list of exception types was caught). Now any exception falls back and is reported.

### Behaviour documented, not changed

- **Iteration budget**: when a call exceeds `max_iterations` (default 10,000,000), the native work is discarded and the whole call is replayed in Python. A 12,000,000-iteration loop took 0.450 s in Python, 0.384 s with the default budget and 0.015 s with `max_iterations=10**9`.

### Benchmarks (`python -m examples.benchmark_suite`, 7 warm samples, median)

| Workload | Python ms | PyRust ms | Python / PyRust | Numba ms | First call incl. compile | Disk-cache first call |
|---|---:|---:|---:|---:|---:|---:|
| integer_loop | 25.52 | 0.929 | 27.5× | 0.588 | 0.175 s | 0.047 s |
| float_recurrence | 7.21 | 1.510 | 4.8× | 1.751 | 0.183 s | 0.062 s |
| array_energy | 5.91 | 0.144 | 40.9× | 0.056 | 0.179 s | 0.049 s |
| tiny_scalar (per call) | 0.000034 | 0.002078 | 0.016× (slower) | 0.000113 | 0.150 s | 0.048 s |
| json_array_pipeline | 5.93 | 4.40 | 1.35× | — | — | — |

Full samples: `benchmarks/results-v0.3.0.json`. One machine, controlled workloads; not production load testing. Numba was faster on the integer and array kernels; PyRust was slightly faster on the float recurrence in this run.

### Still untested

Windows and macOS, ARM, production workloads, memory pressure, generated-program fuzzing of array kernels, and comparisons with Cython, PyPy and Codon.

---

## 0.2.0 validation (original author) — 6 November 2026

## Executed checks

- **34 automated tests passed, zero failures and zero skips** on Linux x86_64 with CPython 3.12.14 and Rust 1.75.0.
- Over **12,000 deterministic randomized differential comparisons** across signed integer arithmetic, stepped/nested/while loops, floating-point arithmetic and array kernels, plus boundary/error cases. These use fixed function templates, not arbitrary generated programs.
- Verified signed division/modulo behavior, 64-bit boundaries, overflow replay, zero-divisor exceptions, zero-step exceptions, negative array indices, bounds errors, iteration-budget fallback, NaN/infinity/signed-zero cases and unsupported input layouts/types.
- Verified explicit prepare/reset, compiler missing/timeout/failure fallback, missing-symbol recovery, corrupt metadata and corrupt binary rebuilds, private cache cleanup and background compilation fallback.
- Verified 400 calls across 8 threads with synchronized counters and 4 concurrent processes sharing one cache; only one process compiled a fresh entry.
- Verified profile API and CLI on a trusted script; profiler identifies supported top-level functions without modifying source.
- Built and installed the v0.2 wheel locally.

The machine used for testing is the execution environment, not the user's computer. Windows/macOS CI is configured but has **not run**. No production load testing, formal proof or security audit has been performed.

## Benchmark method

Seven warm timing samples per workload, medians reported. Each tiny-scalar sample batches 1,000 calls. Whole Python/native wrapper call times are measured, including marshalling and diagnostics overhead. Compilation and disk-cache startup are separately recorded in `benchmarks/results.json`. Equality checks run before timing; floating-point comparisons to Numba/NumPy use tolerance. Input arrays/data are prepared before kernel timings; the pipeline includes JSON parsing, conversion and serialization each invocation.

Environment: Python 3.12.14, Linux-6.18.57-x86_64-with-glibc2.39, rustc 1.75.0 (82e1608df 2023-12-21) (built from a source tarball), NumPy 2.5.4, Numba 0.68.1.

| Workload | Python ms | PyRust ms | Python / PyRust | Numba ms | NumPy ms |
|---|---:|---:|---:|---:|---:|
| integer_loop | 39.618481 | 1.719815 | 23.036× | 0.996074 | — |
| float_recurrence | 14.443996 | 2.512869 | 5.748× | 2.712011 | — |
| array_energy | 11.806823 | 0.285934 | 41.292× | 0.079676 | 0.065509 |
| tiny_scalar | 0.000036 | 0.007492 | 0.005× | 0.000167 | — |
| json_array_pipeline | 8.285511 | 6.219206 | 1.332× | — | — |

A ratio above 1 means PyRust was faster than Python; below 1 means slower. Results are one run on one machine and are not statistically established superiority claims.

- Integer/float/array loops benefited relative to their Python-loop baselines.
- Numba was faster for integer and array kernels; the float recurrence was close in this run. Vectorized NumPy was faster than this array kernel.
- The tiny addition was approximately 208× slower under PyRust: do not decorate trivial functions.
- The complete JSON/array pipeline improved only about 1.33× because parsing/conversion remained in Python.
- Numba comparison inputs stayed within int64 range. Numba does not supply this middleware's arbitrary-precision overflow replay semantics.
- NumPy uses a different reduction order, so float equivalence uses tolerance; its expression also allocates an intermediate array.
- Each benchmark result is specific to its workload and does not imply every application benefits.

## Cold startup

| Workload | Fresh compilation + call (s) | Disk-cache first call (s) |
|---|---:|---:|
| integer_loop | 0.7938 | 0.0939 |
| float_recurrence | 0.8684 | 0.0538 |
| array_energy | 0.5003 | 0.0448 |
| tiny_scalar | 0.4262 | 0.0456 |

Disk-cache first calls still invoke rustc for version identification, validate the binary hash and load the library. Precompile before latency-sensitive requests. Full samples and Numba cold-call timings are in the JSON report.

## Remaining evaluation gaps

Cython, PyPy and Codon comparisons; production traces; memory-pressure/cancellation testing; Windows/macOS execution; ARM hardware; generated-program fuzzing. Those have not been completed.

No GitHub repository was created or pushed; publication requires the user's GitHub access.
