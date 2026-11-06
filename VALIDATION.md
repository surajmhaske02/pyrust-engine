# Validation — 6 November 2026

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
