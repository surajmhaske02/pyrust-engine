# Benchmarks

Speedup over plain Python, measured by CI on GitHub Actions runners with Python 3.13 and Rust 1.99. Raw samples are in the [`benchmarks/`](https://github.com/surajmhaske02/pyrust-engine/tree/main/benchmarks) folder; every CI run also compares against the last release and prints the table in the job summary.

| Workload | Linux x86_64 | macOS arm64 | Windows x86_64 |
|---|---:|---:|---:|
| Integer loop | 31x | 39x | 29x |
| Float recurrence | 14x | 5.7x | 7.8x |
| Array kernel (read) | 85x | 42x | 86x |
| Write into an output array | 89x | 73x | 44x |
| Recursive `fib(25)` | 12x | 13x | 12x |
| `math` functions loop | 11x | 14x | 15x |
| Helper function in a loop | 64x | 91x | 83x |
| Bit-mixing hash | 112x | 72x | 104x |
| Tiny `add(a, b)` | ≈1x (runs as Python) | ≈1x | ≈1x |

New in 0.7.1 (Linux x86_64, Python 3.13, same CI setup):

| Workload | Plain Python | PyRust 0.7.1 | Speedup |
|---|---:|---:|---:|
| Edit distance of two 288-character strings | 11.2 ms | 0.758 ms | 15x |
| Monte Carlo π with `random.random()` (200k points, same numbers as Python) | 24.9 ms | 3.38 ms | 7.4x |
| `@vectorize` tiny function over 100k rows, vs a Python loop | 13.4 ms | 0.653 ms | 20x |
| Variance of a 100k-float Python list (copy into native memory included) | 6.07 ms | 3.12 ms | 1.9x |

New in 0.7.2 (Linux x86_64, Python 3.13):

| Workload | Baseline | PyRust 0.7.2 | Speedup |
|---|---:|---:|---:|
| `ai.tokenize` on 1.1M characters | 60.4 ms (Python) | 15.4 ms | 3.9x |
| Word count with a dict, 160k words | 22.9 ms (Python) | 13.3 ms | 1.7x |
| `ai.token_f1` over 100 answer pairs (incl. call overhead) | 4.08 ms (Python) | 2.12 ms | 1.9x |
| `ai.levenshtein`, two 500-character strings | 53.0 ms (Python) | 3.09 ms | 17x |
| float32 dot product, 100k elements | 26.0 ms (Python) | 0.219 ms | 119x |
| BM25 query, 1,977 documents | 27.9 ms (pure-Python BM25 scanning documents) | 0.106 ms | 263x |
| Cosine top-10, 50k x 384 float32 | 20.2 ms (vectorized NumPy / BLAS) | 22.1 ms | 0.9x |

Dict and string code gains less than numeric loops because CPython's own dicts and string methods are already written in C; the gain comes from removing the interpreter around them. For dense matrix–vector products NumPy's BLAS is the right tool; `cosine_top_k` is about even with it while avoiding temporary arrays.

Lists are copied for each call, so simple work on big lists gains less; pass a NumPy array to skip the copy. `random` calls copy Python's generator state (about 5 KB) in and out once per call.

## Compared with Numba

The CI job `numba-comparison` runs the same workloads with Numba 0.68 (`fastmath=False`) on Linux, Python 3.12. Speedup over plain Python in that run:

| Workload | PyRust 0.7 | Numba |
|---|---:|---:|
| Integer loop | 28x | 53x |
| Float recurrence | 8.0x | 8.0x |
| Array kernel (read) | 62x | 164x |
| `math` functions loop | 11x | 10x |
| Helper function in a loop | 75x | fails: the helper isn't `@njit` |
| Bit-mixing hash | 85x | 116x |
| Recursive `fib(25)` | 10x | fails: recursion through a plain Python global |
| Write into an output array | 58x | 823x |
| Tiny `add(a, b)` | 1.0x | 0.25x (slower than Python) |

Numba is a mature LLVM JIT and is faster on simple numeric kernels, especially dense array writes. PyRust matches it on math-heavy and float code, runs code Numba rejects, and never changes results. The trade-offs:

| | PyRust | Numba |
|---|---|---|
| Integer overflow | exact (replays in Python) | wraps at 64 bits |
| Unsupported code | runs as Python, with a hint | compile error |
| Plain Python helpers called from the hot function | compiled together | must also be `@njit` |
| Runtime dependencies | none (Rust only to compile) | numba + llvmlite |
| Parallel loops | `prange` with exact-result rules | `prange` (reductions may reorder floats) |

## Reproduce

```sh
git clone https://github.com/surajmhaske02/pyrust-engine && cd pyrust-engine
pip install ".[arrays]"
python -m examples.benchmark_suite --output my-results.json
python benchmarks/compare.py my-results.json
```
