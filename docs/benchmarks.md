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
