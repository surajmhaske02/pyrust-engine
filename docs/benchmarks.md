# Benchmarks

Speedup over plain Python, measured by CI on GitHub Actions runners with Python 3.13 and Rust 1.99. Raw samples are in the [`benchmarks/`](https://github.com/surajmhaske02/pyrust-engine/tree/main/benchmarks) folder; every CI run also compares against the last release and prints the table in the job summary.

| Workload | Linux x86_64 | macOS arm64 | Windows x86_64 |
|---|---:|---:|---:|
| Integer loop | 35.9x | 39.5x | 37.5x |
| Float recurrence | 12.0x | 5.3x | 8.9x |
| Array kernel (read) | 73.8x | 37.8x | 69.7x |
| Write into an output array | 179x | 128x | 152x |
| Recursive `fib(25)` | 15.1x | 17.0x | 19.9x |
| `math` functions loop | 10.6x | 15.3x | 15.5x |
| Helper function in a loop | 68.1x | 87.2x | 61.6x |
| Bit-mixing hash | 126x | 60.0x | 131x |
| Tiny `add(a, b)` | ≈1x (runs as Python) | ≈1x | ≈1x |

## Compared with Numba

The CI job `numba-comparison` runs the same workloads with Numba (`fastmath=False`) and uploads the timings. Numba is a mature JIT and is usually faster on the same kernels; PyRust's trade-offs are different:

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
