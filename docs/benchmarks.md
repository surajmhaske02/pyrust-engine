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

New in 0.7.3 (CI, Python 3.13; Linux x86_64 / macOS Apple Silicon / Windows x86_64):

| Workload | Baseline | Linux | macOS | Windows |
|---|---|---:|---:|---:|
| Native call of a tiny function (`native_loop_free=True`), per call | 0.7.2: 4.5 µs on Linux | **2.8 µs** | 1.3 µs | 1.8 µs |
| `BM25Okapi` query, 5,000 documents | `rank_bm25` | **15x** | 25x | 25x |
| `BM25Okapi` build, 5,000 documents | `rank_bm25` | 1.1x | 0.7x | 0.9x |
| `hamming_top_k`, 200k x 384-bit codes | NumPy (`np.bitwise_count`) | **3.4x** | 1.8x | 4.7x |
| `int8_top_k`, 100k x 384 int8 | NumPy (cast to int32, then matmul) | 1.0x | 1.0x | 1.6x |
| `tokenize_many`, 2,000 documents | `tokenize` one by one (also compiled) | 1.2x | 1.1x | 1.2x |
| `@vectorize` over a 200k-row **text** column | pandas `Series.map` | 1.4x | 2.1x | 0.9x |
| `@vectorize` over a 200k-row **text** column | polars `map_elements` | 0.5x | 1.0x | 0.4x |

What these numbers mean:

- **BM25Okapi** returns exactly rank_bm25's scores. Queries touch only the documents that contain the words, so they are much faster; building the index costs about the same.
- **Binary codes** are 32x smaller than float32 and faster to search than with NumPy. **int8** search is about as fast as NumPy, but NumPy needs an int32 copy of the whole matrix on every query (4x the memory of the int8 matrix), and `int8 @ int8` in NumPy overflows.
- **Text columns** (0.7.3 numbers): converting each string separately cost about as much as the work. 0.7.4 passes the column as one string, see below.
- **Threads with the GIL:** converting strings in and out takes turns on the GIL; 0.7.4 groups documents per native call to reduce that. On free-threaded Python (3.14t) nothing takes turns.

New in 0.7.4 (CI, Python 3.13):

| Workload | Baseline | Linux | macOS | Windows |
|---|---|---:|---:|---:|
| Native call of a tiny function, per call | 0.7.3: 2.8 / 1.3 / 1.8 µs | **0.18 µs** | **0.16 µs** | **0.19 µs** |
| `@vectorize` over a 200k-row text column | pandas `Series.map` | **4.5x** | 3.9x | 3.4x |
| `@vectorize` over a 200k-row text column | polars `map_elements` | **1.7x** | 2.2x | 1.5x |
| `tokenize_many`, 2,000 documents | `tokenize` one by one (also compiled) | **1.65x** | 1.9x | 1.8x |
| `chunk_many`, 2,000 documents | `chunk_text` one by one (also compiled) | **2.6x** | 3.5x | 2.2x |

Calls with `int`/`float`/`bool` arguments now go from CPython straight into the native code (no `ctypes`), which makes a native call about 10x cheaper. Text columns are passed as one string with row offsets, and each row is a view into it, so there is no per-row conversion. Batches process groups of documents per native call.

New in 0.7.5 (CI, Python 3.13):

| Workload | Baseline | Linux | macOS | Windows |
|---|---|---:|---:|---:|
| Method updating `self.x`, `self.y`, `self.vx`, `self.vy`, 20k steps | plain Python | **12x** | 7.8x | 10x |
| 1000 calls on a 32-element float64 array | plain Python | **8.9x** | 6.1x | 6.3x |
| 1000 calls on a 20-character string | plain Python | **1.8x** | 3.0x | 2.1x |

Methods pass each attribute as its own native value and write assigned ones back once, after the loop. Calls with NumPy arrays, `str` and `bytes` now skip `ctypes`: the arrays are read through the buffer protocol without copying.

Note: string- and dict-heavy rows (`dict_word_count`, `str_tokenize`) vary a lot between CI machines (1.0–1.9x for the word count on Linux within one run, across Python versions); compare several runs before reading a change into them.

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
