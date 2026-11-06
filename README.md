# PyRust Engine

An experimental middleware that compiles a conservative subset of Python integer functions into Rust native libraries. Add `@optimize` to a computational function. Unsupported functions and values run in Python by default.

**This is a working prototype, not a general Python compiler.** It does not automatically profile or rewrite whole applications. Floats, strings, lists, arrays, classes, third-party calls, async code, generators and arbitrary dynamic Python are not accelerated in v0.1.

## Install

Requires Python 3.10+ and Rust (`rustc`, edition 2021 support) on PATH for acceleration. Install Rust from https://rustup.rs. Windows also needs the matching C linker/build tools. No Rust crates or third-party Python runtime packages are needed. Without Rust, functions continue in Python and the report explains why.

From this repository:

```sh
python -m pip install .
python -m unittest discover -s tests -v
python -m examples.benchmark
```

## Usage

Save the function in a `.py` file; interactive source inspection may be unavailable.

```python
from pyrust_engine import optimize

@optimize(backend="rust", fallback=True)
def calculate(n: int) -> int:
    total = 0
    for i in range(n):
        total += i * i
    return total

print(calculate(1_000_000))
print(calculate.report())
print(calculate.rust_source())
assert calculate(10) == calculate.python(10)
```

Compilation happens on the first eligible call. Subsequent calls reuse the library. Disk cache keys include generated code, Rust version/host, platform, ABI version and optimization flags. Calls with keyword arguments and ordinary defaults are supported. Type annotations are optional; runtime arguments must be exact built-in `int` values fitting signed 64 bits.

## Compatibility

| Feature | Native behavior |
|---|---|
| Integer literals, variables, `+`, `-`, `*`, unary minus | Checked signed 64-bit operations |
| Assignments and `+=`, `-=`, `*=` | Supported for definitely assigned local integers |
| `for` over built-in `range(stop)` or `range(start, stop)` | Supported; negative bounds and nested loops supported |
| `if` / `else`, single integer comparisons | Supported |
| Final integer `return`, early integer returns | Supported conservatively |
| Python big integers or intermediate overflow | Python fallback; no silent wrapping |
| Lists, floats, division, modulo, calls, while, break, range steps | Python fallback |
| Closures, global reads, side effects, methods | Outside the native subset |
| Missing source/compiler, compiler failure | Python fallback with reason |

Unsupported source is rejected before native execution. No user source is interpolated as raw Rust syntax: the compiler generates code only from allowed AST nodes and maps identifiers to internal names. The accepted subset has no external side effects, so overflow replay does not duplicate external actions. This is conservative engineering, not a formal equivalence proof.

Python exceptions remain Python exceptions when falling back. `fallback=False` raises `OptimizationError` on rejected source, inputs, compiler errors or native overflow. This option is useful when benchmarking so fallback cannot silently masquerade as acceleration.

`report()` reports loaded backend, last reason, native/fallback call counts and cache reuse. A `rust` backend means the library loaded; an individual call can still fall back. Counters are diagnostic, not a thread-safe telemetry system.

## Architecture

1. Inspect function source and parse Python AST.
2. Reject unsupported operations, closure capture and possibly unbound variables.
3. Generate Rust `Option<i64>` operations using `checked_add/sub/mul/neg`.
4. Invoke `rustc` to build a native `cdylib`, then atomically publish to the cache.
5. Call through Python's standard-library `ctypes` C ABI.
6. Return an integer or execute the original Python function on overflow/rejection.

This version uses **ctypes rather than PyO3/Maturin**, avoiding crate downloads and Python-version-specific extension builds. A future array/object interface could use PyO3. Native calls release the GIL through ctypes. Do not mutate function code, globals or builtins concurrently with execution.

## Benchmark honestly

`python -m examples.benchmark` checks equal results, reports first-call latency separately, and compares median warm timings over five runs. Speedup is null if Rust did not load. The included arithmetic benchmark is small and compiler-friendly; its result is not an application-wide performance promise. Measure representative workloads and include conversion, startup and compilation costs in your application budget. No comparison against Numba, Cython, PyPy or Codon is included yet.

## Safety and limitations

Only run trusted local Python. This is **not** a sandbox for uploaded or hostile code; fallback executes normal Python. Keep the native cache private and do not use shared writable cache directories. Native libraries execute inside your process. Very long native loops cannot be promptly interrupted by Python signals. Avoid applying the decorator to functions whose behavior depends on tracing, stack inspection or concurrent monkey-patching.

Functions created with `exec`, notebook cells without retrievable source, decorated wrappers and unusual builtins environments may fall back. First-call compilation has a 120-second timeout. Per-function locking prevents duplicate compilation within one wrapper; separate processes may compile the same cache key concurrently. Compile failures remain Python-only for that wrapper; recreate it or restart after fixing the compiler environment.

CI is configured for Ubuntu, Windows and macOS with Python 3.10/3.12/3.13. Configuration alone is not proof those jobs have passed. See VALIDATION.md for checks actually run in this session.

## Push to GitHub

Extract this folder and run the commands below with Git and the GitHub CLI installed. The example creates a **private** repository; choose public only if you intend to publish the source.

```sh
git init -b main
git add .
git commit -m "Initial Python Rust middleware prototype"
gh auth login
gh repo create pyrust-engine --private --source=. --remote=origin --push
```

Alternatively create an empty repository at https://github.com/new, then:

```sh
git remote add origin https://github.com/YOUR_USERNAME/pyrust-engine.git
git push -u origin main
```

Choose a license before making this repository public. No license has been selected on your behalf.

## Roadmap

Profile-guided hotspot suggestions; typed array kernels; float semantic policy; additional integer operations with Python-compatible errors; stronger differential/property testing; PyO3 object interoperability; comparative benchmarks. These are planned, not implemented.
