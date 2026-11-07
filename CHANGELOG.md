# 0.6.1

Clear, actionable handling of missing or broken Rust installs.

- **Diagnosis with the exact fix** for: Rust not installed, rustup without a selected
  toolchain, a half-installed toolchain ("missing manifest"), rustc that won't start or
  hangs, and missing C linkers (macOS command line tools, Windows Build Tools, Linux
  `cc`). Each message says that results are still correct as plain Python.
- **One warning per process** (`RustToolchainWarning`, a `PyRustFallbackWarning`) for
  toolchain problems instead of one per function.
- **`python -m pyrust_engine doctor`**: checks Python, Rust, NumPy and runs one real test
  compile; exit code 0 when native code works. Also `pyrust_engine.doctor()` and
  `pyrust_engine.check_rust()`.
- **Finds rustc outside `PATH`** in the usual install folders (`~/.cargo/bin`, Homebrew's
  rustup), which Jupyter and IDEs often miss; `PYRUST_RUSTC` selects a specific compiler.
- `fn.explain()` shows the toolchain problem and fix; `report()` has `toolchain_problem`.
- README: requirements table, per-OS install steps, troubleshooting table, and absolute
  links so the PyPI page shows the chart and links correctly.

# 0.6.0

Recursion, writing into arrays, int64 arrays, exact int/float comparisons, type hints,
and the docs and tooling for contributors.

- **Recursion**, including mutual recursion, for functions annotated `-> int` or
  `-> float`. Native recursion is capped at 300 levels and every call between compiled
  functions spends one unit of `max_iterations`; deeper or longer runs replay in Python.
- **Array writes:** `a[i] = v`, `a[i] += v`, `-=`, `*=` (and `/=` for `F64Array`).
  Written arrays use a private copy that is published only when the native call
  succeeds, so a replay always starts from the original data. Read-only arrays and
  written arrays that share memory with another argument run in Python.
- **`I64Array`** parameters: `int(a[i])`, `float(a[i])` and int writes.
- **Functions without a return value** (Python returns `None`) compile.
- **Exact int vs float comparisons** between variables (`i < x`), matching CPython at the
  2**53 and 2**63 boundaries and for NaN and infinities.
- **Type hints:** `OptimizedFunction` protocol, `optimize()` overloads, `py.typed`.
- **Docs and tooling:** `docs/API.md`, `CONTRIBUTING.md`, issue and pull request
  templates, VALIDATION.md for 0.4 to 0.6, and an updated comparison notebook.
- **CI:** a package job builds the sdist and wheel, runs `twine check --strict` and
  installs the wheel in a clean environment. A manual "Publish to PyPI" workflow uses
  trusted publishing (needs a one-time setup on pypi.org).
- **Benchmarks:** recursive `fib` and an array-writing smoothing kernel; README speed
  numbers for Linux, macOS (Apple Silicon) and Windows from CI.

# 0.5.0

Much more Python compiles natively, plus tools that tell you why code doesn't.

- **Operators:** integer `**` (non-negative exponents), `& | ^ ~ << >>`, `int / int`
  (exact when both fit in 2**53, otherwise replayed); float `//`, `%` and `**` using
  CPython's exact `float_divmod`, `float_rem` and `float_pow` rules.
- **Builtins:** `abs`, `min` / `max` with any number of arguments, `int(x)`, `pow(a, b)`.
- **`math`:** `sqrt exp log log2 log10 sin cos tan asin acos atan sinh cosh tanh fabs`,
  `floor ceil trunc`, `isnan isinf isfinite`, `pi e tau inf nan`, as `math.f` or
  `from math import f`. Domain errors and overflow replay in Python, so the same
  exception is raised (new native status 6).
- **Calls between functions:** plain or `@optimize` functions called from a compiled
  function are compiled into the same library (recursion stays in Python). Callee names
  and code objects are checked on every call; a change falls back to Python.
- **Syntax:** `x if c else y`, `while True`, module-level int/float constants (guarded
  against reassignment).
- **Developer tools:** error messages include function name, line number and the
  offending code; `PyRustFallbackWarning` once per reason (`warn_on_fallback=False` to
  silence); `fn.explain()`; `python -m pyrust_engine check file.py [--json]`.
- Loop-free functions that call another function are compiled instead of being routed
  to plain Python.
- Tests: `test_v5.py` (edge cases for every new operation) and an extended fuzzer;
  89 tests in total, passing on Linux, Windows and macOS.
- Benchmarks: new math, helper-call and bit-hash workloads; README numbers now come from
  a CI run on GitHub Actions `ubuntu-latest`.

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
