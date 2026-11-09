# 0.7.7

More real code compiles: methods calling methods, `try`/`except`, NumPy slices and more
dtypes, faster calls with lists, and a way to see what runs natively in production.

**Methods calling methods**
- `self.helper(...)` compiles when `helper` is a plain method of the object's class
  (base classes and subclass overrides are found like Python finds them). The runtime
  collects the attributes the whole call chain uses; each call checks that the method is
  unchanged and not hidden by an instance attribute. Recursive methods work with a return
  annotation.
- Lists, dicts and sets stored on objects can be read; list elements can be written.
- `suggest`/`apply` see methods that call methods.

**Exceptions**
- `try`/`except`/`else`/`finally`, `raise` and `assert` compile. An exception replays the
  call in Python, which runs the matching `except` exactly; fast when exceptions are rare.

**NumPy**
- Read-only views: `a[lo:hi]` on 1-D arrays and rows `m[i]` / `m[i, :]`, including
  `for x in a[1:]`.
- `int32`, `uint8` and `bool` arrays: compare, test, convert and store their values (with
  NumPy's out-of-range errors); arithmetic on the values runs as Python (NumPy wraps).
- `np.sqrt`, `np.floor`, `np.ceil`, `np.trunc`, `np.abs`, `np.square`, `np.isnan`,
  `np.isinf`, `np.isfinite`, and `np.pi`, `np.e`, `np.inf`, `np.nan`.

**Faster calls**
- The direct CPython entry point now reads `list[float]` / `list[int]` arguments (and
  copies written lists back) and builds `str` and list-of-number results itself.
- Variance of a 100,000-float Python list: 7.9x faster than plain Python on Linux (was
  1.9x in 0.7.5), 5.8x on macOS, 4.3x on Windows (CI, Python 3.13).

272 tests pass on Linux, macOS and Windows (Python 3.10–3.15) and free-threaded 3.14t/3.15t.

**Production**
- `pyrust_engine.stats()`, `print_stats()`, `reset_stats()`: calls, native calls, Python
  calls and the top reasons per function. `on_fallback(callback)` for metrics and alerts.
  Fallbacks and compiles are logged under the `pyrust_engine` logger.

# 0.7.6

Tools catch up with classes, and everything you see first is current.

**Tools**
- `suggest`, `apply` and the MCP tools `check_code` / `rust_source` look at class methods
  too. Attribute types come from the class (annotations, dataclass fields, class
  attributes, simple `__init__` assignments); `apply` adds `@optimize` at the method's
  indentation. Static and class methods are left alone.
- MCP protocol 2025-11-25: version negotiation (2024-11-05 to 2025-11-25), tool titles and
  hints (`readOnlyHint`, `destructiveHint`, ...), and `structuredContent` with every result.
- `print()` used as a statement gets its own message and hint.

**Project**
- CI: current GitHub Actions (checkout v7, setup-python v7, upload-artifact v7, Pages v5;
  Node 20 is deprecated), free-threaded Python 3.15t next to 3.14t.
- PyPI metadata: an accurate description and classifiers (Python versions, operating
  systems, free threading, typed).
- README, docs home, Getting started and Examples: 0.7.5 benchmark numbers on all three
  platforms, a new speed chart, a release table, class and direct-call examples, a current
  Numba comparison (PyRust is faster on strings and Python lists).
- Notebook: the "can't be sped up" demo used a list, which compiles since 0.7.1; it now
  uses `print()`.

# 0.7.5

Classes, and direct calls for arrays and strings.

**Methods and objects**
- `self.x` and `obj.x` compile: reading attributes (int, float, bool, str, bytes, NumPy
  arrays) and assigning int and float attributes (`self.x = ...`, `self.x += ...`).
  Dataclasses, `__slots__`, plain classes and class attributes work; `@optimize` can be
  used directly on a method.
- Each attribute becomes its own native value; assigned attributes are written back only
  after the native call succeeds, so a replay in Python starts from the untouched object.
- Properties, custom `__getattribute__`/`__setattr__`, frozen dataclasses, calling methods
  on the object and aliasing run as Python with the same result.
- A method updating `self.x`, `self.y`, `self.vx`, `self.vy` in a 20k-step loop: 7.8–12x
  faster than plain Python (CI, Python 3.13).

**Direct calls with arrays, str and bytes**
- The direct CPython entry point now also takes NumPy arrays (read through the buffer
  protocol, without copying), `str` and `bytes`, with the same checks as before: exact
  types and dtypes, C-contiguous, aligned, writable when written, no overlapping writes.
- 1000 calls on a 32-element array: 6.1–8.9x faster than plain Python (about 0.4 µs per
  call); 1000 calls on a 20-character string: 1.8–3.0x. `ai.token_f1` (two str arguments)
  went from 1.9x to 4.6x.

257 tests pass on Linux, macOS and Windows (Python 3.10–3.15) and free-threaded 3.14t.

# 0.7.4

Faster calls and faster text: the two weak spots measured in 0.7.3.

**Direct calls from CPython**
- Functions with `int`/`float`/`bool` arguments and a number, bool or `None` result get a
  C-level entry point that CPython calls directly, instead of going through `ctypes`.
  It reads the arguments, checks their exact types, the 64-bit range, the function's
  code and defaults and every global the compiled code depends on, runs the native code
  without the GIL, and builds the result object. Anything else (other types, keyword
  arguments, big ints, a changed global, a native error) goes to the Python path, so
  results are unchanged.
- The C-API is reached through function pointers from `ctypes.pythonapi`, so the
  generated library links against nothing: same on Linux, macOS, Windows and
  free-threaded Python. `PYRUST_DIRECT_CALLS=0` turns it off; `report()['dispatch']`
  says `'direct'` when it is in use.
- Per call (CI, Python 3.13): 0.18 µs on Linux (was 2.8 µs in 0.7.3), 0.16 µs on macOS
  (was 1.3 µs), 0.19 µs on Windows (was 1.8 µs). Plain Python calls cost about 0.05 µs.

**Text**
- Slices of a `str` argument (`text[a:b]`) are views into the argument instead of copies.
- `@vectorize` passes a text column as one string plus row offsets; each row is a view.
  A 200k-row text column: 3.4–4.5x faster than pandas `Series.map` (was 0.9–2.1x) and
  1.5–2.2x faster than polars `map_elements` (was 0.4–1.0x).
- `tokenize_many` and `chunk_many` process groups of documents (about 64k characters) per
  native call; a group with text that needs Python (non-ASCII letters) falls back to one
  call per document. Measured:
  `tokenize_many` 1.7–1.9x faster than tokenizing one by one (was 1.1–1.2x), `chunk_many`
  2.2–3.5x.

**Fixes**
- Fallback warnings point at the caller's line on every Python version.

242 tests pass on Linux, macOS and Windows (Python 3.10–3.15) and free-threaded 3.14t.

# 0.7.3

Five things AI and data engineers asked for: faster calls, pandas 3 and polars columns,
a drop-in BM25 for `rank_bm25`, compact vector search with filters, and batches on all cores.

**Faster calls**
- Functions with `int`/`float`/`bool` arguments and a number, bool or `None` result get a
  second native entry point that returns the value and the status in one struct. The
  Python side no longer builds out-parameter objects; per-call cost is about 40% lower
  (2.8 µs instead of 4.5 µs on the Linux CI machine).
- Fewer lock and dict operations per call; a failed fast call continues on the normal path
  without running the native code twice.

**pandas 3 and polars**
- `@vectorize` compiles `str`, `bool` and `float32` columns (pandas 3 `str` dtype, `object`
  columns of strings, polars `String`/`Boolean`/`Float32`, lists of str or bool) and may
  return strings. polars Series in, polars Series out (works with `map_batches`).
- `@optimize` accepts pandas `float32` Series, pandas Series of str and polars Series
  (floats, ints, str) without nulls. Series of str allow `s[i]`, `len`, loops and
  `sorted/list/set/min/max/join`; Series operators run as Python with a hint.

**`pyrust_engine.ai`**
- `BM25Okapi`: same constructor, attributes and scores as `rank_bm25.BM25Okapi`
  (bit-for-bit, checked against the package in CI), compiled and sparse.
- `BM25` and `BM25Okapi` indexes: `add()`, `remove()` (ids never change), `save()` /
  `load()` (`.npz`, no pickle), `search(..., mask=)`.
- `hamming_top_k` (binary codes, 32x smaller than float32) and `int8_top_k` (exact integer
  dot products, no int32 copy), `binarize`, `quantize_int8`.
- `mask=` filters on `cosine_top_k`, `top_k` and searches.
- `batch`, `tokenize_many`, `chunk_many`: thread pools that keep input order.

**Language**
- `int.bit_count()`.
- One-character strings use whole-string methods: `','.join(words)`, `'-'.split(...)`
  (they did not compile before).

**Measured honestly (see the benchmarks page):** BM25Okapi queries 15–25x faster than
rank_bm25 with the same scores (building costs about the same); binary search 1.8–4.7x
faster than NumPy; int8 search as fast as NumPy without its int32 copy; text columns in
`@vectorize` only 0.9–2.1x vs pandas `map` and slower than polars `map_elements`;
`tokenize_many` 1.1–1.2x with the GIL.

**Platforms**
- Free-threaded Python 3.14t is tested in CI on Linux, macOS and Windows; the GIL stays off.
  `doctor` reports which build is running.

# 0.7.2

For AI engineers and agents: float32 embeddings, real string processing, lists, dicts and
sets, a ready-made AI library, and tools that coding agents can drive.

**Language**
- **float32 arrays** (`F32Array`, `F32Array2`, NumPy 2) with NumPy's exact promotion rules:
  float32 op float32 stays float32 and Python numbers adapt to float32. Values from
  float64 arrays are tracked so mixed cases run as Python instead of guessing.
- **Strings as values:** slicing, `split`, `strip`, `find`, `rfind`, `index`, `count`,
  `startswith`, `endswith`, `replace`, `join`, `lower`, `upper`, `isdigit`/`isalpha`/...
  on whole strings, `+`, `*`, `+=` (in place), f-strings with ints and strings,
  `str(int)`, `int(s)`, `float(s)`. CPython's whitespace table and index rules.
- **Lists:** `[]`, `append`, `pop`, `extend`, `sort`, `reverse`, `index`, `count`,
  `del lst[i]`, literals, comprehensions, slicing, `sorted()`, `list()`, `min`/`max`
  of a list, `+`, `zip()`, `reversed(range())`, lists of strings.
- **dict and set:** insertion-ordered dicts with `get`/`pop`/`setdefault`/`del`/`in`,
  loops over `items()`/`keys()`/`values()` (size changes during a loop replay and raise
  like Python), `max(d, key=d.get)`; sets with `add`/`discard`/`remove`/`update`,
  `& | - ^`; module-level `frozenset`/tuple constants.
- Lists of strings, dicts and sets as arguments (read-only) and as return values.
- Helper functions may be called as statements to fill a list or dict you pass them.
- Array annotations (`F64Array`, ...) are `Any` for type checkers.

**New: `pyrust_engine.ai`**
- `cosine_top_k`, `top_k`, `BM25`, `tokenize`, `chunk_text`, `ngrams`, `levenshtein`,
  `similarity`, `jaccard`, `minhash`, `minhash_similarity`, `token_f1`, `exact_match`,
  `rouge_l`. Each is plain Python compiled by PyRust, identical to its source.

**Tools for coding agents**
- `python -m pyrust_engine suggest file.py [--json]` and `apply file.py [--write]`.
- MCP server: `python -m pyrust_engine mcp` (also `pyrust-mcp`) with `check_code`,
  `suggest_file`, `apply_file`, `rust_source`, `benchmark`, `doctor`.
- `AGENTS.md` and `llms.txt`.

**Examples and docs**
- Hybrid RAG retriever, FastAPI service, evaluation script, Docker image that runs
  without Rust; new "AI and agents" docs page.

# 0.7.1

Everyday Python, not only numbers: lists, strings, pandas columns, random numbers, and
`vectorize` for small functions over whole columns.

**Language**
- **Lists:** `list[float]` and `list[int]` arguments can be read, looped over and
  written. Written lists are copied back into the same list object after the call
  succeeds. New lists from `[0.0] * n` / `[0] * n` can be filled and returned.
- **Strings and bytes:** `len(s)`, `s[i]`, `for ch in s`, `enumerate(s)`, character
  comparisons, `ch in "aeiou"`, `ch in s`, `ord`, `chr`, `isdigit isalpha isalnum
  isspace isupper islower lower upper` (exact for ASCII; other characters replay), and
  one-character return values. `bytes` elements are ints.
- **pandas Series** (`float64`/`int64`, default index) are read without copying:
  `s[i]` keeps Python's label semantics (`s[-1]` raises `KeyError`), `s.iloc[i]` is
  positional.
- **random:** `random()`, `uniform`, `randint`, `randrange`, `gauss` and `choice`
  continue Python's own Mersenne Twister sequence bit for bit, so `random.seed()` and
  the following draws behave exactly as in plain Python.

**New: `vectorize`**
- `@vectorize` (or `@optimize.vectorize`) runs a one-value function over NumPy arrays,
  lists or pandas columns in one native loop, returning a NumPy array (or a Series with
  the same index). `parallel=True` uses all CPU cores. This is the fix for tiny
  functions: no per-element call overhead.

**Other**
- Hints updated for the new features; the guide documents running compiled functions
  in web servers from worker threads (compiled calls don't hold the GIL).
- New CI benchmarks: edit distance on strings, list variance, Monte Carlo with
  `random`, and `vectorize` versus a Python loop.
- The generated code changed, so libraries built by `python -m pyrust_engine build`
  with 0.7.0 are rebuilt (or run as Python until you run `build` again).

# 0.7.0

Everything on the improvement list: much more Python compiles, no budget trap, safer
array writes, parallel loops, deployment without Rust, and better tooling.

**Language**
- `bool` values and parameters, `and`/`or`/`not` as values, bools in arithmetic.
- `for x in array`, `for i, x in enumerate(array, start)`, bare `a[i]` reads.
- `sum(array)`, `round(x)`, `divmod(a, b)`, `bool(x)`, `math.atan2`, `math.copysign`,
  `math.fmod` (CPython's exact special cases). `math.hypot` is explained instead of
  compiled, because CPython uses its own algorithm.
- Tuples: `a, b = b, a + b`, `q, r = divmod(...)`, `rows, cols = m.shape`, and
  functions that return several values.
- Keyword and default arguments in calls to your own functions; helpers may write
  arrays they receive.
- 2-D arrays (`F64Array2` / `I64Array2`): `a[i, j]`, `a[i][j]`, `.shape`, `.size`.
- New arrays from `np.zeros` / `np.empty` / `np.ones` (and `_like`, `dtype=`) can be
  filled and returned.
- `prange()` parallel loops on all CPU cores, limited to patterns whose result is
  exactly the sequential one (each iteration writes its own element; int `+=` totals).
- Error messages end with a `hint:` line that says how to rewrite the code.

**Runtime**
- One native specialization per combination of argument types, so unannotated
  functions and floats passed to int-annotated parameters run natively.
- No budget trap: calls longer than 50M steps re-run in a background thread with no
  limit, and Ctrl-C stops them. `max_iterations=N` still sets a hard limit.
- Array writes use an undo log instead of copying the whole array on every call.
- Rust panics are caught and replayed in Python instead of aborting the process.
- Per-call timing is opt-in (`timing=True`), which trims the wrapper. A faster call path
  than `ctypes` is still on the roadmap (native calls cost about 2–3 µs).
- Benchmarks (Linux CI): writing an output array is 89x faster than Python (0.6: 179x,
  which copied the whole array on every call; the undo log avoids that copy for sparse
  writes and switches to one snapshot for dense ones).
- Comparisons between ints above 2**53 and floats replay in Python (exact for both
  Python floats and NumPy scalars).

**Deployment and tools**
- `python -m pyrust_engine build file.py` compiles functions ahead of time into
  `__pyrust__/`; machines without Rust load them (verified by SHA-256 and platform).
- Jupyter: `%load_ext pyrust_engine`, `%pyrust_check`, `%pyrust_benchmark`.
- Automatic cache cleanup (entries unused for 30 days, `PYRUST_CACHE_MAX_AGE_DAYS`).
- Code formatted with ruff; ruff and mypy run in CI.
- CI: Python 3.10 to 3.15 on Linux, macOS and Windows (15 jobs), coverage summary,
  benchmark comparison with the last release in every job summary, and a Numba
  comparison job.
- Documentation website (MkDocs Material on GitHub Pages) with a guide, examples
  gallery, benchmarks, deployment and troubleshooting.

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
