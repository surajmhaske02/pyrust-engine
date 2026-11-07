# PyRust Engine API reference (0.6.1)

Everything public is importable from `pyrust_engine`:

```python
from pyrust_engine import (optimize, F64Array, I64Array, OptimizedFunction,
                           OptimizationError, PyRustFallbackWarning, RustToolchainWarning,
                           check_rust, doctor, Unsupported, prune_cache)
```

## `optimize`

```python
@optimize
def f(...): ...

@optimize(fallback=True, cache_dir=None, max_iterations=10_000_000,
          native_loop_free=False, warn_on_fallback=True)
def g(...): ...

h = optimize(plain_function)          # also works as a function call
```

| Option | Default | What it does |
|---|---|---|
| `fallback` | `True` | When a call can't run natively, run it as plain Python. `False` raises `OptimizationError` instead (useful in benchmarks and tests). |
| `cache_dir` | `~/.cache/pyrust_engine` | Where compiled libraries are stored. Must be private to you on Linux/macOS. |
| `max_iterations` | `10_000_000` | Native budget: every loop iteration and every call between compiled functions spends one unit. A call that runs out is replayed in Python. |
| `native_loop_free` | `False` | Functions with no loop and no calls run as plain Python because native dispatch would be slower. `True` compiles them anyway. |
| `warn_on_fallback` | `True` | Emit `PyRustFallbackWarning` the first time each distinct reason makes a call run as plain Python. |
| `backend` | `'rust'` | Only `'rust'` exists. |

`optimize` returns an `OptimizedFunction`: call it exactly like the original.

## `OptimizedFunction` methods

| Method | Returns | Description |
|---|---|---|
| `fn(...)` | the original's result | Runs natively when possible, otherwise as plain Python. Results and exceptions are always the original's. |
| `fn.explain()` | `str` | Whether the function compiles; if not, the function, line and code that stop it. Lists parameter types, return type and compiled helper functions. |
| `fn.report()` | `dict` | Counters and timings (see below). |
| `fn.prepare(background=False)` | `dict` | Compile now. With `background=True`, compile in a thread; calls run as Python until it's ready. |
| `fn.reset()` | `dict` | Forget the compiled library and compile again on the next call. |
| `fn.rust_source()` | `str` | The generated Rust code. |
| `fn.python` | function | The original, undecorated function. |

## `report()` fields

| Field | Meaning |
|---|---|
| `backend` | `'rust'`, `'python'`, `'pending'` (not compiled yet) or `'compiling'`. |
| `toolchain_problem` | `True` when the last compile failed because Rust or the linker is missing or broken. |
| `dispatch` | `'wrapper'` (normal) or `'python-direct'` (loop-free function running as plain Python with no wrapper). |
| `call_counting` | `False` for `python-direct` functions, whose calls are not counted. |
| `reason` | Why the latest call ran as Python, or `None`. |
| `compile_error` | Why compilation failed, or `None`. |
| `calls`, `native_calls`, `fallback_calls` | Call counters. A native attempt that is then replayed counts in both `native_calls` and `fallback_calls`. |
| `fallback_reasons` | `{reason: count}` for every fallback so far. |
| `fallback_rate` | `fallback_calls / calls`. |
| `compile_seconds`, `compile_attempts` | Time spent compiling, and how many times. |
| `cache_hit`, `cache_rebuilt` | Whether the library came from the disk cache, and whether a corrupt cache entry was rebuilt. |
| `native_seconds`, `native_average_seconds` | Time inside native code. |
| `fallback_seconds`, `fallback_average_seconds` | Time running as plain Python. |
| `wrapper_seconds` | Total time including checks and conversions. |

## Parameter and return types

| Annotation | Native type | Accepted values |
|---|---|---|
| none or `int` | 64-bit integer | exact `int` between -2**63 and 2**63-1 (not `bool`) |
| `float` | 64-bit float | exact `float` |
| `F64Array` | `float64` array | 1-D, C-contiguous, aligned, native-endian NumPy `float64` array |
| `I64Array` | `int64` array | same, `int64` |

Anything else passed at call time runs as plain Python. Return annotations are only needed for recursive functions (`-> int` or `-> float`). A function with no `return value` returns `None`.

## Fallback reasons and native statuses

| Reason in `report()` | When |
|---|---|
| `integer overflow` | A 64-bit integer result doesn't fit. Python's big integers take over. |
| `zero divisor or zero range step` | Division or modulo by zero, or `range(..., 0)`. |
| `array index out of bounds` | `a[i]` outside the array. |
| `native iteration budget exceeded` | More than `max_iterations` loop iterations or calls. |
| `no scalar return` | A value-returning function reached its end without `return`. |
| `value outside the native fast path ...` | Math domain errors, float overflow, negative integer exponents, huge int division, recursion deeper than 300 levels. |
| `<name> has been rebound` | A builtin, `math`, global constant or helper function was reassigned after compiling. |
| `<param>: expected ...` | An argument had a type or array layout the native code can't take. |
| `array arguments share memory ...` | A written array overlaps another array argument. |
| compile messages | `<function>, line <n>: <problem>: \`<code>\`` |

In every case the call is run (or re-run) as plain Python, so the result or exception is exactly Python's.

## Warnings and errors

- `PyRustFallbackWarning` (subclass of `RuntimeWarning`): emitted once per reason per function.
- `RustToolchainWarning` (subclass of `PyRustFallbackWarning`): Rust is missing or broken. Emitted **once per process**, because it affects every function; the message includes the fix. Silence with `warn_on_fallback=False` or `warnings.filterwarnings('ignore', category=PyRustFallbackWarning)`.
- `OptimizationError`: raised instead of falling back when `fallback=False`.
- `Unsupported`: raised by the compiler internally; you see it as a fallback reason or in `explain()`.

## Checking the Rust toolchain

```python
pyrust_engine.check_rust()
# {'ok': True, 'rustc': '/Users/you/.cargo/bin/rustc', 'version': '1.99.1', 'problem': None, 'fix': []}

pyrust_engine.doctor()      # prints a full report and does one real test compile; returns True/False
```

When Rust is missing or broken, `check_rust()` returns `ok: False`, a one-line `problem` and the `fix` commands. PyRust finds `rustc` on `PATH`, then in `$CARGO_HOME/bin`, `~/.cargo/bin` and Homebrew's rustup folder. Set `PYRUST_RUSTC` to force a specific compiler.

| Problem detected | Fix suggested |
|---|---|
| rustc not found | OS-specific install commands |
| no default toolchain | `rustup default stable` |
| half-installed toolchain | `rustup toolchain uninstall stable && rustup default stable` |
| rustc hangs or won't start | `rustup default stable` / `rustup update stable` |
| macOS developer tools missing | `xcode-select --install` |
| Windows `link.exe` missing | Visual Studio Build Tools with C++ |
| Linux `cc` missing | `sudo apt install build-essential` |

## Command line

```sh
python -m pyrust_engine doctor                     # Python, Rust, linker, NumPy + a test compile
python -m pyrust_engine check file.py [--json]     # explain every function in a file
python -m pyrust_engine profile script.py [args]    # run a script once, rank hot functions
python -m pyrust_engine cache [--days 30] [--delete] [--directory DIR]
```

`check` loads the file without running its `if __name__ == "__main__":` block. `profile` runs the script. Only use either on trusted code.

## `prune_cache`

```python
prune_cache(path=None, *, older_than_days=30, dry_run=True)
```

Lists (and with `dry_run=False` deletes) cache entries older than the given age.
