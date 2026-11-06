# Contributing to PyRust Engine

Thanks for helping. Bug reports, failing examples and pull requests are all welcome.

## Set up

You need Python 3.10+ and Rust (`rustc` on `PATH`, from <https://rustup.rs>).

```sh
git clone https://github.com/surajmhaske02/pyrust-engine.git
cd pyrust-engine
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[arrays]"
python -m unittest discover -s tests -v
```

Without Rust, the tests that need it are skipped and the rest still run.

## The one rule: results must match Python exactly

Every native operation must either produce exactly what CPython produces (same value, same type, same `repr`, including `-0.0`, NaN and infinities) or return a non-zero status so the call is replayed in Python. When in doubt, replay. A wrong answer is a much worse bug than a missed speedup.

## Where things live

| File | What it does |
|---|---|
| `pyrust_engine/compiler.py` | Python AST → Rust source. `HELPERS` holds the Rust helper functions; `Compiler.expr`, `condition`, `block` translate expressions, conditions and statements. |
| `pyrust_engine/runtime.py` | `@optimize`: compiling, caching, argument checks, guards, calling the library with `ctypes`, fallback, warnings, `explain()`. |
| `pyrust_engine/cache.py` | Disk cache, file locks, hashing. |
| `tests/test_fuzz.py` | Generates random programs and compares native results with CPython. |
| `tests/test_v*.py` | Tests grouped by the release that added the feature. |

## Adding a new Python feature

1. Translate it in `compiler.py`. Use `self.fail(node, "message")` for anything you don't support, so users get the line and code.
2. Return a status code for every case where Rust and Python could differ (overflow, domain errors, special float values).
3. Add tests that compare against plain Python at the edges: 0, -0.0, ±1, NaN, ±inf, the largest and smallest 64-bit values.
4. If it's an operator or function, add it to the extended fuzzer in `tests/test_fuzz.py`.
5. Update the supported-syntax table in `README.md`, `docs/API.md` if needed, and `CHANGELOG.md`.

## Checks before a pull request

```sh
python -m unittest discover -s tests -v
PYRUST_FUZZ_PROGRAMS=300 python -m unittest tests.test_fuzz -v
```

CI runs the tests on Linux, Windows and macOS with Python 3.10, 3.12 and 3.13, and checks that the package builds.

## Reporting a bug

Open an issue with the smallest function that shows the problem, the arguments, the plain-Python result and the PyRust result. `fn.explain()`, `fn.report()` and `fn.rust_source()` output help a lot.

## License

By contributing you agree that your contribution is licensed under the [MIT License](LICENSE).
