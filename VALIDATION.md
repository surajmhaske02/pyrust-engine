# Validation — 6 November 2026

Executed on Linux x86_64, CPython 3.12.14, Rust 1.75.0.

- Package built into a wheel and installed successfully using pip.
- `python -m unittest discover -s tests -v`: **12 tests passed**, no skips with Rust available.
- Before Rust setup: 9 tests passed; 3 native-only tests skipped.
- Tests cover randomized Python/native result comparisons (100 input rounds), nested loops, branch behavior, loop variable mutation, missing compiler, unsupported operations, argument binding, exact integer guards, big integer inputs, intermediate overflow replay, unbound locals, range shadowing/rebinding, strict rejection, native dispatch and disk cache reuse.
- Strict mode confirmed native execution for all five supported test kernels.
- CI configuration provided for Linux, macOS and Windows; remote CI has **not** run. Only the Linux/Python/Rust combination above was verified locally.

## One measured microbenchmark

`python -m examples.benchmark`, 1,000,000 iterations, median of five warm runs, equality checked against Python:

| Metric | Result |
|---|---:|
| First call including compilation and execution | 0.8771 seconds |
| Python median | 0.067075 seconds |
| Native wrapper median | 0.002502 seconds |
| Warm speedup | 26.81× |
| Native calls | 6 |
| Fallback calls | 0 |
| Compilation cache hit on first call | No |

This is one compiler-friendly integer microbenchmark on this environment. It is not a guarantee for other functions or whole applications. For one invocation, compilation made this example slower than Python; repeated calls amortize startup. No third-party compiler comparison or production load testing was performed.

## Publication

No GitHub repository was created or pushed. GitHub integration was declined for this request. README includes CLI and Git remote instructions for publishing from your own machine.
