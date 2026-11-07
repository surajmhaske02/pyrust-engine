# Troubleshooting

## Start with `doctor`

```sh
python -m pyrust_engine doctor
```

It checks Python, Rust, the C linker and NumPy, runs a test compile, and prints the exact command to fix anything that is missing.

| Problem | Fix |
|---|---|
| Rust not installed | macOS: `brew install rustup && rustup default stable`; Linux: `curl ... sh.rustup.rs \| sh`; Windows: `winget install Rustlang.Rustup` |
| rustup installed but no toolchain | `rustup default stable` |
| "missing manifest" / half-installed toolchain | `rustup toolchain uninstall stable && rustup default stable` |
| macOS `xcrun: error` | `xcode-select --install` |
| Windows `link.exe` not found | Visual Studio Build Tools with "Desktop development with C++" |
| Linux `cc` not found | `sudo apt install build-essential` |
| Works in the terminal, not in Jupyter | restart the kernel; PyRust also searches `~/.cargo/bin` and Homebrew's rustup; or set `PYRUST_RUSTC` |

## My function is not faster

```python
print(fn.explain())
```

shows whether it compiled and, if not, the line, the reason and a hint. A `PyRustFallbackWarning` is also shown the first time a function runs as plain Python. Common causes:

- Lists, dicts, strings, `print()` inside the function: move them outside or use NumPy arrays.
- `total = 0` then `total += 0.5`: start float totals at `0.0`.
- Tiny functions without loops run as plain Python on purpose (native calls cost about 2–4 µs).
- Arguments of other types (`np.float32` arrays, Python lists) run as Python; convert with `np.asarray(x, dtype=np.float64)`.

## Long-running calls and Ctrl-C

Calls that need more than 50 million loop steps move to a background thread automatically; Ctrl-C stops them. `max_iterations=N` sets a hard limit instead (calls beyond it replay in Python).
