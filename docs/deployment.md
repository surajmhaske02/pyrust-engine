# Deployment

PyRust compiles each function the first time it runs, which needs Rust on that machine. For servers, Docker images and applications you ship to users, build the libraries ahead of time instead.

## Build ahead of time

```sh
python -m pyrust_engine build mypackage/kernels.py
```

This compiles every function in the file that PyRust can compile, for the argument types in its annotations (unannotated parameters are `int`), into `mypackage/__pyrust__/` with a `manifest.json`. Ship that folder with your package (for example as package data).

At run time PyRust looks for a matching library before compiling:

1. directories in the `PYRUST_PREBUILT` environment variable (separated by `:` on Linux/macOS, `;` on Windows),
2. `__pyrust__/` next to the module that defines the function.

A library is used only if it was built from the exact same generated code for the same platform (OS and CPU), and its SHA-256 matches the manifest. Anything else falls back to compiling (if Rust is available) or to plain Python.

Build once per platform you deploy to (for example in CI for `linux-x86_64`, `darwin-arm64`, `win32-amd64`) into the same folder: the manifest keeps entries for every platform.

## Docker

```dockerfile
FROM python:3.13-slim
RUN pip install "pyrust-engine[arrays]"
COPY mypackage/ /app/mypackage/          # includes mypackage/__pyrust__/ built for linux-x86_64
```

No Rust needed in the image.

## Cache

Compiled libraries are cached in `~/.cache/pyrust_engine` (override with `PYRUST_CACHE_DIR`). Entries not used for 30 days are removed automatically (`PYRUST_CACHE_MAX_AGE_DAYS`, `0` to disable). `python -m pyrust_engine cache --days 7 --delete` cleans up manually.
