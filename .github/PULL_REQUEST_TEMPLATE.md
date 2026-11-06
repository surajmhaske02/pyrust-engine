## What this changes

## How results stay identical to Python

<!-- Every native path must match CPython exactly or return a status so the call replays in Python. -->

## Checklist

- [ ] Tests compare against plain Python, including edge cases (0, -0.0, NaN, ±inf, 64-bit limits)
- [ ] `python -m unittest discover -s tests -v` passes
- [ ] README / docs/API.md / CHANGELOG.md updated if behaviour changed
