# Getting started

## 1. Requirements

| Needed | Version | Check |
|---|---|---|
| Python | 3.10 or newer | `python3 --version` |
| Rust | any recent stable | `rustc --version` |
| C linker | system default | installed with the steps below |
| NumPy | optional, for arrays | `pip install "pyrust-engine[arrays]"` |

Without Rust everything still runs correctly as plain Python, and PyRust shows one warning with the exact fix.

## 2. Install Rust

=== "macOS"
    ```sh
    xcode-select --install          # C linker, skip if already installed
    brew install rustup && rustup default stable
    ```
=== "Linux"
    ```sh
    sudo apt install build-essential    # Fedora: sudo dnf install gcc
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    ```
=== "Windows"
    Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with
    "Desktop development with C++", then run `winget install Rustlang.Rustup`.

## 3. Install PyRust Engine

```sh
pip install "pyrust-engine[arrays]"
python -m pyrust_engine doctor
```

`doctor` checks Python, Rust, the linker and NumPy, and compiles a test function.

## 4. Speed up a function

```python
from pyrust_engine import optimize

@optimize
def collatz_steps(limit: int) -> int:
    longest = 0
    for start in range(1, limit):
        n, steps = start, 0
        while n != 1:
            n = n // 2 if n % 2 == 0 else 3 * n + 1
            steps += 1
        longest = max(longest, steps)
    return longest

collatz_steps(1_000_000)
print(collatz_steps.explain())
```

The first call compiles the function (about 0.1–0.4 s, cached on disk afterwards). Later calls run natively.

## 5. Methods work too

```python
from dataclasses import dataclass

@dataclass
class Account:
    balance: float
    rate: float

    @optimize
    def grow(self, years: int) -> float:
        for _ in range(years):
            self.balance += self.balance * self.rate   # written back to the object after the call
        return self.balance
```

## 6. Let the tools find what to speed up

```bash
python -m pyrust_engine suggest app.py          # which functions and methods compile, and what blocks the rest
python -m pyrust_engine apply app.py --write    # add @optimize where it compiles and helps
```

## 7. In Jupyter

```python
%load_ext pyrust_engine
%pyrust_check collatz_steps
%pyrust_benchmark collatz_steps(100_000)
```
