"""Find and diagnose the Rust toolchain PyRust needs, with a precise fix for each problem.

PyRust never needs Rust to give correct answers: without a working toolchain every
function runs as plain Python. These helpers make that situation obvious and fixable.
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from .compiler import Unsupported

INSTALL_URL = 'https://rustup.rs'


class ToolchainError(Unsupported):
    """The machine can't compile Rust right now (as opposed to the function being unsupported).

    The message says what is wrong and how to fix it.
    """


def _extra_dirs():
    """Places rustc is usually installed but that may be missing from PATH
    (common in Jupyter, IDEs and GUI apps that don't read shell profiles)."""
    home = Path.home()
    dirs = [home / '.cargo' / 'bin']
    cargo_home = os.environ.get('CARGO_HOME')
    if cargo_home:
        dirs.insert(0, Path(cargo_home) / 'bin')
    if sys.platform == 'darwin':
        dirs += [
            Path('/opt/homebrew/opt/rustup/bin'),
            Path('/usr/local/opt/rustup/bin'),
            Path('/opt/homebrew/bin'),
            Path('/usr/local/bin'),
        ]
    elif sys.platform != 'win32':
        dirs += [Path('/usr/local/bin'), Path('/usr/bin')]
    return [str(d) for d in dirs]


def find_rustc():
    """Path to rustc, or None. Honours PYRUST_RUSTC, then PATH, then the usual install folders."""
    override = os.environ.get('PYRUST_RUSTC')
    if override:
        return override if Path(override).is_file() else None
    return shutil.which('rustc') or shutil.which('rustc', path=os.pathsep.join(_extra_dirs()))


def install_steps():
    """Platform-specific commands to install Rust and the linker it needs."""
    if sys.platform == 'darwin':
        return [
            'brew install rustup && rustup default stable     (or follow https://rustup.rs)',
            'xcode-select --install                             (C linker, if not installed yet)',
        ]
    if sys.platform == 'win32':
        return [
            'winget install Rustlang.Rustup                     (or download rustup-init.exe from https://rustup.rs)',
            'Visual Studio Build Tools with "Desktop development with C++" (the linker Rust uses on Windows)',
        ]
    return [
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh",
        'sudo apt install build-essential                   (C linker; on Fedora: sudo dnf install gcc)',
    ]


def _advice(problem, steps):
    lines = [problem, 'Your code still runs correctly as plain Python, just without the speedup.', 'To fix:']
    lines += [f'  {s}' for s in steps]
    lines.append('Then restart Python (or the Jupyter kernel). Check with: python -m pyrust_engine doctor')
    return '\n'.join(lines)


def not_found_error():
    return ToolchainError(_advice('Rust compiler (rustc) not found.', install_steps()))


def classify_version_failure(output):
    """Turn a failed `rustc -vV` (often rustup without a usable toolchain) into a ToolchainError."""
    text = output or ''
    lowered = text.lower()
    if 'no default toolchain' in lowered or 'no override and no default' in lowered:
        return ToolchainError(
            _advice('rustup is installed but no Rust toolchain is selected.', ['rustup default stable'])
        )
    if (
        'missing manifest' in lowered
        or 'is not installed' in lowered
        or 'not a directory' in lowered
        or ("command failed: 'rustc'" in lowered and 'no such file' in lowered)
    ):
        # rustup is present but the toolchain it points to is missing files.
        return ToolchainError(
            _advice(
                'The Rust toolchain is installed incompletely (an install was probably interrupted).',
                ['rustup toolchain uninstall stable && rustup default stable'],
            )
        )
    first = next((line.strip() for line in text.splitlines() if line.strip()), 'no output')
    return ToolchainError(
        _advice(
            f'rustc is installed but does not run: {first}',
            ['rustup update stable     (or reinstall from https://rustup.rs)'],
        )
    )


def classify_build_failure(stderr):
    """A ToolchainError when a build failed because of the environment (no linker,
    missing SDK, ...), or None when the generated code itself was the problem."""
    text = stderr or ''
    lowered = text.lower()
    if 'xcrun: error' in lowered or 'invalid active developer path' in lowered:
        return ToolchainError(
            _advice(
                'The macOS command line developer tools (the C linker Rust uses) are missing.',
                ['xcode-select --install'],
            )
        )
    if 'link.exe' in lowered and ('not found' in lowered or 'could not be found' in lowered):
        return ToolchainError(
            _advice(
                'The Microsoft C++ linker (link.exe) that Rust uses on Windows is missing.',
                [
                    'Install Visual Studio Build Tools with "Desktop development with C++" '
                    '(https://visualstudio.microsoft.com/visual-cpp-build-tools/)'
                ],
            )
        )
    if (
        'linker `cc` not found' in lowered
        or 'linker `gcc` not found' in lowered
        or ('linking with `cc` failed' in lowered and 'not found' in lowered)
    ):
        return ToolchainError(
            _advice('No C linker (cc) found; Rust needs one to build libraries.', [install_steps()[1]])
        )
    return None


_version_cache: dict[str, str] = {}


def rustc_version(rustc):
    """`rustc -vV` output; cached per path. Raises ToolchainError with a fix when it fails."""
    if rustc in _version_cache:
        return _version_cache[rustc]
    try:
        result = subprocess.run([rustc, '-vV'], capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        raise not_found_error() from None
    except subprocess.TimeoutExpired:
        raise ToolchainError(
            _advice(
                'rustc did not answer within 30 seconds (rustup may be installing a toolchain).',
                ['rustup default stable     (run it once in a terminal and let it finish)'],
            )
        ) from None
    except OSError as exc:
        raise ToolchainError(
            _advice(
                f'rustc could not be started: {exc}', ['rustup update stable     (or reinstall from https://rustup.rs)']
            )
        ) from None
    if result.returncode != 0:
        raise classify_version_failure(result.stderr + result.stdout)
    _version_cache[rustc] = result.stdout
    return result.stdout


def check_rust():
    """Describe the Rust toolchain without compiling anything.

    Returns {'ok': bool, 'rustc': path or None, 'version': str or None,
             'problem': str or None, 'fix': list of commands}.
    """
    rustc = find_rustc()
    if rustc is None:
        return {
            'ok': False,
            'rustc': None,
            'version': None,
            'problem': 'Rust compiler (rustc) not found.',
            'fix': install_steps(),
        }
    try:
        version = rustc_version(rustc)
    except ToolchainError as exc:
        lines = str(exc).splitlines()
        fix = [line.strip() for line in lines[3:-1]]
        return {'ok': False, 'rustc': rustc, 'version': None, 'problem': lines[0], 'fix': fix}
    release = next(
        (line.split(':', 1)[1].strip() for line in version.splitlines() if line.startswith('release:')),
        version.splitlines()[0] if version else '?',
    )
    return {'ok': True, 'rustc': rustc, 'version': release, 'problem': None, 'fix': []}


def doctor(stream=None):
    """Check everything PyRust needs, including one real compile. Returns True when native code works."""
    import tempfile

    out = stream or sys.stdout
    say = lambda text='': print(text, file=out)
    ok = True
    say('PyRust Engine doctor')
    say('====================')
    from . import __version__

    say(f'PyRust Engine : {__version__}')
    python_ok = sys.version_info >= (3, 10)
    say(
        f'Python        : {platform.python_version()} ({sys.executable})'
        + ('' if python_ok else '  <- needs 3.10 or newer')
    )
    ok &= python_ok
    say(f'Platform      : {platform.system()} {platform.machine()}')
    status = check_rust()
    if status['ok']:
        say(f'Rust          : {status["version"]} ({status["rustc"]})')
    else:
        ok = False
        say(f'Rust          : PROBLEM - {status["problem"]}')
        for step in status['fix']:
            say(f'                fix: {step}')
    try:
        import numpy

        say(f'NumPy         : {numpy.__version__} (F64Array / I64Array available)')
    except ImportError:
        say('NumPy         : not installed (only needed for arrays: pip install "pyrust-engine[arrays]")')
    if status['ok']:
        from .runtime import optimize

        def doctor_probe(n: int) -> int:
            total = 0
            for i in range(n):
                total += i
            return total

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as cache:
            probe = optimize(doctor_probe, cache_dir=cache, warn_on_fallback=False)
            try:
                works = probe(1000) == 499500 and probe.report()['backend'] == 'rust'
            except Exception:
                works = False
            if works:
                say(f'Test compile  : OK ({probe.report()["compile_seconds"]:.2f} s)')
            else:
                ok = False
                say('Test compile  : FAILED')
                for line in (probe.report()['compile_error'] or 'unknown error').splitlines():
                    say(f'                {line}')
    say()
    say(
        'Everything is ready: decorated functions will run as native code.'
        if ok
        else 'Native code is not available yet. Your code still runs correctly as plain Python.'
    )
    return ok
