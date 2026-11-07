"""Private native cache with OS process locks, hashes and atomic publication."""

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

SUFFIX = '.dll' if sys.platform == 'win32' else '.dylib' if sys.platform == 'darwin' else '.so'

_AUTO_PRUNED: set = set()


def cache_root(path=None):
    root = Path(path) if path else Path(os.environ.get('PYRUST_CACHE_DIR') or Path.home() / '.cache' / 'pyrust_engine')
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    root = root.resolve()
    if os.name == 'posix':
        info = root.stat()
        if info.st_uid != os.getuid() or info.st_mode & 0o022:
            raise PermissionError('Native cache must be owned by this user and not group/world writable')
    if root not in _AUTO_PRUNED:
        _AUTO_PRUNED.add(root)
        auto_prune(root)
    return root


def auto_prune(root):
    """Delete cache entries unused for PYRUST_CACHE_MAX_AGE_DAYS (default 30), at most once a day.

    Set PYRUST_CACHE_MAX_AGE_DAYS=0 to turn this off. Errors are ignored: a library that is
    in use elsewhere is simply kept."""
    try:
        days = float(os.environ.get('PYRUST_CACHE_MAX_AGE_DAYS', '30'))
    except ValueError:
        days = 30.0
    if days <= 0:
        return
    marker = root / '.last-prune'
    try:
        if marker.exists() and time.time() - marker.stat().st_mtime < 86400:
            return
        marker.touch()
        prune_cache(root, older_than_days=days, dry_run=False)
    except Exception:
        pass


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


@contextmanager
def process_lock(path, timeout=150):
    # Advisory OS locks are released automatically if the process dies.
    f = open(path, 'a+b')
    try:
        f.seek(0, 2)
        if f.tell() == 0:
            f.write(b'0')
            f.flush()
        start = time.monotonic()
        while True:
            try:
                f.seek(0)
                if sys.platform == 'win32':
                    import msvcrt

                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (BlockingIOError, OSError):
                if time.monotonic() - start >= timeout:
                    raise TimeoutError('Timed out waiting for native cache lock')
                time.sleep(0.02)
        try:
            yield
        finally:
            if sys.platform == 'win32':
                import msvcrt

                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    finally:
        f.close()


def cached_path(root, key):
    try:
        meta = json.loads((root / (key + '.json')).read_text())
        sha = meta['sha256']
        if not isinstance(sha, str) or not re.fullmatch('[0-9a-f]{64}', sha):
            return None
        path = root / (key + '-' + sha + SUFFIX)
        if path.is_file() and digest(path) == sha:
            try:
                os.utime(root / (key + '.json'))  # remember last use, so auto-prune keeps it
            except OSError:
                pass
            return path
        return None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def compile_library(root, key, compiler, source):
    with tempfile.TemporaryDirectory(dir=root, prefix='build-') as work:
        rs = Path(work) / 'kernel.rs'
        binary = Path(work) / ('kernel' + SUFFIX)
        rs.write_text(source, encoding='utf-8')
        result = subprocess.run(
            [compiler, str(rs), '--crate-type', 'cdylib', '--edition', '2021', '-C', 'opt-level=3', '-o', str(binary)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode:
            from .toolchain import classify_build_failure

            problem = classify_build_failure(result.stderr)
            if problem is not None:
                raise problem
            raise RuntimeError('Rust compilation failed: ' + result.stderr[-3000:])
        sha = digest(binary)
        path = root / (key + '-' + sha + SUFFIX)
        # Never truncate a library another process could have mapped.
        if not path.exists() or digest(path) != sha:
            os.replace(binary, path)
        meta = Path(work) / 'metadata.json'
        meta.write_text(json.dumps({'sha256': sha, 'created': time.time()}))
        os.replace(meta, root / (key + '.json'))
        return path


def prune_cache(path=None, *, older_than_days=30, dry_run=True):
    """List/remove old cache entries. Defaults to a dry run. Lock files are retained."""
    if older_than_days < 0:
        raise ValueError('older_than_days must be nonnegative')
    root = Path(path).resolve() if isinstance(path, Path) else cache_root(path)
    entries = []
    errors = []
    for meta in root.glob('*.json'):
        key = meta.stem
        if not re.fullmatch('[0-9a-f]{64}', key):
            continue
        with process_lock(root / (key + '.lock')):
            if not meta.exists() or time.time() - meta.stat().st_mtime < older_than_days * 86400:
                continue
            for file in [*root.glob(key + '-*' + SUFFIX), meta]:
                entries.append(str(file))
                if not dry_run:
                    try:
                        file.unlink(missing_ok=True)
                    except OSError as exc:
                        errors.append(str(exc))
    return {'dry_run': dry_run, 'entries': entries, 'errors': errors}
