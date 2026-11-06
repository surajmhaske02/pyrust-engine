"""Private native cache with OS process locks, hashes and atomic publication."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

SUFFIX = '.dll' if sys.platform=='win32' else '.dylib' if sys.platform=='darwin' else '.so'

def cache_root(path=None):
    root = Path(path) if path else Path.home()/'.cache'/'pyrust_engine'
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    root=root.resolve()
    if os.name=='posix':
        info=root.stat()
        if info.st_uid!=os.getuid() or info.st_mode & 0o022:
            raise PermissionError('Native cache must be owned by this user and not group/world writable')
    return root

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

@contextmanager
def process_lock(path, timeout=150):
    # Advisory OS locks are released automatically if the process dies.
    f=open(path,'a+b')
    try:
        f.seek(0,2)
        if f.tell()==0:
            f.write(b'0');f.flush()
        start=time.monotonic()
        while True:
            try:
                f.seek(0)
                if sys.platform=='win32':
                    import msvcrt
                    msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
                else:
                    import fcntl
                    fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                break
            except (BlockingIOError,OSError):
                if time.monotonic()-start >= timeout:
                    raise TimeoutError('Timed out waiting for native cache lock')
                time.sleep(0.02)
        try:
            yield
        finally:
            if sys.platform=='win32':
                import msvcrt
                f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(f.fileno(),fcntl.LOCK_UN)
    finally:
        f.close()

def cached_path(root,key):
    try:
        meta=json.loads((root/(key+'.json')).read_text())
        sha=meta['sha256']
        if not isinstance(sha,str) or not re.fullmatch('[0-9a-f]{64}',sha):
            return None
        path=root/(key+'-'+sha+SUFFIX)
        return path if path.is_file() and digest(path)==sha else None
    except (OSError,ValueError,KeyError,TypeError):
        return None

def compile_library(root,key,compiler,source):
    with tempfile.TemporaryDirectory(dir=root,prefix='build-') as work:
        rs=Path(work)/'kernel.rs';binary=Path(work)/('kernel'+SUFFIX)
        rs.write_text(source,encoding='utf-8')
        result=subprocess.run([compiler,str(rs),'--crate-type','cdylib','--edition','2021',
                               '-C','opt-level=3','-C','panic=abort','-o',str(binary)],
                              capture_output=True,text=True,timeout=120)
        if result.returncode:
            raise RuntimeError('Rust compilation failed: '+result.stderr[-3000:])
        sha=digest(binary)
        path=root/(key+'-'+sha+SUFFIX)
        # Never truncate a library another process could have mapped.
        if not path.exists() or digest(path)!=sha:
            os.replace(binary,path)
        meta=Path(work)/'metadata.json'
        meta.write_text(json.dumps({'sha256':sha,'created':time.time()}))
        os.replace(meta,root/(key+'.json'))
        return path

def prune_cache(path=None, *, older_than_days=30, dry_run=True):
    """List/remove old cache entries. Defaults to a dry run. Lock files are retained."""
    if older_than_days < 0:
        raise ValueError('older_than_days must be nonnegative')
    root=cache_root(path);entries=[];errors=[]
    for meta in root.glob('*.json'):
        key=meta.stem
        if not re.fullmatch('[0-9a-f]{64}',key):
            continue
        with process_lock(root/(key+'.lock')):
            if not meta.exists() or time.time()-meta.stat().st_mtime < older_than_days*86400:
                continue
            for file in [*root.glob(key+'-*'+SUFFIX),meta]:
                entries.append(str(file))
                if not dry_run:
                    try:
                        file.unlink(missing_ok=True)
                    except OSError as exc:
                        errors.append(str(exc))
    return {'dry_run':dry_run,'entries':entries,'errors':errors}
