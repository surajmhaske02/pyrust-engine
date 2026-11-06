"""Profile a trusted script once and report its top-level native candidates."""
import contextlib
import cProfile
import inspect
from pathlib import Path
import pstats
import runpy
import sys
from .compiler import Compiler

def profile_script(path, arguments=(), limit=20):
    path=Path(path).resolve()
    profiler=cProfile.Profile()
    old_argv=sys.argv[:];old_path=sys.path[:]
    try:
        sys.argv=[str(path),*arguments]
        sys.path.insert(0,str(path.parent))
        with contextlib.redirect_stdout(sys.stderr):
            namespace=profiler.runcall(runpy.run_path,str(path),run_name='__main__')
    finally:
        sys.argv=old_argv;sys.path[:]=old_path
    candidates={}
    for value in namespace.values():
        if not inspect.isfunction(value):
            continue
        value=getattr(value,'python',value)
        if Path(value.__code__.co_filename).resolve()!=path:
            continue
        key=(value.__code__.co_filename,value.__code__.co_firstlineno,value.__name__)
        try:
            compiler=Compiler(value);compiler.generate()
            eligibility={'supported':True,'reason':None,'parameter_types':
                         {p:compiler.types[p] for p in compiler.signature.parameters}}
        except (ValueError,TypeError,OSError,SyntaxError) as exc:
            eligibility={'supported':False,'reason':str(exc)}
        candidates[key]=eligibility
    rows=[]
    for (filename,line,name),(primitive,calls,self_time,cumulative,callers) in pstats.Stats(profiler).stats.items():
        if filename=='~' or Path(filename).resolve()!=path or name=='<module>':
            continue
        rows.append({'function':name,'line':line,'calls':calls,'self_seconds':self_time,
                     'cumulative_seconds':cumulative,
                     **candidates.get((filename,line,name),{'supported':False,'reason':'Not a retained top-level function'})})
    return {'script':str(path),'scope':'functions in entry script; no automatic source rewriting',
            'hotspots':sorted(rows,key=lambda r:r['self_seconds'],reverse=True)[:limit]}
