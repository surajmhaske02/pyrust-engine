"""Typed native dispatch, explicit preparation/retry, and synchronized metrics."""
import ast
import builtins
from collections import Counter
import ctypes
import functools
import hashlib
import inspect
import platform
import shutil
import subprocess
import textwrap
import threading
import time
import types
from .compiler import Compiler, Unsupported
from .cache import cache_root, process_lock, cached_path, compile_library

_BUILTINS = {n:getattr(builtins,n) for n in ('range','len','float')}
_STATUS = {1:'integer overflow',2:'zero divisor or zero range step',3:'array index out of bounds',
           4:'native iteration budget exceeded',5:'no scalar return'}

class OptimizationError(RuntimeError):
    pass

LOOP_FREE_REASON=('loop-free function: a native call costs more than the work it saves, '
                  'so it runs as plain Python (pass native_loop_free=True to force Rust)')

def _is_loop_free(fn):
    """True when the function body has no for/while loop.

    Such a function does a fixed, tiny amount of work per call, so the ~2 us ctypes
    dispatch can never pay for itself. Unknown source is treated as not loop-free.
    """
    try:
        tree=ast.parse(textwrap.dedent(inspect.getsource(fn)))
    except (OSError,TypeError,SyntaxError,IndentationError):
        return False
    if not tree.body or not isinstance(tree.body[0],(ast.FunctionDef,ast.AsyncFunctionDef)):
        return False
    return not any(isinstance(node,(ast.For,ast.AsyncFor,ast.While)) for node in ast.walk(tree.body[0]))

def _plain_python(fn,max_iterations):
    """Zero-overhead result for loop-free functions: a real Python function, not a wrapper.

    Calls cost exactly what the original costs. Call counters are not tracked because
    there is no wrapper to count them.
    """
    clone=types.FunctionType(fn.__code__,fn.__globals__,fn.__name__,fn.__defaults__,fn.__closure__)
    clone.__kwdefaults__=dict(fn.__kwdefaults__) if fn.__kwdefaults__ else None
    functools.update_wrapper(clone,fn)

    def report():
        return {'backend':'python','reason':LOOP_FREE_REASON,'compile_error':None,'native_calls':0,
                'fallback_calls':0,'calls':0,'cache_hit':False,'cache_rebuilt':False,'compile_seconds':0.0,
                'native_seconds':0.0,'fallback_seconds':0.0,'wrapper_seconds':0.0,'compile_attempts':0,
                'fallback_reasons':{},'fallback_rate':0.0,'native_average_seconds':None,
                'fallback_average_seconds':None,'dispatch':'python-direct','call_counting':False}

    def prepare(*,background=False):
        return report()

    clone.python=fn;clone.report=report;clone.prepare=prepare;clone.reset=report
    clone.rust_source=lambda:Compiler(fn,max_iterations).generate()
    return clone

def optimize(function=None, *, backend='rust', fallback=True, cache_dir=None, max_iterations=10_000_000,
             native_loop_free=False):
    """Compile a numeric Python function to Rust, falling back to Python when needed.

    native_loop_free: loop-free functions (no for/while) run as plain Python by default,
    because native dispatch overhead (~2 us) is larger than their work. Set True to
    compile them anyway (for example to test the native path).
    """
    if backend!='rust':
        raise ValueError('Only backend="rust" is supported')
    if type(max_iterations) is not int or not 1<=max_iterations<2**64:
        raise ValueError('max_iterations must be a positive unsigned 64-bit integer')
    def decorate(fn):
        if not isinstance(fn,types.FunctionType):
            raise TypeError('Decorate an ordinary Python function')
        if not native_loop_free and _is_loop_free(fn):
            return _plain_python(fn,max_iterations)
        signature=inspect.signature(fn)
        original=(fn.__code__,fn.__defaults__,fn.__kwdefaults__)
        compile_lock=threading.RLock();metrics_lock=threading.Lock()
        state={'backend':'pending','reason':None,'compile_error':None,'native_calls':0,'fallback_calls':0,
               'calls':0,'cache_hit':False,'cache_rebuilt':False,'compile_seconds':0.0,'native_seconds':0.0,
               'fallback_seconds':0.0,'wrapper_seconds':0.0,'compile_attempts':0}
        reasons=Counter()
        loaded=None;model=None;source=None;attempted=False;worker=None
        # (lib, entry, output_type, kinds, used_builtins) published as one tuple so the
        # hot path can read it without taking compile_lock.
        snapshot=None
        param_names=tuple(signature.parameters)
        n_params=len(param_names)
        fn_globals=fn.__globals__
        fn_builtins=fn.__builtins__ if isinstance(fn.__builtins__,dict) else vars(fn.__builtins__)

        def update(**values):
            with metrics_lock:
                state.update(values)

        def report():
            with metrics_lock:
                data=dict(state);data['fallback_reasons']=dict(reasons)
            calls=data['calls']
            data['fallback_rate']=data['fallback_calls']/calls if calls else 0.0
            data['native_average_seconds']=data['native_seconds']/data['native_calls'] if data['native_calls'] else None
            data['fallback_average_seconds']=data['fallback_seconds']/data['fallback_calls'] if data['fallback_calls'] else None
            data['dispatch']='wrapper';data['call_counting']=True
            return data

        def python_call(reason,args,kwargs):
            with metrics_lock:
                state['reason']=reason
            if not fallback:
                raise OptimizationError(reason)
            start=time.perf_counter()
            try:
                return fn(*args,**kwargs)
            finally:
                with metrics_lock:
                    state['fallback_calls']+=1
                    state['fallback_seconds']+=time.perf_counter()-start
                    reasons[reason]+=1

        def bind_library(path,description):
            lib=ctypes.CDLL(str(path))
            entry=lib.pyrust_run
            argtypes=[]
            for p in signature.parameters:
                kind=description.types[p]
                argtypes.extend([ctypes.POINTER(ctypes.c_double),ctypes.c_size_t] if kind=='array' else
                                [ctypes.c_double if kind=='f64' else ctypes.c_int64])
            output=ctypes.c_double if description.return_type=='f64' else ctypes.c_int64
            entry.argtypes=argtypes+[ctypes.POINTER(output)]
            entry.restype=ctypes.c_int32
            return lib,entry,output

        def compile_once():
            nonlocal loaded,model,source,attempted,snapshot
            with compile_lock:
                if attempted:
                    return
                attempted=True
                start=time.perf_counter()
                with metrics_lock:
                    state['compile_attempts']+=1
                    state['backend']='compiling'
                try:
                    description=Compiler(fn,max_iterations)
                    source=description.generate()
                    compiler=shutil.which('rustc')
                    if not compiler:
                        raise Unsupported('rustc is not installed or not on PATH')
                    version=subprocess.check_output([compiler,'-vV'],text=True,timeout=15)
                    key=hashlib.sha256((source+version+platform.platform()+'abi3-O3-ed2021').encode()).hexdigest()
                    root=cache_root(cache_dir)
                    with process_lock(root/(key+'.lock')):
                        path=cached_path(root,key)
                        hit=path is not None
                        rebuilt=(root/(key+'.json')).exists() and not hit
                        if path is None:
                            path=compile_library(root,key,compiler,source)
                        try:
                            candidate=bind_library(path,description)
                        except (OSError,AttributeError):
                            if not hit:
                                raise
                            # A valid hash does not guarantee loadability or a valid ABI symbol.
                            path=compile_library(root,key,compiler,source)
                            candidate=bind_library(path,description)
                            rebuilt=True;hit=False
                    model=description
                    loaded=candidate
                    snapshot=(candidate[0],candidate[1],candidate[2],
                              tuple(description.types[p] for p in param_names),
                              tuple(sorted(description.used_builtins)))
                    update(backend='rust',reason=None,compile_error=None,cache_hit=hit,cache_rebuilt=rebuilt)
                except Exception as exc:  # any compiler/loader failure means Python fallback
                    loaded=None;snapshot=None
                    update(backend='python',reason=str(exc),compile_error=str(exc))
                finally:
                    with metrics_lock:
                        state['compile_seconds']+=time.perf_counter()-start

        def prepare(*,background=False):
            nonlocal worker
            if background:
                with compile_lock:
                    if not attempted and (worker is None or not worker.is_alive()):
                        worker=threading.Thread(target=compile_once,name='pyrust-compile',daemon=True)
                        worker.start()
                return report()
            compile_once()
            if loaded is None and not fallback:
                raise OptimizationError(report()['compile_error'])
            return report()

        def reset():
            nonlocal loaded,model,source,attempted,worker,snapshot
            # Finish background preparation before clearing compilation state.
            active=worker
            if active is not None and active.is_alive():
                active.join()
            with compile_lock:
                loaded=None;model=None;source=None;attempted=False;worker=None;snapshot=None
                update(backend='pending',reason=None,compile_error=None,cache_hit=False,cache_rebuilt=False)
            return report()

        def changed():
            return any(a is not b for a,b in zip(original,(fn.__code__,fn.__defaults__,fn.__kwdefaults__)))

        def convert(values,kinds):
            args=[]
            for name,kind,value in zip(param_names,kinds,values):
                if kind=='i64':
                    if type(value) is not int or not -(2**63)<=value<2**63:
                        raise Unsupported(f'{name}: expected exact signed 64-bit int')
                    args.append(value)
                elif kind=='f64':
                    if type(value) is not float:
                        raise Unsupported(f'{name}: expected exact Python float')
                    args.append(value)
                else:
                    try:
                        import numpy as np
                    except ImportError:
                        raise Unsupported('NumPy is required for F64Array') from None
                    if (type(value) is not np.ndarray or value.dtype!=np.dtype('float64') or
                        not value.dtype.isnative or value.ndim!=1 or not value.flags.c_contiguous or
                        not value.flags.aligned or value.size>2**63-1):
                        raise Unsupported(f'{name}: expected aligned contiguous native float64 1-D ndarray')
                    args.extend([value.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),value.size])
            return args

        perf_counter=time.perf_counter

        @functools.wraps(fn)
        def wrapped(*args,**kwargs):
            start=perf_counter()
            native_elapsed=None
            try:
                if (fn.__code__ is not original[0] or fn.__defaults__ is not original[1]
                        or fn.__kwdefaults__ is not original[2]):
                    return python_call('function code or defaults changed; create a new wrapper',args,kwargs)
                if kwargs or len(args)!=n_params:
                    # Slow path for keywords/defaults; raises CPython's TypeError for bad calls.
                    bound=signature.bind(*args,**kwargs);bound.apply_defaults()
                    values=tuple(bound.arguments.values())
                else:
                    values=args  # fast path: exact positional call
                active=snapshot
                if active is None:
                    if worker is not None and worker.is_alive():
                        return python_call('background compilation in progress',args,kwargs)
                    compile_once()
                    active=snapshot
                    if active is None:
                        return python_call(report()['compile_error'] or 'native backend unavailable',args,kwargs)
                # Holding `active` keeps the library loaded for the duration of the call.
                lib,entry,output,kinds,used=active
                for name in used:
                    if fn_globals.get(name,fn_builtins.get(name)) is not _BUILTINS[name]:
                        return python_call(f'{name} has been rebound',args,kwargs)
                try:
                    native_args=convert(values,kinds)
                except Unsupported as exc:
                    return python_call(str(exc),args,kwargs)
                out=output()
                t0=perf_counter()
                status=entry(*native_args,ctypes.byref(out))
                elapsed=perf_counter()-t0
                if status:
                    with metrics_lock:
                        state['native_calls']+=1
                        state['native_seconds']+=elapsed
                    return python_call(_STATUS.get(status,'unknown native guard failure'),args,kwargs)
                native_elapsed=elapsed
                return out.value
            finally:
                end=perf_counter()
                # One lock acquisition per call for all counters.
                with metrics_lock:
                    state['calls']+=1
                    state['wrapper_seconds']+=end-start
                    if native_elapsed is not None:
                        state['native_calls']+=1
                        state['native_seconds']+=native_elapsed
                        state['reason']=None

        wrapped.python=fn;wrapped.report=report;wrapped.prepare=prepare;wrapped.reset=reset
        wrapped.rust_source=lambda:source if source is not None else Compiler(fn,max_iterations).generate()
        return wrapped
    return decorate(function) if function is not None else decorate
