"""Reproducible warm/cold, end-to-end and optional Numba/NumPy comparisons.

Run: python -m examples.benchmark_suite --output benchmarks/results.json
"""
import argparse
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile
import time
import random
import numpy as np
from pyrust_engine import optimize, vectorize
from .workloads import integer_work,float_work,array_energy,scalar_add,math_work,calls_work,bits_work,fib,smooth
from .workloads import edit_distance,list_variance,monte_carlo_pi,price_score

def measure(fn,args,repeat=7,batch=1):
    samples=[]
    for _ in range(repeat):
        start=time.perf_counter()
        for _ in range(batch):
            fn(*args)
        samples.append((time.perf_counter()-start)/batch)
    return {'median_seconds':statistics.median(samples),'min_seconds':min(samples),'samples_seconds':samples}

def same(a,b):
    if a is None:
        assert b is None,b
    elif type(a) is int:
        assert a==b,(a,b)
    else:
        assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),(a,b)

def run():
    try:
        import numba
    except ImportError:
        numba=None
    rng=np.random.default_rng(73)
    array=rng.standard_normal(100_000)
    jobs=[('integer_loop',integer_work,(500_000,),1,{}),
          ('float_recurrence',float_work,(300_000,1.0),1,{}),
          ('array_energy',array_energy,(array,),1,{}),
          ('math_loop',math_work,(300_000,),1,{}),
          ('function_calls',calls_work,(300_000,),1,{}),
          ('bit_hash',bits_work,(300_000,),1,{}),
          ('recursion_fib',fib,(25,),1,{}),
          ('array_write',smooth,(array,np.empty_like(array)),1,{}),
          ('edit_distance_strings',edit_distance,('kitten sitting on a mat '*12,'sitting kitten in a hat '*12),1,{}),
          ('list_variance',list_variance,(array.tolist(),),1,{}),
          # Default: loop-free functions run as plain Python (no dispatch overhead).
          ('tiny_scalar',scalar_add,(12,19),1000,{}),
          # Forced native: shows the raw ctypes dispatch cost per call.
          ('tiny_scalar_forced_native',scalar_add,(12,19),1000,{'native_loop_free':True})]
    rows=[]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as cache:
        for name,fn,args,batch,options in jobs:
            expected=fn(*args)
            native=optimize(fn,cache_dir=cache,fallback=False,**options)
            start=time.perf_counter();actual=native(*args);first=time.perf_counter()-start
            same(expected,actual)
            py=measure(fn,args,batch=batch)
            rust=measure(native,args,batch=batch)
            row={'name':name,'python':py,'pyrust':rust,'pyrust_first_call_seconds':first,
                 'python_over_pyrust':py['median_seconds']/rust['median_seconds'],'report':native.report()}
            again=optimize(fn,cache_dir=cache,fallback=False,**options)
            start=time.perf_counter();same(expected,again(*args))
            row['disk_cache_first_call_seconds']=time.perf_counter()-start
            if numba is not None and name!='tiny_scalar_forced_native':
                # fastmath disabled; no Python big-int semantics claimed for Numba.
                try:
                    jitted=numba.njit(fn,fastmath=False)
                    start=time.perf_counter();same(expected,jitted(*args))
                    row['numba_first_call_seconds']=time.perf_counter()-start
                    row['numba']=measure(jitted,args,batch=batch)
                    row['pyrust_over_numba_time']=rust['median_seconds']/row['numba']['median_seconds']
                except Exception as exc:  # e.g. Numba can't call a plain Python helper
                    row['numba_error']=str(exc).strip().splitlines()[0][:200]
            if name=='array_energy':
                vectorized=lambda a:np.sum(a*a)
                same(expected,vectorized(*args))
                row['numpy_vectorized']=measure(vectorized,args)
            rows.append(row)
        # random.random(): both runs start from the same seed and must give the same answer.
        native=optimize(monte_carlo_pi,cache_dir=cache,fallback=False)
        random.seed(5);expected=monte_carlo_pi(200_000);random.seed(5);same(expected,native(200_000))
        a=measure(monte_carlo_pi,(200_000,));b=measure(native,(200_000,))
        rows.append({'name':'random_monte_carlo','python':a,'pyrust':b,
                     'python_over_pyrust':a['median_seconds']/b['median_seconds'],'report':native.report()})
        # vectorize: a tiny function over 100k elements, versus a Python loop over the same arrays.
        prices=np.abs(array)*10;qty=rng.integers(0,30,len(array))
        def python_loop(p,q):
            return np.array([price_score(x,y) for x,y in zip(p.tolist(),q.tolist())])
        fast=vectorize(price_score,cache_dir=cache,fallback=False)
        assert python_loop(prices,qty).tobytes()==fast(prices,qty).tobytes()
        a=measure(python_loop,(prices,qty));b=measure(fast,(prices,qty))
        rows.append({'name':'vectorize_tiny_function','python':a,'pyrust':b,
                     'python_over_pyrust':a['median_seconds']/b['median_seconds'],'report':fast.report()})
        payload=json.dumps(rng.standard_normal(20_000).tolist())
        native=optimize(array_energy,cache_dir=cache,fallback=False);native.prepare()
        def pipeline(kernel):
            def run(payload):
                # Include real conversion, JSON parsing and output serialization each time.
                values=np.asarray(json.loads(payload),dtype=np.float64)
                result=kernel(values)
                return json.dumps({'energy':result,'count':len(values)})
            return run
        python_pipeline=pipeline(array_energy);native_pipeline=pipeline(native)
        assert python_pipeline(payload)==native_pipeline(payload)
        a=measure(python_pipeline,(payload,));b=measure(native_pipeline,(payload,))
        rows.append({'name':'json_array_pipeline','python':a,'pyrust':b,
                     'python_over_pyrust':a['median_seconds']/b['median_seconds']})
    return {'environment':{'python':platform.python_version(),'platform':platform.platform(),
                           'rust':subprocess.check_output(['rustc','--version'],text=True).strip(),
                           'numpy':np.__version__,'numba':numba.__version__ if numba else None},
            'method':'7 warm samples; median; wrapper/conversion costs included; cold startup separate; tiny scalar batched 1000 calls/sample',
            'limitations':'One machine; controlled workloads; not production load testing. No Cython/PyPy/Codon comparison.',
            'results':rows}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output');args=parser.parse_args()
    data=run();text=json.dumps(data,indent=2)
    if args.output:
        target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(text+'\n')
    print(text)
