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

def ai_rows(cache, rng):
    """0.7.2: float32, strings, dicts and the pyrust_engine.ai library, against plain Python and NumPy."""
    from pyrust_engine import ai
    from .workloads import word_count, dot32
    rows=[]
    def row(name, python_fn, fast_fn, args, extra=None):
        assert repr(python_fn(*args))==repr(fast_fn(*args)), name
        a=measure(python_fn,args);b=measure(fast_fn,args)
        out={'name':name,'python':a,'pyrust':b,'python_over_pyrust':a['median_seconds']/b['median_seconds']}
        out.update(extra or {})
        rows.append(out)
    text=' '.join(f'word{int(x)%5000} the of and' for x in rng.integers(0,10**6,40_000))
    row('dict_word_count',word_count,optimize(word_count,cache_dir=cache,fallback=False),(text,))
    row('str_tokenize_200kb',ai._tokenize,ai.tokenize,(text,))
    a='the quick brown fox jumps over the lazy dog '*12;b='the quick brown cat jumped over a lazy dog '*12
    row('ai_levenshtein_500_chars',ai._levenshtein,ai.levenshtein,(a,b))
    pairs=[(text[i:i+200],text[i+50:i+260]) for i in range(0,20_000,200)]
    f1_py=lambda ps:[ai._token_f1(p,r) for p,r in ps];f1_rs=lambda ps:[ai.token_f1(p,r) for p,r in ps]
    row('ai_token_f1_100_pairs',f1_py,f1_rs,(pairs,))
    v=rng.standard_normal(100_000).astype(np.float32);w=rng.standard_normal(100_000).astype(np.float32)
    row('float32_dot_100k',dot32,optimize(dot32,cache_dir=cache,fallback=False),(v,w))
    docs=[text[i:i+400] for i in range(0,len(text)-400,400)]
    index=ai.BM25(docs);query='word12 word99 the word4000'
    def bm25_python(q):
        tokens=[ai._tokenize(d) for d in docs] if not hasattr(bm25_python,'t') else bm25_python.t
        bm25_python.t=tokens
        n=len(tokens);avgdl=sum(map(len,tokens))/n;scores=[]
        df={}
        for d in tokens:
            for t in set(d): df[t]=df.get(t,0)+1
        for d in tokens:
            s=0.0;counts={}
            for t in d: counts[t]=counts.get(t,0)+1
            for t in ai._tokenize(q):
                if t in counts:
                    idf=math.log((n-df[t]+0.5)/(df[t]+0.5)+1.0);f=counts[t]
                    s+=idf*f*2.5/(f+1.5*(1-0.75+0.75*len(d)/avgdl))
            scores.append(s)
        return scores
    bm25_python(query)
    a_=measure(bm25_python,(query,));b_=measure(index.scores,(query,))
    rows.append({'name':'ai_bm25_query_%d_docs'%len(docs),'python':a_,'pyrust':b_,'python_over_pyrust':a_['median_seconds']/b_['median_seconds']})
    m=rng.standard_normal((50_000,384)).astype(np.float32);q=m[123]+0.01
    def numpy_topk(q,m):
        s=(m@q)/(np.linalg.norm(m,axis=1)*np.linalg.norm(q))
        return np.argsort(-s)[:10]
    ai.cosine_top_k(q,m,10)
    a_=measure(numpy_topk,(q,m));b_=measure(ai.cosine_top_k,(q,m,10))
    rows.append({'name':'ai_cosine_top10_50k_x_384_vs_numpy','python':a_,'pyrust':b_,
                 'python_over_pyrust':a_['median_seconds']/b_['median_seconds'],
                 'note':'baseline is vectorized NumPy (BLAS), not a Python loop'})
    return rows

def v073_rows(cache, rng):
    """0.7.3: BM25Okapi vs rank_bm25, binary/int8 search vs NumPy, batches, str columns."""
    from pyrust_engine import ai
    from .workloads import domain_length
    rows=[]
    def add(name, slow, fast, args=(), note=None, check=None):
        if check is not None:
            check()
        a=measure(slow,args);b=measure(fast,args)
        row={'name':name,'python':a,'pyrust':b,'python_over_pyrust':a['median_seconds']/b['median_seconds']}
        if note: row['note']=note
        rows.append(row)
    words=[f'w{i}' for i in range(3000)]
    corpus=[[words[int(x)] for x in rng.zipf(1.3,size=60)%3000] for _ in range(5000)]
    query=['w1','w17','w250','w2999']
    try:
        from rank_bm25 import BM25Okapi as Reference
    except ImportError:
        Reference=None
    if Reference is not None:
        ref=Reference(corpus);ours=ai.BM25Okapi(corpus)
        assert np.array_equal(ref.get_scores(query),ours.get_scores(query))
        add('bm25okapi_build_5k_docs_vs_rank_bm25',lambda:Reference(corpus),lambda:ai.BM25Okapi(corpus),
            note='baseline is the rank_bm25 package (pure Python)')
        add('bm25okapi_query_5k_docs_vs_rank_bm25',ref.get_scores,ours.get_scores,(query,),
            note='baseline is the rank_bm25 package; scores are identical')
    codes=rng.integers(0,256,(200_000,48),dtype=np.uint8);q=codes[77]^np.uint8(1)
    def numpy_hamming(q,m):
        d=np.bitwise_count(m^q).sum(axis=1,dtype=np.int64) if hasattr(np,'bitwise_count') else np.unpackbits(m^q,axis=1).sum(axis=1)
        return ai.top_k(-d,10)[0]
    add('hamming_top10_200k_x_384bit_vs_numpy',numpy_hamming,lambda q,m:ai.hamming_top_k(q,m,10)[0],(q,codes),
        note='baseline is NumPy (np.bitwise_count)',
        check=lambda:np.testing.assert_array_equal(numpy_hamming(q,codes),ai.hamming_top_k(q,codes,10)[0]))
    m8=rng.integers(-127,128,(100_000,384),dtype=np.int8);q8=m8[5]
    def numpy_int8(q,m):
        return ai.top_k(m.astype(np.int32)@q.astype(np.int32),10)[0]
    add('int8_top10_100k_x_384_vs_numpy',numpy_int8,lambda q,m:ai.int8_top_k(q,m,10)[0],(q8,m8),
        note='baseline is NumPy casting to int32 then matmul (int8 @ int8 would overflow)',
        check=lambda:np.testing.assert_array_equal(numpy_int8(q8,m8),ai.int8_top_k(q8,m8,10)[0]))
    texts=[' '.join(f'Token{int(x)} and, more!' for x in rng.integers(0,10**6,400)) for _ in range(2000)]
    one_by_one=lambda ts:[ai.tokenize(t) for t in ts]
    add('tokenize_many_2000_docs_threads_vs_one_by_one',one_by_one,ai.tokenize_many,(texts,),
        note='both compiled; shows the gain from running on several threads',
        check=lambda:assert_same(one_by_one(texts),ai.tokenize_many(texts)))
    one_by_one_chunks=lambda ts:[ai.chunk_text(t,500,50) for t in ts]
    add('chunk_many_2000_docs_groups_vs_one_by_one',one_by_one_chunks,lambda ts:ai.chunk_many(ts,500,50),(texts,),
        note='both compiled; 0.7.4 processes groups of documents per native call',
        check=lambda:assert_same(one_by_one_chunks(texts),ai.chunk_many(texts,500,50)))
    emails=[f'user{i}@{"example" if i%3 else "site"}.com' if i%7 else f'user{i}' for i in range(200_000)]
    fast=vectorize(domain_length,cache_dir=cache,fallback=False)
    try:
        import pandas as pd
        column=pd.Series(emails)
        add('vectorize_str_column_vs_pandas_map',lambda c:c.map(domain_length),fast,(column,),
            note='pandas Series of str (pandas 3 str dtype when installed)',
            check=lambda:assert_same(column.map(domain_length).tolist(),fast(column).tolist()))
    except ImportError:
        pass
    try:
        import polars as pl
        pcol=pl.Series('email',emails)
        slow=lambda c:c.map_elements(domain_length,return_dtype=pl.Int64)
        add('vectorize_str_column_vs_polars_map_elements',slow,fast,(pcol,),
            note='polars map_elements (Python function per row)',
            check=lambda:assert_same(slow(pcol).to_list(),fast(pcol).to_list()))
    except ImportError:
        pass
    return rows

def v075_rows(cache, rng):
    """0.7.5: methods with self.x, and direct calls with small arrays and strings."""
    from .workloads import Body, short_sum, vowel_count
    rows=[]
    def add(name, slow, fast, args, note, check):
        check()
        a=measure(slow,args);b=measure(fast,args)
        rows.append({'name':name,'python':a,'pyrust':b,'python_over_pyrust':a['median_seconds']/b['median_seconds'],
                     'note':note,'report':fast.report() if hasattr(fast,'report') else None})
    step=optimize(Body.advance,cache_dir=cache,fallback=False)
    def check_body():
        a,b=Body(0.0,1.0,0.5,-2.0),Body(0.0,1.0,0.5,-2.0)
        step(a,0.001,10_000);Body.advance(b,0.001,10_000);assert vars(a)==vars(b)
    body=Body(0.0,1.0,0.5,-2.0)
    add('method_self_attributes_20k_steps',Body.advance,step,(body,0.001,20_000),
        'a method reading and writing self.x / self.vx (state written back after the call)',check_body)
    small=rng.standard_normal(32);fast_sum=optimize(short_sum,cache_dir=cache,fallback=False)
    many=lambda f,a:[f(a) for _ in range(1000)]
    add('small_array_1000_calls',lambda a:many(short_sum,a),lambda a:many(fast_sum,a),(small,),
        '1000 calls on a 32-element array (direct call, no ctypes)',lambda:assert_same(short_sum(small),fast_sum(small)))
    word='programming language'
    fast_vowels=optimize(vowel_count,cache_dir=cache,fallback=False)
    add('short_string_1000_calls',lambda w:many(vowel_count,w),lambda w:many(fast_vowels,w),(word,),
        '1000 calls on a 20-character string (direct call, no ctypes)',lambda:assert_same(vowel_count(word),fast_vowels(word)))
    return rows

def assert_same(a,b):
    assert a==b

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
        rows += ai_rows(cache, rng)
        rows += v073_rows(cache, rng)
        rows += v075_rows(cache, rng)
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
