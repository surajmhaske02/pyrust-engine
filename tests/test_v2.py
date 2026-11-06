import concurrent.futures
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from pyrust_engine import OptimizationError, prune_cache
from tests.support import optimize
from pyrust_engine.profiling import profile_script
from tests import kernels as k

RUST=bool(shutil.which('rustc'))
NUMPY=importlib.util.find_spec('numpy') is not None

class Diagnostics(unittest.TestCase):
    def test_retry_after_missing_compiler(self):
        fn=optimize(k.add)
        with patch('shutil.which',return_value=None):
            self.assertEqual(fn(1,2),3)
        self.assertIn('rustc',fn.report()['compile_error'])
        fn.reset()
        if RUST:
            fn.prepare()
            self.assertEqual(fn.report()['backend'],'rust')
            self.assertEqual(fn.report()['compile_attempts'],2)

    def test_profile_and_cli(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            path=Path(d)/'work.py'
            path.write_text('def work(n):\n    s=0\n    for i in range(n):\n        s+=i\n    return s\nwork(1000)\n')
            report=profile_script(path)
            row=next(r for r in report['hotspots'] if r['function']=='work')
            self.assertTrue(row['supported'])
            self.assertEqual(row['calls'],1)
            result=subprocess.run([sys.executable,'-m','pyrust_engine','profile',str(path)],capture_output=True,text=True,check=True)
            self.assertEqual(json.loads(result.stdout)['hotspots'][0]['function'],'work')

    def test_bad_configuration(self):
        with self.assertRaises(ValueError):
            optimize(k.add,max_iterations=0)
        with self.assertRaises(TypeError):
            optimize(len)

@unittest.skipUnless(RUST,'rustc required')
class Native(unittest.TestCase):
    def test_5000_integer_cases(self):
        rng=random.Random(91)
        fn=optimize(k.divmod_mix,fallback=False)
        for _ in range(5000):
            a=rng.randint(-10**8,10**8);b=rng.choice([-999,-17,-1,1,7,1000])
            self.assertEqual(fn(a,b),k.divmod_mix(a,b))
        self.assertEqual(fn.report()['fallback_calls'],0)

    def test_stepped_while_and_nested_cases(self):
        rng=random.Random(32)
        stepfn=optimize(k.stepped,fallback=False)
        others=[optimize(f,fallback=False) for f in (k.while_sum,k.annotated,k.nested_control)]
        for _ in range(1000):
            a=rng.randint(-30,30);b=rng.randint(-30,30);step=rng.choice([-9,-2,-1,1,2,7])
            self.assertEqual(stepfn(a,b,step),k.stepped(a,b,step))
            for fn in others:
                self.assertEqual(fn(b),fn.python(b))

    def test_2000_float_cases(self):
        rng=random.Random(6);fn=optimize(k.float_math,fallback=False)
        for _ in range(2000):
            a=rng.uniform(-1e4,1e4);b=rng.uniform(1,1e4)
            self.assertEqual(fn(a,b),k.float_math(a,b))
        for a,b in [(float('nan'),1.0),(float('inf'),2.0),(-0.0,1.0),(1e308,2.0),(1e-300,1e300)]:
            x,y=fn(a,b),k.float_math(a,b)
            if math.isnan(y):
                self.assertTrue(math.isnan(x))
            else:
                self.assertEqual(x,y)
                if y==0.0:
                    self.assertEqual(math.copysign(1,x),math.copysign(1,y))

    def test_exception_and_boundary_semantics(self):
        for fn,args,error in [(k.divmod_mix,(1,0),ZeroDivisionError),
                              (k.stepped,(1,2,0),ValueError),(k.float_math,(1.0,0.0),ZeroDivisionError)]:
            wrapped=optimize(fn)
            with self.assertRaises(error):
                wrapped(*args)
            self.assertEqual(wrapped.report()['backend'],'rust')
        fn=optimize(k.divmod_mix)
        for a,b in [(-2**63,-1),(-2**63,1),(2**63-1,-3),(-2**63,-2**63)]:
            self.assertEqual(fn(a,b),k.divmod_mix(a,b))
        fn=optimize(k.min_literal,fallback=False)
        self.assertEqual(fn(0),-2**63)

    def test_boolean_conditions(self):
        fn=optimize(k.bool_logic,fallback=False)
        for a in (-2,0,2):
            for b in (-3,0,15):
                self.assertEqual(fn(a,b),k.bool_logic(a,b))

    def test_budget_fallback(self):
        fn=optimize(k.annotated,max_iterations=10)
        self.assertEqual(fn(100),k.annotated(100))
        self.assertIn('budget',fn.report()['reason'])
        with self.assertRaises(OptimizationError):
            optimize(k.annotated,max_iterations=10,fallback=False)(100)

    def test_float_and_int_guards(self):
        fn=optimize(k.float_math)
        self.assertEqual(fn(2,3),k.float_math(2,3))
        self.assertEqual(fn.report()['fallback_calls'],1)
        fn=optimize(k.add)
        self.assertEqual(fn(True,2),3)
        self.assertEqual(fn(2**90,1),2**90+1)

    def test_background_compilation(self):
        import pyrust_engine.runtime as runtime
        started=threading.Event();release=threading.Event();original=runtime.compile_library
        def delayed(*args):
            started.set()
            if not release.wait(5):
                raise RuntimeError('test release timed out')
            return original(*args)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            fn=optimize(k.add,cache_dir=d)
            with patch.object(runtime,'compile_library',side_effect=delayed):
                fn.prepare(background=True)
                self.assertTrue(started.wait(5))
                try:
                    self.assertEqual(fn(2,3),5)
                    self.assertEqual(fn.report()['reason'],'background compilation in progress')
                finally:
                    release.set()
                fn.prepare()
            self.assertEqual(fn.report()['backend'],'rust')
            self.assertEqual(fn(4,5),9)

    def test_threaded_counters(self):
        fn=optimize(k.add,fallback=False);fn.prepare()
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            values=list(pool.map(lambda i:fn(i,1),range(400)))
        self.assertEqual(values,list(range(1,401)))
        self.assertEqual(fn.report()['native_calls'],400)
        self.assertEqual(fn.report()['calls'],400)

    def test_defaults_mutation(self):
        def add_default(n,b=2):
            return n+b
        fn=optimize(add_default);self.assertEqual(fn(1),3)
        add_default.__defaults__=(9,)
        self.assertEqual(fn(1),10)
        self.assertIn('defaults changed',fn.report()['reason'])

    def test_corrupt_cache_rebuild(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            fn=optimize(k.add,cache_dir=d,fallback=False);fn.prepare()
            meta=next(Path(d).glob('*.json'))
            # Damage the metadata, never bytes mapped by a running process.
            meta.write_text('{corrupted')
            again=optimize(k.add,cache_dir=d,fallback=False)
            self.assertEqual(again(1,2),3)
            self.assertTrue(again.report()['cache_rebuilt'])
            listing=prune_cache(d,older_than_days=0)
            self.assertTrue(listing['entries']);self.assertTrue(meta.exists())
            cleanup=prune_cache(d,older_than_days=0,dry_run=False)
            self.assertFalse(meta.exists())
            self.assertEqual(fn(1,3),4)

    def test_missing_symbol_falls_back_and_can_retry(self):
        with patch('ctypes.CDLL',return_value=object()):
            fn=optimize(k.add)
            self.assertEqual(fn(1,2),3)
            self.assertEqual(fn.report()['backend'],'python')
            self.assertIn('pyrust_run',fn.report()['compile_error'])
        fn.reset();fn.prepare()
        self.assertEqual(fn.report()['backend'],'rust')

    def test_compilation_failures(self):
        for error in (subprocess.TimeoutExpired('rustc',120), RuntimeError('compiler rejected source'), OSError('linker missing')):
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
                fn=optimize(k.add,cache_dir=d)
                with patch('pyrust_engine.runtime.compile_library',side_effect=error):
                    self.assertEqual(fn(3,4),7)
                self.assertEqual(fn.report()['backend'],'python')
                self.assertIsNotNone(fn.report()['compile_error'])

    @unittest.skipIf(sys.platform=='win32','Windows cannot replace a DLL that is loaded in this process')
    def test_corrupt_binary_rebuild(self):
        from pyrust_engine.cache import digest, SUFFIX
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            fn=optimize(k.add,cache_dir=d,fallback=False);fn.prepare()
            library=next(Path(d).glob('*'+SUFFIX))
            # Atomic replacement leaves existing mapped bytes intact.
            bad=Path(d)/'bad';bad.write_bytes(b'invalid shared library')
            os.replace(bad,library)
            again=optimize(k.add,cache_dir=d,fallback=False)
            self.assertEqual(again(5,6),11)
            self.assertTrue(again.report()['cache_rebuilt'])
            meta=json.loads(next(Path(d).glob('*.json')).read_text())
            # Builds are not byte-identical on every platform (macOS embeds a build UUID),
            # so the rebuilt library may have a new content-addressed name.
            rebuilt=library.with_name(library.name.split('-')[0]+'-'+meta['sha256']+SUFFIX)
            self.assertEqual(digest(rebuilt),meta['sha256'])

    def test_process_cache_contention(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            script='from tests.kernels import add; from pyrust_engine import optimize; import json,sys; f=optimize(add,cache_dir=sys.argv[1],fallback=False,native_loop_free=True); assert f(2,3)==5; print(json.dumps(f.report()))'
            children=[subprocess.Popen([sys.executable,'-c',script,d],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for _ in range(4)]
            reports=[]
            for child in children:
                out,err=child.communicate(timeout=60)
                self.assertEqual(child.returncode,0,err)
                reports.append(json.loads(out))
            self.assertEqual(sum(not r['cache_hit'] for r in reports),1)

@unittest.skipUnless(RUST and NUMPY,'rustc and NumPy required')
class Arrays(unittest.TestCase):
    def test_1000_array_cases(self):
        import numpy as np
        rng=np.random.default_rng(5);fn=optimize(k.sum_squares,fallback=False)
        for _ in range(1000):
            a=rng.standard_normal(int(rng.integers(0,100)))
            before=a.copy()
            self.assertEqual(fn(a),k.sum_squares(a))
            np.testing.assert_array_equal(a,before)
        self.assertEqual(fn.report()['fallback_calls'],0)

    def test_bounds_and_negative_indices(self):
        import numpy as np
        a=np.arange(5,dtype=np.float64);fn=optimize(k.array_at)
        for i in range(-5,5):
            self.assertEqual(fn(a,i),float(a[i]))
        for a,i in [(a,5),(a,-6),(np.array([],dtype=np.float64),0)]:
            with self.assertRaises(IndexError):
                fn(a,i)
            self.assertIn('out of bounds',fn.report()['reason'])

    def test_unaligned_and_subclass_fallback(self):
        import numpy as np
        raw=bytearray(8*5+1)
        unaligned=np.ndarray((5,),dtype=np.float64,buffer=raw,offset=1)
        unaligned[:]=np.arange(5)
        class CustomArray(np.ndarray):
            pass
        fn=optimize(k.sum_squares)
        for a in (unaligned,np.arange(5,dtype=np.float64).view(CustomArray)):
            self.assertEqual(fn(a),k.sum_squares(a))
            self.assertIn('float64',fn.report()['reason'])

    def test_layout_dtype_and_readonly(self):
        import numpy as np
        fn=optimize(k.sum_squares)
        base=np.arange(20,dtype=np.float64)
        for a in (base[::2],base[::-1],base.astype(np.float32),base.astype('>f8')):
            self.assertEqual(fn(a),k.sum_squares(a))
            self.assertIn('float64',fn.report()['reason'])
        base.setflags(write=False)
        self.assertEqual(fn(base),k.sum_squares(base))
        self.assertIsNone(fn.report()['reason'])

if __name__=='__main__':
    unittest.main()
