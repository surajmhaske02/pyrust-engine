"""0.7.0: bools, for-in / enumerate, new builtins, tuples, keyword/default arguments,
per-type specialization, 2-D and new arrays, undo-log writes, prange, long calls,
panic safety, prebuilt libraries, Jupyter magics and cache cleanup."""
import importlib.util
import io
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
import warnings

from pyrust_engine import optimize, check_rust, OptimizationError
from pyrust_engine.compiler import Compiler, Unsupported

NUMPY = importlib.util.find_spec('numpy') is not None
RUST = check_rust()['ok']
ROOT = Path(__file__).resolve().parent.parent
if NUMPY:
    import numpy as np
    from tests import kernels7 as k


def run(fn, args):
    try:
        return ('ok', fn(*args))
    except Exception as exc:
        return ('raise', type(exc).__name__)


def same_value(a, b):
    if isinstance(a, tuple) or isinstance(b, tuple):
        return isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b) and all(
            same_value(x, y) for x, y in zip(a, b))
    if NUMPY and (isinstance(a, np.ndarray) or isinstance(b, np.ndarray)):
        return (isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype
                and a.shape == b.shape and a.tobytes() == b.tobytes())
    if isinstance(a, bool) or isinstance(b, bool) or (NUMPY and isinstance(b, np.bool_)):
        return bool(a) == bool(b)
    if isinstance(a, float) or isinstance(b, float):
        return repr(float(a)) == repr(float(b))
    return int(a) == int(b) and type(a) is int


def outcome(fn, args):
    try:
        return ('ok', repr(fn(*args)))
    except Exception as exc:
        return ('raise', type(exc).__name__)


@unittest.skipUnless(NUMPY, 'NumPy required')
class Compile(unittest.TestCase):
    """Pure-Python checks: run with or without Rust."""

    def test_new_kernels_compile(self):
        for name in ('any_negative', 'count_flags', 'iterate', 'enumerate_weighted', 'builtins_mix', 'stats',
                     'fib_pair', 'uses_defaults', 'matrix_trace', 'matrix_scale', 'outer', 'cumulative',
                     'int_histogram', 'write_many', 'parallel_square', 'parallel_fail', 'long_loop'):
            Compiler(getattr(k, name)).generate()

    def test_return_kinds(self):
        for name, expected in (('stats', 'tuple(f64,f64)'), ('fib_pair', 'tuple(i64,i64)'), ('outer', 'array2'),
                               ('cumulative', 'array'), ('int_histogram', 'iarray'), ('any_negative', 'bool')):
            compiler = Compiler(getattr(k, name))
            compiler.generate()
            self.assertEqual(compiler.return_type, expected, name)

    def test_specialization_kinds(self):
        Compiler(k.untyped_sum, kinds={'n': 'i64', 'step': 'i64'}).generate()
        with self.assertRaises(Unsupported) as ctx:
            Compiler(k.untyped_sum, kinds={'n': 'i64', 'step': 'f64'}).generate()
        self.assertIn('changes type', str(ctx.exception))

    def test_parallel_rules(self):
        with self.assertRaises(Unsupported) as ctx:
            Compiler(k.bad_parallel_float_sum).generate()
        self.assertIn('order of additions', str(ctx.exception))
        with self.assertRaises(Unsupported) as ctx:
            Compiler(k.bad_parallel_neighbour).generate()
        self.assertIn('loop index', str(ctx.exception))

    def test_hints(self):
        def uses_print(n):
            total = 0
            for i in range(n):
                print(i)
            return total

        def uses_list(n):
            total = 0
            for i in range(n):
                total += len([i])
            return total

        def rounds(x: float) -> float:
            total = 0.0
            for i in range(3):
                total += round(x, 2)
            return total
        for fn, text in ((uses_print, 'Remove print()'), (uses_list, 'NumPy array'), (rounds, 'round(x) is supported')):
            with self.assertRaises(Unsupported) as ctx:
                Compiler(fn).generate()
            self.assertIn('hint:', str(ctx.exception))
            self.assertIn(text, str(ctx.exception))

    def test_hypot_explained(self):
        def h(x: float) -> float:
            total = 0.0
            for i in range(2):
                total += math.hypot(x, 1.0)
            return total
        with self.assertRaises(Unsupported) as ctx:
            Compiler(h).generate()
        self.assertIn('math.sqrt(x * x + y * y)', str(ctx.exception))

    def test_ipython_magics(self):
        from pyrust_engine import ipython

        class FakeShell:
            def __init__(self):
                self.user_ns = {'iterate': k.iterate, 'data': np.arange(5.0)}
                self.magics = {}

            def register_magic_function(self, func, kind, name):
                self.magics[name] = func
        shell = FakeShell()
        import pyrust_engine
        pyrust_engine.load_ipython_extension(shell)
        self.assertEqual(set(shell.magics), {'pyrust_check', 'pyrust_benchmark'})
        out = io.StringIO()
        import contextlib
        with contextlib.redirect_stdout(out), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            shell.magics['pyrust_check']('iterate')
            shell.magics['pyrust_benchmark']('iterate(data)')
        text = out.getvalue()
        self.assertIn('iterate: compiles to Rust', text)
        self.assertIn('same result', text)

    def test_auto_prune(self):
        from pyrust_engine import cache
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            root = Path(d)
            old = root / ('a' * 64 + '.json')
            old.write_text('{"sha256": "' + 'b' * 64 + '"}')
            lib = root / ('a' * 64 + '-' + 'b' * 64 + cache.SUFFIX)
            lib.write_bytes(b'x')
            past = time.time() - 40 * 86400
            os.utime(old, (past, past))
            cache.auto_prune(root)
            self.assertFalse(old.exists())
            self.assertFalse(lib.exists())
            self.assertTrue((root / '.last-prune').exists())


@unittest.skipUnless(NUMPY and RUST, 'NumPy and a working Rust required')
class Native(unittest.TestCase):
    def check(self, kernel, make_args, *, native=True):
        """Same result (or exception type) and same array contents as plain Python.

        Python returns NumPy scalars for values that come from arrays; PyRust returns plain
        float/int with the same value, so values are compared, not types."""
        fn = optimize(kernel, warn_on_fallback=False)
        native_args, python_args = make_args(), make_args()
        got, want = run(fn, native_args), run(kernel, python_args)
        self.assertEqual(got[0], want[0], (kernel.__name__, got, want))
        if got[0] == 'ok':
            self.assertTrue(same_value(got[1], want[1]), (kernel.__name__, got[1], want[1]))
        for a, b in zip(native_args, python_args):
            if isinstance(a, np.ndarray):
                self.assertEqual(a.tobytes(), b.tobytes(), kernel.__name__)
        if native:
            report = fn.report()
            self.assertEqual(report['backend'], 'rust', report['compile_error'])
            self.assertGreater(report['native_calls'], 0)
        return fn

    def test_bools(self):
        self.check(k.any_negative, lambda: (np.array([1.0, 2.0, -3.0]),))
        self.check(k.any_negative, lambda: (np.array([1.0, 2.0]),))
        self.check(k.count_flags, lambda: (100, True))
        self.check(k.count_flags, lambda: (100, False))
        self.assertIs(optimize(k.any_negative)(np.array([-1.0])), True)

    def test_iteration(self):
        data = np.random.default_rng(1).standard_normal(1000)
        fn = self.check(k.iterate, lambda: (data.copy(),))
        self.assertIsInstance(fn(data), float)
        self.check(k.enumerate_weighted, lambda: (data.copy(),))
        self.check(k.iterate, lambda: (np.zeros(0),))

    def test_builtins(self):
        data = np.arange(10.0)
        self.check(k.builtins_mix, lambda: (data.copy(), 17, 5))
        self.check(k.builtins_mix, lambda: (data.copy(), -17, 5))
        self.check(k.builtins_mix, lambda: (data.copy(), 17, 0), native=False)   # ZeroDivisionError
        self.check(k.builtins_mix, lambda: (np.zeros(0), 1, 1), native=False)     # sum([]) is the int 0

    def test_tuples(self):
        self.assertEqual(optimize(k.fib_pair)(50), k.fib_pair(50))
        self.assertEqual(optimize(k.fib_pair)(100), k.fib_pair(100))  # overflow: replayed, exact
        data = np.random.default_rng(2).standard_normal(500)
        fn = optimize(k.stats)
        self.assertEqual(fn(data), k.stats(data))
        self.assertEqual(fn.report()['backend'], 'rust')

    def test_defaults_and_keywords(self):
        self.check(k.uses_defaults, lambda: (50,))

    def test_specialization(self):
        fn = optimize(k.untyped_sum, warn_on_fallback=False)
        self.assertEqual(fn(100, 3), k.untyped_sum(100, 3))
        self.assertEqual(fn(100, 0.5), k.untyped_sum(100, 0.5))   # float step: runs in Python, exact
        self.assertEqual(fn(100, True), k.untyped_sum(100, True))
        kinds = {tuple(s) for s in fn.report()['specializations']}
        self.assertIn(('i64', 'i64'), kinds)
        self.assertIn(('i64', 'bool'), kinds)

    def test_two_d_arrays(self):
        m = np.arange(12.0).reshape(3, 4)
        self.check(k.matrix_trace, lambda: (m.copy(),))
        self.check(k.matrix_scale, lambda: (m.copy(), 2.5))
        self.check(k.matrix_scale, lambda: (np.asfortranarray(m), 2.5), native=False)  # not C-contiguous

    def test_new_arrays(self):
        a, b = np.arange(3.0), np.arange(4.0)
        fn = optimize(k.outer)
        result = fn(a, b)
        np.testing.assert_array_equal(result, k.outer(a, b))
        self.assertEqual(result.shape, (3, 4))
        self.assertEqual(result.dtype, np.float64)
        np.testing.assert_array_equal(optimize(k.cumulative)(np.arange(5.0)), k.cumulative(np.arange(5.0)))
        ints = np.arange(-20, 100, dtype=np.int64)
        counts = optimize(k.int_histogram)(ints, 7)
        self.assertEqual(counts.dtype, np.int64)
        np.testing.assert_array_equal(counts, k.int_histogram(ints, 7))
        empty = optimize(k.cumulative)(np.zeros(0))
        self.assertEqual(empty.shape, (0,))

    def test_undo_log(self):
        # Many writes, then a failure: the array must be restored before the Python replay.
        self.check(k.write_many, lambda: (np.zeros(5), 1000), native=False)
        self.check(k.write_many, lambda: (np.zeros(5), 999))
        self.check(k.write_many, lambda: (np.zeros(5000), 3), native=False)

    def test_prange(self):
        data = np.random.default_rng(3).standard_normal(100_000)
        fn = self.check(k.parallel_square, lambda: (data.copy(), np.zeros(data.size)))
        self.assertIn('parallel', fn.explain())
        self.check(k.parallel_square, lambda: (data[:10].copy(), np.zeros(10)))
        values = data.copy()
        values[5000] = 0.0
        self.check(k.parallel_fail, lambda: (values.copy(), np.zeros(values.size)), native=False)

    def test_long_call_has_no_budget_trap(self):
        fn = optimize(k.long_loop)
        n = 60_000_000  # more than the 50M inline budget
        self.assertEqual(fn(n), (n // 7) * 21 + sum(range(n % 7)))
        report = fn.report()
        self.assertEqual(report['fallback_calls'], 0)
        self.assertEqual(report['long_calls'], 1)

    def test_strict_budget_still_available(self):
        fn = optimize(k.long_loop, max_iterations=1000, warn_on_fallback=False)
        self.assertEqual(fn(5000), k.long_loop(5000))
        self.assertIn('budget', fn.report()['reason'])

    def test_ctrl_c_stops_long_call(self):
        script = ('import _thread, threading, sys; sys.path.insert(0, sys.argv[1])\n'
                  'from tests.kernels7 import long_loop\nfrom pyrust_engine import optimize\n'
                  'fn = optimize(long_loop)\nfn(10)\n'
                  'threading.Timer(1.0, _thread.interrupt_main).start()\n'
                  'try:\n    fn(10**15)\nexcept KeyboardInterrupt:\n    print("interrupted")\n')
        out = subprocess.run([sys.executable, '-c', script, str(ROOT)], capture_output=True, text=True, timeout=120)
        self.assertIn('interrupted', out.stdout, out.stderr)

    def test_prebuilt_runs_without_rust(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            module = Path(d) / 'shipped.py'
            module.write_text('def work(n):\n    t = 0\n    for i in range(n):\n        t += i * i\n    return t\n'
                              'def tiny(a, b):\n    return a + b\n')
            env = dict(os.environ, PYTHONPATH=str(ROOT))
            out = subprocess.run([sys.executable, '-m', 'pyrust_engine', 'build', str(module)],
                                 capture_output=True, text=True, env=env, timeout=300)
            self.assertIn('built work', out.stdout, out.stderr)
            self.assertTrue((Path(d) / '__pyrust__' / 'manifest.json').exists())
            script = ('import sys; sys.path.insert(0, sys.argv[1])\n'
                      'from unittest.mock import patch\n'
                      'import shipped\nfrom pyrust_engine import optimize\n'
                      'with patch("pyrust_engine.toolchain.find_rustc", return_value=None):\n'
                      '    fn = optimize(shipped.work)\n'
                      '    print(fn(1000), fn.report()["backend"])\n')
            out = subprocess.run([sys.executable, '-c', script, d], capture_output=True, text=True, env=env,
                                 timeout=120)
            self.assertIn(f'{sum(i * i for i in range(1000))} rust', out.stdout, out.stderr)


if __name__ == '__main__':
    unittest.main()
