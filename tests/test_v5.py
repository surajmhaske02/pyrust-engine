"""0.5.0: math operators, math module, builtins, globals, calls between functions,
fallback warnings, explain() and the `check` CLI."""
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch

from pyrust_engine import optimize, PyRustFallbackWarning, OptimizationError
from pyrust_engine.compiler import Compiler, Unsupported
from tests import kernels5 as k

RUST = bool(shutil.which('rustc'))
NUMPY = importlib.util.find_spec('numpy') is not None
ROOT = Path(__file__).resolve().parent.parent


def outcome(fn, args):
    try:
        return ('ok', repr(fn(*args)))
    except Exception as exc:
        return ('raise', type(exc).__name__)


class Explain(unittest.TestCase):
    """Pure-Python checks: run with or without Rust."""

    def test_new_features_compile(self):
        for name in ('int_ops', 'float_ops', 'math_ops', 'ternary_and_while_true', 'globals_used',
                     'nan_checks', 'calls_helpers', 'calls_with_array', 'shift_edges',
                     'int_true_division', 'float_floor_mod', 'to_int', 'float_pow', 'int_pow'):
            Compiler(getattr(k, name)).generate()  # must not raise

    def test_callees_listed(self):
        compiler = Compiler(k.calls_helpers)
        compiler.generate()
        self.assertEqual(set(compiler.callees), {'helper_square', 'helper_sum', 'decorated_helper'})

    def test_recursion_rejected_with_line(self):
        with self.assertRaises(Unsupported) as ctx:
            Compiler(k.recursive_unannotated).generate()
        self.assertIn('recursive call', str(ctx.exception))
        self.assertIn('annotate its return type', str(ctx.exception))
        self.assertIn('line ', str(ctx.exception))

    def test_error_message_has_line_and_code(self):
        def uses_list(n):
            total = 0
            for i in range(n):
                total += len([i, i])
            return total
        with self.assertRaises(Unsupported) as ctx:
            Compiler(uses_list).generate()
        message = str(ctx.exception)
        self.assertIn('uses_list, line', message)
        self.assertIn('[i, i]', message)

    def test_type_change_hint(self):
        def accumulates(n: int) -> float:
            total = 0
            for i in range(n):
                total += i / 2
            return total
        with self.assertRaises(Unsupported) as ctx:
            Compiler(accumulates).generate()
        self.assertIn('start float variables at 0.0', str(ctx.exception))

    def test_explain_native(self):
        text = optimize(k.calls_helpers, warn_on_fallback=False).explain()
        self.assertIn('compiles to Rust', text)
        self.assertIn('n: int', text)
        self.assertIn('returns: float', text)
        self.assertIn('helper_square', text)

    def test_explain_unsupported(self):
        text = optimize(k.recursive_unannotated, warn_on_fallback=False).explain()
        self.assertIn('runs as plain Python', text)
        self.assertIn('recursive call', text)

    def test_explain_loop_free(self):
        text = optimize(k.helper_square).explain()
        self.assertIn('on purpose', text)
        self.assertIn('native_loop_free=True', text)

    def test_loop_free_caller_of_loop_function_is_compiled(self):
        fn = optimize(k.wraps_helper, warn_on_fallback=False)
        self.assertEqual(fn.report()['dispatch'], 'wrapper')
        self.assertIn('helper_sum', fn.explain())

    def test_fallback_warning_once_per_reason(self):
        from pyrust_engine import runtime
        runtime._TOOLCHAIN_WARNED.clear()  # toolchain warnings are once per process
        fn = optimize(k.int_ops)
        with patch('shutil.which', return_value=None):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                for _ in range(3):
                    self.assertEqual(fn(20), k.int_ops(20))
        relevant = [w for w in caught if issubclass(w.category, PyRustFallbackWarning)]
        self.assertEqual(len(relevant), 1)
        self.assertIn('int_ops ran as plain Python', str(relevant[0].message))
        self.assertIn('rustc', str(relevant[0].message))

    def test_fallback_warning_can_be_disabled(self):
        fn = optimize(k.int_ops, warn_on_fallback=False)
        with patch('shutil.which', return_value=None):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                fn(5)
        self.assertFalse([w for w in caught if issubclass(w.category, PyRustFallbackWarning)])

    def test_check_cli(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            path = Path(d) / 'work.py'
            path.write_text(
                'import math\n'
                'def good(n):\n    t = 0.0\n    for i in range(n):\n        t += math.sqrt(i)\n    return t\n'
                'def bad(n):\n    t = 0\n    for i in range(n):\n        t += len([i])\n    return t\n'
                'def tiny(a, b):\n    return a + b\n'
                'if __name__ == "__main__":\n    raise SystemExit("main block must not run")\n')
            env = dict(os.environ, PYTHONPATH=str(ROOT))
            out = subprocess.run([sys.executable, '-m', 'pyrust_engine', 'check', str(path), '--json'],
                                 capture_output=True, text=True, env=env, timeout=60)
            self.assertEqual(out.returncode, 0, out.stderr)
            rows = {r['function']: r for r in json.loads(out.stdout)}
            self.assertTrue(rows['good']['native'])
            self.assertFalse(rows['bad']['native'])
            self.assertIn('line 10', rows['bad']['explanation'])
            self.assertFalse(rows['tiny']['native'])
            self.assertIn('on purpose', rows['tiny']['explanation'])
            text = subprocess.run([sys.executable, '-m', 'pyrust_engine', 'check', str(path)],
                                  capture_output=True, text=True, env=env, timeout=60).stdout
            self.assertIn('1 of 3 functions compile to Rust.', text)


@unittest.skipUnless(RUST, 'rustc required')
class Native(unittest.TestCase):
    def check(self, kernel, cases, *, native=True):
        fn = optimize(kernel, warn_on_fallback=False)
        for args in cases:
            self.assertEqual(outcome(fn, args), outcome(kernel, args), f'{kernel.__name__}{args}')
        if native:
            self.assertEqual(fn.report()['backend'], 'rust', fn.report()['compile_error'])
            self.assertGreater(fn.report()['native_calls'], 0)
        return fn

    def test_int_ops(self):
        fn = self.check(k.int_ops, [(0,), (1,), (200,)])
        self.assertEqual(fn.report()['fallback_calls'], 0)

    def test_float_ops(self):
        self.check(k.float_ops, [(1, 0.5), (50, 3.25), (20, -1.75), (5, float('nan')), (5, float('inf'))])

    def test_math_ops(self):
        fn = self.check(k.math_ops, [(1, 0.5), (100, 2.0), (30, 1e6)])
        self.assertEqual(fn.report()['fallback_calls'], 0)

    def test_math_domain_errors_replay(self):
        fn = self.check(k.math_domain, [(10.0,), (2.5,), (1.0,), (-1.0,), (float('nan'),), (float('inf'),)])
        self.assertIn('outside the native fast path', ' '.join(fn.report()['fallback_reasons']))

    def test_float_pow_edges(self):
        values = [0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, 2.5, 1e300, -1e300, 1e-300,
                  float('inf'), float('-inf'), float('nan')]
        cases = [(a, b) for a in values for b in values]
        self.check(k.float_pow, cases)

    def test_int_pow_edges(self):
        cases = [(a, b) for a in (0, 1, -1, 2, -2, 3, 10, 2**31) for b in (0, 1, 2, 3, 31, 62, 63, 64, -1, -2)]
        cases += [(a, 2**33) for a in (0, 1, -1)]  # huge exponent: only cheap bases
        self.check(k.int_pow, cases)

    def test_shifts(self):
        cases = [(a, b) for a in (0, 1, -1, 5, -5, 2**62, -(2**63)) for b in (0, 1, 2, 62, 63, 64, 100, -1)]
        self.check(k.shift_edges, cases)

    def test_int_true_division(self):
        cases = [(7, 2), (-7, 2), (1, 3), (0, 5), (5, 0), (2**53, 3), (2**53 + 1, 3), (2**62, 7), (-(2**63), -1)]
        self.check(k.int_true_division, cases)

    def test_float_floor_div_and_mod(self):
        values = [7.5, -7.5, 2.0, -2.0, 0.0, -0.0, 1e-300, 1e300, float('inf'), float('-inf'), float('nan'), 0.1]
        self.check(k.float_floor_mod, [(a, b) for a in values for b in values])

    def test_to_int(self):
        self.check(k.to_int, [(x,) for x in (0.0, -0.5, 2.9, -2.9, 1e18, 9.3e18, -9.3e18, float('nan'), float('inf'))])

    def test_ternary_and_while_true(self):
        self.check(k.ternary_and_while_true, [(1,), (10,), (1000,)])

    def test_globals_and_rebinding(self):
        fn = self.check(k.globals_used, [(10,), (1000,)])
        old = k.SCALE
        try:
            k.SCALE = 5
            self.assertEqual(fn(10), k.globals_used(10))
            self.assertIn('SCALE has been rebound', fn.report()['reason'])
        finally:
            k.SCALE = old

    def test_nan_checks(self):
        self.check(k.nan_checks, [(1.0,), (float('nan'),), (float('inf'),), (float('-inf'),)])

    def test_loop_free_caller_compiles_callee(self):
        self.check(k.wraps_helper, [(0,), (10,), (1000,)])

    def test_calls_between_functions(self):
        fn = self.check(k.calls_helpers, [(0,), (25,), (500,)])
        self.assertEqual(fn.report()['fallback_calls'], 0)

    def test_callee_rebinding_falls_back(self):
        fn = self.check(k.calls_helpers, [(10,)])
        original = k.helper_square
        try:
            k.helper_square = lambda v: 0.0
            self.assertEqual(fn(10), k.calls_helpers(10))
            self.assertIn('helper_square has been rebound', fn.report()['reason'])
        finally:
            k.helper_square = original

    @unittest.skipUnless(NUMPY, 'NumPy required')
    def test_array_passed_to_callee(self):
        import numpy as np
        data = np.arange(50, dtype=np.float64)
        self.check(k.calls_with_array, [(data,), (np.array([], dtype=np.float64),)])

    def test_math_imported_name_rebinding(self):
        fn = self.check(k.math_ops, [(5, 1.0)])
        original = k.square_root
        try:
            k.square_root = lambda y: 0.0
            self.assertEqual(fn(5, 1.0), k.math_ops(5, 1.0))
            self.assertIn('square_root has been rebound', fn.report()['reason'])
        finally:
            k.square_root = original

    def test_strict_mode_reports_domain_error(self):
        fn = optimize(k.math_domain, fallback=False)
        with self.assertRaises(OptimizationError):
            fn(-5.0)


if __name__ == '__main__':
    unittest.main()
