"""0.6.1: finding Rust, diagnosing missing or broken toolchains, and reporting it once."""
import io
import os
import subprocess
import unittest
import warnings
from unittest.mock import patch

from pyrust_engine import optimize, check_rust, doctor, RustToolchainWarning, OptimizationError
from pyrust_engine import runtime, toolchain
from tests import kernels5 as k


def no_rust():
    return patch('pyrust_engine.toolchain.find_rustc', return_value=None)


class Diagnosis(unittest.TestCase):
    def setUp(self):
        runtime._TOOLCHAIN_WARNED.clear()

    def test_not_found_message_has_fix(self):
        message = str(toolchain.not_found_error())
        self.assertIn('rustc', message)
        self.assertIn('still runs correctly as plain Python', message)
        self.assertIn('python -m pyrust_engine doctor', message)
        self.assertTrue(any(word in message for word in ('rustup', 'brew', 'winget')))

    def test_rustup_without_default_toolchain(self):
        error = toolchain.classify_version_failure(
            "error: rustup could not choose a version of rustc to run, because one wasn't specified "
            "explicitly, and no default is configured.\nhelp: run 'rustup default stable' "
            "to download the latest stable release of Rust and set it as your default toolchain.\n"
            "error: no override and no default toolchain set")
        self.assertIn('rustup default stable', str(error))

    def test_interrupted_install(self):
        for text in ("error: missing manifest in toolchain 'stable-aarch64-apple-darwin'",
                     "error: command failed: 'rustc': No such file or directory (os error 2)"):
            error = toolchain.classify_version_failure(text)
            self.assertIn('installed incompletely', str(error))
            self.assertIn('rustup toolchain uninstall stable && rustup default stable', str(error))

    def test_unknown_failure_shows_first_line(self):
        error = toolchain.classify_version_failure('\n  something odd happened\nmore')
        self.assertIn('something odd happened', str(error))

    def test_build_failures(self):
        self.assertIn('xcode-select --install', str(toolchain.classify_build_failure(
            'xcrun: error: invalid active developer path (/Library/Developer/CommandLineTools)')))
        self.assertIn('Build Tools', str(toolchain.classify_build_failure(
            'error: linker `link.exe` not found')))
        self.assertIn('C linker', str(toolchain.classify_build_failure(
            'error: linker `cc` not found\n  = note: No such file or directory')))
        self.assertIsNone(toolchain.classify_build_failure('error[E0308]: mismatched types'))

    def test_override_must_exist(self):
        with patch.dict(os.environ, {'PYRUST_RUSTC': '/definitely/not/here/rustc'}):
            self.assertIsNone(toolchain.find_rustc())

    def test_rustc_version_errors(self):
        toolchain._version_cache.clear()
        failed = subprocess.CompletedProcess(['rustc', '-vV'], 1, '', "error: missing manifest in toolchain 'x'")
        with patch('subprocess.run', return_value=failed):
            with self.assertRaises(toolchain.ToolchainError):
                toolchain.rustc_version('/fake/rustc')
        with patch('subprocess.run', side_effect=subprocess.TimeoutExpired('rustc', 30)):
            with self.assertRaises(toolchain.ToolchainError) as ctx:
                toolchain.rustc_version('/fake/rustc2')
        self.assertIn('30 seconds', str(ctx.exception))
        with patch('subprocess.run', side_effect=FileNotFoundError()):
            with self.assertRaises(toolchain.ToolchainError):
                toolchain.rustc_version('/fake/rustc3')

    def test_check_rust_missing(self):
        with no_rust():
            status = check_rust()
        self.assertFalse(status['ok'])
        self.assertIsNone(status['rustc'])
        self.assertIn('not found', status['problem'])
        self.assertTrue(status['fix'])

    def test_check_rust_broken(self):
        toolchain._version_cache.clear()
        failed = subprocess.CompletedProcess(['rustc', '-vV'], 1, '', 'error: no default toolchain set')
        with patch('pyrust_engine.toolchain.find_rustc', return_value='/fake/rustc'), \
                patch('subprocess.run', return_value=failed):
            status = check_rust()
        self.assertFalse(status['ok'])
        self.assertEqual(status['fix'], ['rustup default stable'])

    def test_one_warning_for_all_functions(self):
        a = optimize(k.int_ops)
        b = optimize(k.helper_sum)
        with no_rust(), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self.assertEqual(a(10), k.int_ops(10))
            self.assertEqual(b(10), k.helper_sum(10))
            self.assertEqual(a(20), k.int_ops(20))
        relevant = [w for w in caught if issubclass(w.category, RustToolchainWarning)]
        self.assertEqual(len(relevant), 1)
        self.assertIn('Shown once', str(relevant[0].message))
        self.assertTrue(a.report()['toolchain_problem'])
        self.assertTrue(b.report()['toolchain_problem'])

    def test_results_correct_without_rust(self):
        fn = optimize(k.float_ops, warn_on_fallback=False)
        with no_rust():
            self.assertEqual(fn(20, 1.5), k.float_ops(20, 1.5))
        self.assertEqual(fn.report()['backend'], 'python')

    def test_strict_mode_reports_toolchain(self):
        fn = optimize(k.int_ops, fallback=False)
        with no_rust():
            with self.assertRaises(OptimizationError) as ctx:
                fn(5)
        self.assertIn('rustc', str(ctx.exception))

    def test_explain_shows_fix(self):
        with no_rust():
            text = optimize(k.int_ops, warn_on_fallback=False).explain()
        self.assertIn('not found', text)
        self.assertIn('Calls run as plain Python until it is fixed', text)

    def test_retry_after_fixing(self):
        fn = optimize(k.int_ops, warn_on_fallback=False)
        with no_rust():
            fn(3)
        self.assertEqual(fn.report()['backend'], 'python')
        fn.reset()
        if check_rust()['ok']:
            fn(3)
            self.assertEqual(fn.report()['backend'], 'rust')
            self.assertFalse(fn.report()['toolchain_problem'])

    def test_doctor_without_rust(self):
        out = io.StringIO()
        with no_rust():
            ok = doctor(out)
        self.assertFalse(ok)
        text = out.getvalue()
        self.assertIn('Rust          : PROBLEM', text)
        self.assertIn('fix:', text)
        self.assertIn('still runs correctly as plain Python', text)

    @unittest.skipUnless(check_rust()['ok'], 'working Rust required')
    def test_doctor_with_rust(self):
        out = io.StringIO()
        self.assertTrue(doctor(out), out.getvalue())
        self.assertIn('Test compile  : OK', out.getvalue())


if __name__ == '__main__':
    unittest.main()
