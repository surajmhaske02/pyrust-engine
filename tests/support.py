"""Test helper: the existing suites exercise the native path, including on loop-free kernels.

Loop-free functions run as plain Python by default (see test_v4.py); these suites
opt back into native compilation so overflow replay, caching and the ABI stay covered.
"""
from pyrust_engine import optimize as _optimize


def optimize(function=None, **options):
    options.setdefault('native_loop_free', True)
    return _optimize(function, **options)
