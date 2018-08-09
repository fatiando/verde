"""
Testing utilities.
"""
import pytest


def requires_numba(function):
    """
    Skip the decorated test if numba is not installed.
    """
    try:
        import numba
    except ImportError:
        numba = None
    mark = pytest.mark.skipif(numba is None, reason="requires numba")
    return mark(function)
