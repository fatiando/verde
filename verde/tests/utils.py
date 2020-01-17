"""
Testing utilities.
"""
import pytest

try:
    import numba
except ImportError:
    numba = None


def requires_numba(function):
    """
    Skip the decorated test if numba is not installed.
    """
    mark = pytest.mark.skipif(numba is None, reason="requires numba")
    return mark(function)
