"""
Testing utilities.
"""
import pytest

try:
    import numba
except ImportError:
    numba = None

try:
    import dask
except ImportError:
    dask = None


def requires_numba(function):
    """
    Skip the decorated test if numba is not installed.
    """
    mark = pytest.mark.skipif(numba is None, reason="requires numba")
    return mark(function)


def requires_dask(function):
    """
    Skip the decorated test if dask is not installed.
    """
    mark = pytest.mark.skipif(dask is None, reason="requires dask")
    return mark(function)
