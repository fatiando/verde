"""
Testing utilities.
"""
import pytest

try:
    import numba
except ImportError:
    numba = None

try:
    from dask import distributed
except ImportError:
    distributed = None


def requires_numba(function):
    """
    Skip the decorated test if numba is not installed.
    """
    mark = pytest.mark.skipif(numba is None, reason="requires numba")
    return mark(function)


def requires_dask(function):
    """
    Skip the decorated test if dask.distributed is not installed.
    """
    mark = pytest.mark.skipif(distributed is None, reason="requires dask")
    return mark(function)
