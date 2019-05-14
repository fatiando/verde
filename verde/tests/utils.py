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


def requires_dask(function):
    """
    Skip the decorated test if dask.distributed is not installed.
    """
    try:
        import dask.distributed
    except ImportError:
        dask = None
    mark = pytest.mark.skipif(dask is None, reason="requires dask")
    return mark(function)
