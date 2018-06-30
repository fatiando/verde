"""
The Verde public API.
"""
from ._version import get_versions as _get_versions

# Import functions/classes to make the API
from . import datasets
from .base import BaseGridder
from .coordinates import (
    scatter_points,
    grid_coordinates,
    inside,
    profile_coordinates,
    get_region,
    pad_region,
    project_region,
)
from .grid_math import distance_mask
from .utils import variance_to_weights, maxabs
from .blockreduce import block_split, BlockReduce, BlockMean
from .scipygridder import ScipyGridder
from .trend import Trend, trend_jacobian, VectorTrend
from .chain import Chain
from .spline import Spline, spline_jacobian
from .model_selection import cross_val_score, train_test_split
from .vector import Vector2D, vector2d_jacobian


# Get the version number through versioneer
__version__ = _get_versions()["version"]
__commit__ = _get_versions()["full-revisionid"]


def test(doctest=True, verbose=True, coverage=False, figures=True):
    """
    Run the test suite.

    Uses `py.test <http://pytest.org/>`__ to discover and run the tests. If you
    haven't already, you can install it with `conda
    <http://conda.pydata.org/>`__ or `pip <https://pip.pypa.io/en/stable/>`__.

    Parameters
    ----------

    doctest : bool
        If ``True``, will run the doctests as well (code examples that start
        with a ``>>>`` in the docs).
    verbose : bool
        If ``True``, will print extra information during the test run.
    coverage : bool
        If ``True``, will run test coverage analysis on the code as well.
        Requires ``pytest-cov``.
    figures : bool
        If ``True``, will test generated figures against saved baseline
        figures.  Requires ``pytest-mpl`` and ``matplotlib``.

    Raises
    ------

    AssertionError
        If pytest returns a non-zero error code indicating that some tests have
        failed.

    """
    import pytest

    package = __name__
    args = []
    if verbose:
        args.append("-vv")
    if coverage:
        args.append("--cov={}".format(package))
        args.append("--cov-report=term-missing")
    if doctest:
        args.append("--doctest-modules")
    if figures:
        args.append("--mpl")
    args.append("--pyargs")
    args.append(package)
    status = pytest.main(args)
    assert status == 0, "Some tests have failed."
