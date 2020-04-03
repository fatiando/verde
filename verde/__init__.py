# pylint: disable=missing-docstring,import-outside-toplevel
# Import functions/classes to make the public API
from . import datasets
from . import version
from .coordinates import (
    scatter_points,
    grid_coordinates,
    inside,
    block_split,
    rolling_window,
    expanding_window,
    profile_coordinates,
    get_region,
    pad_region,
    longitude_continuity,
)
from .mask import distance_mask, convexhull_mask
from .utils import variance_to_weights, maxabs, grid_to_table
from .io import load_surfer
from .distances import median_distance
from .blockreduce import BlockReduce, BlockMean
from .scipygridder import ScipyGridder
from .trend import Trend
from .chain import Chain
from .spline import Spline, SplineCV
from .model_selection import cross_val_score, train_test_split
from .vector import Vector, VectorSpline2D
from .projections import project_region, project_grid


def test(doctest=True, verbose=True, coverage=False, figures=True):
    """
    Run the test suite.

    Uses `py.test <http://pytest.org/>`__ to discover and run the tests.

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
