# Copyright (c) 2017 The Verde Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
# Import functions/classes to make the public API
from . import datasets, synthetic
from ._version import __version__
from .blockreduce import BlockMean, BlockReduce
from .chain import Chain
from .coordinates import (
    block_split,
    expanding_window,
    get_region,
    grid_coordinates,
    inside,
    longitude_continuity,
    pad_region,
    profile_coordinates,
    rolling_window,
    scatter_points,
)
from .distances import median_distance
from .io import load_surfer
from .mask import convexhull_mask, distance_mask
from .model_selection import (
    BlockKFold,
    BlockShuffleSplit,
    cross_val_score,
    train_test_split,
)
from .projections import project_grid, project_region
from .scipygridder import ScipyGridder
from .spline import Spline, SplineCV
from .trend import Trend
from .utils import grid_to_table, make_xarray_grid, maxabs, variance_to_weights
from .vector import Vector, VectorSpline2D


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
    import warnings

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

    warnings.warn(
        f"The '{package}.test' function is deprecated since v1.7.0 and "
        "will be removed in v2.0.0. "
        f"Use 'pytest --pyargs {package}' to run the tests.",
        FutureWarning,
    )

    assert status == 0, "Some tests have failed."
