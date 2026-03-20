# Copyright (c) 2017 The Verde Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test the utility functions.
"""

import numpy as np
import pytest
import xarray as xr

from ..fill import fill_missing
from ..neighbors import KNeighbors
from ..scipygridder import Cubic, Linear
from ..spline import Spline
from ..trend import Trend
from ..utils import make_xarray_grid


def test_fill_missing_types_metadata_names():
    """
    Test for correct types, retained metadata, and names when filling grid
    NaNs.
    """
    region = (-10, -5, 6, 10)
    easting = np.linspace(*region[:2], 6, dtype=float)
    northing = np.linspace(*region[2:], 5, dtype=float)
    data1 = np.ones((northing.size, easting.size))
    data2 = np.ones((northing.size, easting.size))

    # for data1 make 2 corners nan, and the nearest two points to one of the
    # corners have values of 4 instead of 1
    data1[0][0] = np.nan
    data1[-1][-1] = np.nan
    data1[0][1] = 4
    data1[1][0] = 4

    # for data2 make center value nan
    data2[2][2] = np.nan
    data2[2][2] = np.nan

    # make dataset with 2 variables
    grid = make_xarray_grid(
        (easting, northing),
        (data1, data2),
        data_names=("dummy1", "dummy2"),
        dims=("north", "east"),
    )
    # add attribute to test it's retained
    grid = grid.assign_attrs({"units": "mGal"})
    ds = grid.dummy1.to_dataset(name="dummy1").assign_attrs({"units": "mGal"})

    # 3 types passed, dataarray, dataset with 2 vars, dataset with 1 var
    filled_da = fill_missing(grid.dummy1)
    filled_ds_2var = fill_missing(grid)
    filled_ds_1var = fill_missing(ds)

    # check return types match inputs
    assert isinstance(filled_da, xr.DataArray)
    assert isinstance(filled_ds_2var, xr.Dataset)
    assert isinstance(filled_ds_1var, xr.Dataset)

    # check dimensions and names haven't changed
    assert filled_da.name == "dummy1"
    assert filled_da.dims == ("north", "east")
    assert set(filled_ds_2var.variables) == set(["east", "north", "dummy1", "dummy2"])
    assert set(filled_ds_1var.variables) == set(["east", "north", "dummy1"])

    # assert metadata hasn't changed
    with pytest.raises(AttributeError):
        filled_da.units  # attribute only for dataset, not variable
    assert filled_ds_2var.units == "mGal"
    assert filled_ds_1var.units == "mGal"

    # test that function still works if 1 variable doesn't contain nans
    grid["dummy1"] = filled_ds_2var.dummy1
    assert grid.dummy1.notnull().any()
    filled_ds_no_nans = fill_missing(grid)
    assert filled_ds_no_nans.dummy1.notnull().any()
    assert filled_ds_no_nans.dummy2.notnull().any()


fill_missing_nearest_test = [
    (
        KNeighbors(k=1),
        np.array(
            [
                [4.0, 4.0, 1.0, 1.0, 1.0, 1.0],
                [4.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        ),
    ),
    (
        KNeighbors(k=2),
        np.array(
            [
                [4.0, 4.0, 1.0, 1.0, 1.0, 1.0],
                [4.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        ),
    ),
    (
        KNeighbors(k=3),
        np.array(
            [
                [3.0, 4.0, 1.0, 1.0, 1.0, 1.0],
                [4.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        ),
    ),
]


@pytest.mark.parametrize(("interpolator", "expected"), fill_missing_nearest_test)
def test_fill_missing_nearest(interpolator, expected):
    """
    Test filling NaNs on a small sample grid with multiple numbers of neighbors
    """
    region = (-10, -5, 6, 10)
    easting = np.linspace(*region[:2], 6, dtype=float)
    northing = np.linspace(*region[2:], 5, dtype=float)
    data1 = np.ones((northing.size, easting.size))
    data2 = np.ones((northing.size, easting.size))

    # for data1 make 2 corners nan, and the nearest two points to one of the
    # corners have values of 4 instead of 1
    data1[0][0] = np.nan
    data1[-1][-1] = np.nan
    data1[0][1] = 4
    data1[1][0] = 4

    # for data2 make center value nan
    data2[2][2] = np.nan
    data2[2][2] = np.nan

    # make dataset with 2 variables
    grid = make_xarray_grid(
        (easting, northing),
        (data1, data2),
        data_names=("dummy1", "dummy2"),
    )
    ds = grid.dummy1.to_dataset(name="dummy1")

    # 3 types passed, dataarray, dataset with 2 vars, dataset with 1 var
    filled_da = fill_missing(grid.dummy1, interpolator)
    filled_ds_2var = fill_missing(grid, interpolator)
    filled_ds_1var = fill_missing(ds, interpolator)

    # check no nans remain
    assert filled_da.notnull().any()
    assert filled_ds_2var.dummy1.notnull().any()
    assert filled_ds_2var.dummy2.notnull().any()
    assert filled_ds_1var.dummy1.notnull().any()

    # check correct values
    np.testing.assert_allclose(filled_da.values, expected)
    np.testing.assert_allclose(filled_ds_2var.dummy1.values, expected)
    np.testing.assert_allclose(filled_ds_1var.dummy1.values, expected)


fill_missing_trend_test = [
    (
        Trend(0),
        np.array(
            [
                [1.2962963, 1, 1, 1, 1, 1],
                [1, 2, 2, 2, 1, 1],
                [1, 2, 1.2962963, 2, 1, 1],
                [1, 2, 2, 2, 1, 1],
                [1, 1, 1, 1, 1, 1.2962963],
            ]
        ),
    ),
    (
        Trend(1),
        np.array(
            [
                [1.4616769, 1, 1, 1, 1, 1],
                [1, 2, 2, 2, 1, 1],
                [1, 2, 1.32583434, 2, 1, 1],
                [1, 2, 2, 2, 1, 1],
                [1, 1, 1, 1, 1, 1.13302555],
            ]
        ),
    ),
    (
        Trend(5),
        np.array(
            [
                [0.23084858, 1, 1, 1, 1, 1],
                [1, 2, 2, 2, 1, 1],
                [1, 2, 2.22928874, 2, 1, 1],
                [1, 2, 2, 2, 1, 1],
                [1, 1, 1, 1, 1, 2.22563067],
            ]
        ),
    ),
]


@pytest.mark.parametrize(("interpolator", "expected"), fill_missing_trend_test)
def test_fill_missing_trend(interpolator, expected):
    """
    Test filling NaNs on a small sample grid with multiple trend orders
    """
    region = (-10, -5, 6, 10)
    easting = np.linspace(*region[:2], 6, dtype=float)
    northing = np.linspace(*region[2:], 5, dtype=float)

    # make data with 2 corners as nans, and a nan in the middle surrounded
    # by 2's, and 1's elsewhere
    data = np.array(
        [
            [np.nan, 1, 1, 1, 1, 1],
            [1, 2, 2, 2, 1, 1],
            [1, 2, np.nan, 2, 1, 1],
            [1, 2, 2, 2, 1, 1],
            [1, 1, 1, 1, 1, np.nan],
        ]
    )

    # make dataset
    grid = make_xarray_grid(
        (easting, northing),
        data,
        data_names="dummy",
    )

    filled_da = fill_missing(grid.dummy, interpolator)

    # check no nans remain
    assert filled_da.notnull().any()

    # check correct values
    np.testing.assert_allclose(filled_da.values, expected)


fill_missing_extrapolation = [
    (
        Linear(),
        np.array(
            [
                [np.nan, 1, 1, 1, 1, 1],
                [1, 2, 2, 2, 1, 1],
                [1, 2, 2, 2, 1, 1],
                [1, 2, 2, 2, 1, 1],
                [1, 1, 1, 1, 1, np.nan],
            ]
        ),
    ),
    (
        Cubic(),
        np.array(
            [
                [np.nan, 1, 1, 1, 1, 1],
                [1, 2, 2, 2, 1, 1],
                [1, 2, 2.3629, 2, 1, 1],
                [1, 2, 2, 2, 1, 1],
                [1, 1, 1, 1, 1, np.nan],
            ]
        ),
    ),
]


@pytest.mark.parametrize(("interpolator", "expected"), fill_missing_extrapolation)
def test_fill_missing_linear_and_cubic(interpolator, expected):
    """
    Test filling NaNs on a small sample grid with a linear interpolation
    """
    region = (-10, -5, 6, 10)
    easting = np.linspace(*region[:2], 6, dtype=float)
    northing = np.linspace(*region[2:], 5, dtype=float)

    # make data with 2 corners as nans, and a nan in the middle surrounded
    # by 2's, and 1's elsewhere
    data = np.array(
        [
            [np.nan, 1, 1, 1, 1, 1],
            [1, 2, 2, 2, 1, 1],
            [1, 2, np.nan, 2, 1, 1],
            [1, 2, 2, 2, 1, 1],
            [1, 1, 1, 1, 1, np.nan],
        ]
    )

    # make dataset
    grid = make_xarray_grid(
        (easting, northing),
        data,
        data_names="dummy",
    )

    # check warning about extrapolation raised
    with pytest.warns(UserWarning, match="NaNs are still present in this grid!"):
        filled_da = fill_missing(grid.dummy, interpolator)

    # check correct values
    np.testing.assert_allclose(filled_da.values, expected, rtol=1e-4)


fill_missing_spline = [
    (
        Spline(),
        np.array(
            [
                [0.00254, 1, 1, 1, 1, 1],
                [1, 2, 2, 2, 1, 1],
                [1, 2, 2.33529, 2, 1, 1],
                [1, 2, 2, 2, 1, 1],
                [1, 1, 1, 1, 1, 0.58934],
            ]
        ),
    ),
    (
        Spline(damping=100),
        np.array(
            [
                [0.80490, 1, 1, 1, 1, 1],
                [1, 2, 2, 2, 1, 1],
                [1, 2, 0.25299, 2, 1, 1],
                [1, 2, 2, 2, 1, 1],
                [1, 1, 1, 1, 1, 0.76676],
            ]
        ),
    ),
]


@pytest.mark.parametrize(("interpolator", "expected"), fill_missing_spline)
def test_fill_missing_spline(interpolator, expected):
    """
    Test filling NaNs on a small sample grid with a spline interpolation
    """
    region = (-10, -5, 6, 10)
    easting = np.linspace(*region[:2], 6, dtype=float)
    northing = np.linspace(*region[2:], 5, dtype=float)

    # make data with 2 corners as nans, and a nan in the middle surrounded
    # by 2's, and 1's elsewhere
    data = np.array(
        [
            [np.nan, 1, 1, 1, 1, 1],
            [1, 2, 2, 2, 1, 1],
            [1, 2, np.nan, 2, 1, 1],
            [1, 2, 2, 2, 1, 1],
            [1, 1, 1, 1, 1, np.nan],
        ]
    )

    # make dataset
    grid = make_xarray_grid(
        (easting, northing),
        data,
        data_names="dummy",
    )

    filled_da = fill_missing(grid.dummy, interpolator)

    # check no nans remain
    assert filled_da.notnull().any()

    # check correct values
    np.testing.assert_allclose(filled_da.values, expected, rtol=1e-3)
