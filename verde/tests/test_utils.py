# Copyright (c) 2017 The Verde Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test the utility functions.
"""
from unittest import mock

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
import xarray as xr
from scipy.spatial import cKDTree

from .. import utils
from ..coordinates import grid_coordinates, scatter_points
from ..neighbors import KNeighbors
from ..scipygridder import Cubic, Linear
from ..spline import Spline
from ..trend import Trend
from ..utils import (
    dummy_jit,
    fill_nans,
    get_ndim_horizontal_coords,
    grid_to_table,
    kdtree,
    make_xarray_grid,
    maxabs,
    meshgrid_from_1d,
    meshgrid_to_1d,
    minmax,
    parse_engine,
    partition_by_sum,
    variance_to_weights,
)


def test_fill_nans_types_metadata_names():
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
    filled_da = fill_nans(grid.dummy1)
    filled_ds_2var = fill_nans(grid)
    filled_ds_1var = fill_nans(ds)

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


fill_nans_is_fitted_test = [
    KNeighbors(),
    Linear(),
    Trend(1),
    Cubic(),
    Spline(),
]


@pytest.mark.parametrize(("interpolator"), fill_nans_is_fitted_test)
def test_fill_nans_is_fitted(interpolator):
    """
    Test the correct error is raised if an already-fitted interolator
    is passed.
    """
    region = (-10, -5, 6, 10)
    easting = np.linspace(*region[:2], 6, dtype=float)
    northing = np.linspace(*region[2:], 5, dtype=float)
    data = np.ones((northing.size, easting.size))
    data[-1][-1] = np.nan

    # make dataset
    grid = make_xarray_grid(
        (easting, northing),
        data,
        data_names="dummy",
    )
    da = grid.dummy
    df = grid_to_table(da)
    df_no_nans = df[df.dummy.notna()]
    coords_no_nans = (df_no_nans.northing, df_no_nans.easting)

    # pre-fit the interpolator
    interpolator.fit(coords_no_nans, df_no_nans.dummy)

    with pytest.raises(
        UserWarning, match="The supplied interpolator is already fitted!"
    ):
        fill_nans(da, interpolator)


fill_nans_nearest_test = [
    (
        None,
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


@pytest.mark.parametrize(("interpolator", "expected"), fill_nans_nearest_test)
def test_fill_nans_nearest(interpolator, expected):
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
    filled_da = fill_nans(grid.dummy1, interpolator)
    filled_ds_2var = fill_nans(grid, interpolator)
    filled_ds_1var = fill_nans(ds, interpolator)

    # check no nans remain
    assert filled_da.notnull().any()
    assert filled_ds_2var.dummy1.notnull().any()
    assert filled_ds_2var.dummy2.notnull().any()
    assert filled_ds_1var.dummy1.notnull().any()

    # check correct values
    np.testing.assert_allclose(filled_da.values, expected)
    np.testing.assert_allclose(filled_ds_2var.dummy1.values, expected)
    np.testing.assert_allclose(filled_ds_1var.dummy1.values, expected)


fill_nans_trend_test = [
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


@pytest.mark.parametrize(("interpolator", "expected"), fill_nans_trend_test)
def test_fill_nans_trend(interpolator, expected):
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

    filled_da = fill_nans(grid.dummy, interpolator)

    # check no nans remain
    assert filled_da.notnull().any()

    # check correct values
    np.testing.assert_allclose(filled_da.values, expected)


fill_nans_extrapolation = [
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


@pytest.mark.parametrize(("interpolator", "expected"), fill_nans_extrapolation)
def test_fill_nans_linear_and_cubic(interpolator, expected):
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
    with pytest.warns(
        UserWarning, match="NaNs are still present due to the choice of interpolator"
    ):
        filled_da = fill_nans(grid.dummy, interpolator)

    # check correct values
    np.testing.assert_allclose(filled_da.values, expected, rtol=1e-4)


fill_nans_spline = [
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


@pytest.mark.parametrize(("interpolator", "expected"), fill_nans_spline)
def test_fill_nans_spline(interpolator, expected):
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

    filled_da = fill_nans(grid.dummy, interpolator)

    # check no nans remain
    assert filled_da.notnull().any()

    # check correct values
    np.testing.assert_allclose(filled_da.values, expected, rtol=1e-3)


def test_parse_engine():
    "Check that it works for common input"
    assert parse_engine("numba") == "numba"
    assert parse_engine("numpy") == "numpy"
    with mock.patch.object(utils, "numba", None):
        assert parse_engine("auto") == "numpy"
    with mock.patch.object(utils, "numba", mock.MagicMock()):
        assert parse_engine("auto") == "numba"


def test_parse_engine_fails():
    "Check that the exception is raised for invalid engines"
    with pytest.raises(ValueError):
        parse_engine("some invalid engine")


def test_dummy_jit():
    "Make sure the dummy function raises an exception"

    @dummy_jit(target="cpt")
    def function():
        "Some random function"
        return 0

    with pytest.raises(RuntimeError):
        function()


def test_kdtree():
    "Test that the kdtree returned works for query"
    coords = grid_coordinates((-10, 0, 0, 20), spacing=1)
    for use_pykdtree in [True, False]:
        tree = kdtree(coords, use_pykdtree=use_pykdtree)
        dist, labels = tree.query(np.array([[-10, 0.1]]))
        assert labels.size == 1
        assert labels[0] == 0
        npt.assert_allclose(dist, 0.1)
        if not use_pykdtree:
            assert isinstance(tree, cKDTree)


def test_grid_to_table_order():
    "Check that coordinates are in the right order when converting to tables"
    lon, lat = grid_coordinates(region=(1, 10, -10, -1), shape=(3, 4))
    data = lon**2
    # If the DataArray is created with coords in an order that doesn't match
    # the dims (which is valid), we were getting it wrong because we were
    # relying on the order of the coords instead of dims. This test would have
    # caught that bug.
    grid = xr.DataArray(
        data=data,
        coords={"longitude": lon[0, :], "latitude": lat[:, 0]},
        dims=("latitude", "longitude"),
    ).to_dataset(name="field")
    table = grid_to_table(grid)
    true_lat = [-10, -10, -10, -10, -5.5, -5.5, -5.5, -5.5, -1, -1, -1, -1]
    true_lon = [1, 4, 7, 10, 1, 4, 7, 10, 1, 4, 7, 10]
    true_field = [1, 16, 49, 100, 1, 16, 49, 100, 1, 16, 49, 100]
    npt.assert_allclose(true_lat, table.latitude)
    npt.assert_allclose(true_lon, table.longitude)
    npt.assert_allclose(true_field, table.field)


def test_partition_by_sum_fails_size():
    "Should raise an exception if given more parts than elements."
    with pytest.raises(ValueError) as error:
        partition_by_sum(np.arange(10), 11)
    assert "array of size 10 into 11 parts" in str(error)


def test_partition_by_sum_fails_no_partitions():
    "Should raise an exception if could not find unique partition points"
    with pytest.raises(ValueError) as error:
        partition_by_sum(np.arange(10), 8)
    assert "Could not find partition points" in str(error)


def test_variance_to_weights_pandas_series():
    "A pandas Series input should work even if its backing array is read-only."
    variance = pd.Series([0, 2, 0.2, np.nan])
    weights = variance_to_weights(variance)
    npt.assert_allclose(weights, [1, 0.1, 1, 1])


def test_variance_to_weights_readonly_array():
    "A read-only NumPy array input should still produce normalized weights."
    variance = np.array([0, 2, 0.2, np.nan])
    variance.flags.writeable = False
    weights = variance_to_weights(variance)
    npt.assert_allclose(weights, [1, 0.1, 1, 1])


def test_make_xarray_grid():
    """
    Check if xarray.Dataset is correctly created
    """
    region = (-10, -5, 6, 10)
    spacing = 1
    coordinates = grid_coordinates(region, spacing=spacing)
    data = np.ones_like(coordinates[0])
    grid = make_xarray_grid(coordinates, data, data_names="dummy")
    npt.assert_allclose(grid.easting, [-10, -9, -8, -7, -6, -5])
    npt.assert_allclose(grid.northing, [6, 7, 8, 9, 10])
    npt.assert_allclose(grid.dummy, 1)
    assert grid.dummy.shape == (5, 6)
    # Change dims
    grid = make_xarray_grid(
        coordinates, data, data_names="dummy", dims=("latitude", "longitude")
    )
    npt.assert_allclose(grid.longitude, [-10, -9, -8, -7, -6, -5])
    npt.assert_allclose(grid.latitude, [6, 7, 8, 9, 10])
    npt.assert_allclose(grid.dummy, 1)
    assert grid.dummy.shape == (5, 6)


def test_make_xarray_grid_multiple_data():
    """
    Check if xarray.Dataset with multiple data is correctly created
    """
    region = (-10, -5, 6, 10)
    spacing = 1
    coordinates = grid_coordinates(region, spacing=spacing)
    data_arrays = tuple(i * np.ones_like(coordinates[0]) for i in range(1, 4))
    data_names = list("data_{}".format(i) for i in range(1, 4))
    dataset = make_xarray_grid(coordinates, data_arrays, data_names=data_names)
    npt.assert_allclose(dataset.easting, [-10, -9, -8, -7, -6, -5])
    npt.assert_allclose(dataset.northing, [6, 7, 8, 9, 10])
    for i in range(1, 4):
        npt.assert_allclose(dataset["data_{}".format(i)], i)
        assert dataset["data_{}".format(i)].shape == (5, 6)


def test_make_xarray_grid_no_data():
    """
    Check if the function creates a xarray.Dataset with no data
    """
    region = (-10, -5, 6, 10)
    spacing = 1
    coordinates = grid_coordinates(region, spacing=spacing)
    dataset = make_xarray_grid(coordinates, data=None, data_names=None)
    # Check if no data is present in the grid
    assert len(dataset.data_vars) == 0
    # Check if coordinates are in the grid
    npt.assert_allclose(dataset.easting, [-10, -9, -8, -7, -6, -5])
    npt.assert_allclose(dataset.northing, [6, 7, 8, 9, 10])


def test_make_xarray_grid_extra_coords():
    """
    Check if xarray.Dataset with extra coords is correctly created
    """
    region = (-10, -5, 6, 10)
    spacing = 1
    extra_coords = [1, 2]
    coordinates = grid_coordinates(region, spacing=spacing, extra_coords=extra_coords)
    data = np.ones_like(coordinates[0])
    dataset = make_xarray_grid(
        coordinates,
        data,
        data_names="dummy",
        extra_coords_names=["upward", "time"],
    )
    npt.assert_allclose(dataset.easting, [-10, -9, -8, -7, -6, -5])
    npt.assert_allclose(dataset.northing, [6, 7, 8, 9, 10])
    npt.assert_allclose(dataset.upward, 1)
    npt.assert_allclose(dataset.time, 2)
    npt.assert_allclose(dataset.dummy, 1)
    assert dataset.dummy.shape == (5, 6)
    assert dataset.upward.shape == (5, 6)
    assert dataset.time.shape == (5, 6)


def test_make_xarray_grid_invalid_names():
    """
    Check if errors are raise after invalid data names
    """
    region = (-10, -5, 6, 10)
    spacing = 1
    coordinates = grid_coordinates(region, spacing=spacing)
    # Single data, multiple data_name
    data = np.ones_like(coordinates[0])
    with pytest.raises(ValueError):
        make_xarray_grid(coordinates, data, data_names=["bla_1", "bla_2"])
    # data_names equal to None
    with pytest.raises(ValueError):
        make_xarray_grid(coordinates, data, data_names=None)
    # Multiple data, single data_name
    data = tuple(i * np.ones_like(coordinates[0]) for i in (1, 2))
    with pytest.raises(ValueError):
        make_xarray_grid(coordinates, data, data_names="blabla")


def test_make_xarray_grid_invalid_extra_coords():
    """
    Check if errors are raise after invalid extra coords
    """
    region = (-10, -5, 6, 10)
    spacing = 1
    # No extra coords, extra_coords_name should be ignored
    coordinates = grid_coordinates(region, spacing=spacing)
    data = np.ones_like(coordinates[0])
    make_xarray_grid(coordinates, data, data_names="dummy", extra_coords_names="upward")
    # Single extra coords, extra_coords_name equal to None
    coordinates = grid_coordinates(region, spacing=spacing, extra_coords=1)
    data = np.ones_like(coordinates[0])
    with pytest.raises(ValueError):
        make_xarray_grid(coordinates, data, data_names="dummy", extra_coords_names=None)
    # Multiple extra coords, single extra_coords_name as a str
    coordinates = grid_coordinates(region, spacing=spacing, extra_coords=[1, 2])
    data = np.ones_like(coordinates[0])
    with pytest.raises(ValueError):
        make_xarray_grid(
            coordinates, data, data_names="dummy", extra_coords_names="upward"
        )
    # Multiple extra coords, multiple extra_coords_name but not equal
    coordinates = grid_coordinates(region, spacing=spacing, extra_coords=[1, 2, 3])
    data = np.ones_like(coordinates[0])
    with pytest.raises(ValueError):
        make_xarray_grid(
            coordinates, data, data_names="dummy", extra_coords_names=["upward", "time"]
        )


def test_make_xarray_grid_invalid_2d_coordinates():
    """
    Check if error is raised if invaild 2d coordinates array are passed
    """
    region = (-10, -5, 6, 10)
    spacing = 1
    easting, northing = grid_coordinates(region, spacing=spacing)
    # Change only one element of the easting array
    easting[2, 2] = -1000
    data = np.ones_like(easting)
    with pytest.raises(ValueError):
        make_xarray_grid((easting, northing), data, data_names="dummy")


def test_make_xarray_grid_coordinates_as_1d_arrays():
    """
    Check if it can handle coordinates as 1d-arrays
    """
    region = (-10, -5, 6, 10)
    easting = np.linspace(*region[:2], 6, dtype=float)
    northing = np.linspace(*region[2:], 5, dtype=float)
    data = np.ones((northing.size, easting.size))
    grid = make_xarray_grid((easting, northing), data, data_names="dummy")
    npt.assert_allclose(grid.easting, [-10, -9, -8, -7, -6, -5])
    npt.assert_allclose(grid.northing, [6, 7, 8, 9, 10])
    npt.assert_allclose(grid.dummy, 1)
    assert grid.dummy.shape == (5, 6)


def test_make_xarray_grid_invalid_mixed_coordinates():
    """
    Check if error is raised when horizontal coordinates have mixed dimensions
    """
    region = (-10, -5, 6, 10)
    spacing = 1
    easting, northing = grid_coordinates(region, spacing=spacing)
    data = np.ones_like(easting)
    # easting is 1d, but northing is 2d
    with pytest.raises(ValueError):
        make_xarray_grid((easting[0, :], northing), data, data_names="dummy")
    # northing is 1d, but easting is 2d
    with pytest.raises(ValueError):
        make_xarray_grid((easting, northing[:, 0]), data, data_names="dummy")


def test_meshgrid_to_1d_invalid():
    """
    Check if error is raised after invalid meshgrid
    """
    region = (-10, -5, 6, 10)
    # Modify one element of easting
    easting, northing = grid_coordinates(region=region, spacing=1)
    easting[2, 2] = -9999
    with pytest.raises(ValueError):
        meshgrid_to_1d((easting, northing))
    # Modify one element of northing
    easting, northing = grid_coordinates(region=region, spacing=1)
    northing[2, 3] = -9999
    with pytest.raises(ValueError):
        meshgrid_to_1d((easting, northing))
    # Pass invalid shapes
    easting = np.arange(16).reshape(4, 4)
    northing = np.arange(9).reshape(3, 3)
    with pytest.raises(ValueError):
        meshgrid_to_1d((easting, northing))
    # Pass 1d arrays
    easting = np.linspace(0, 10, 11)
    northing = np.linspace(-4, -4, 9)
    with pytest.raises(ValueError):
        meshgrid_to_1d((easting, northing))


def test_meshgrid_from_1d_invalid():
    """
    Check if error is raised after non 1d arrays passed to meshgrid_from_1d
    """
    coordinates = grid_coordinates(region=(0, 10, -5, 5), shape=(11, 11))
    with pytest.raises(ValueError):
        meshgrid_from_1d(coordinates)


def test_check_ndim_easting_northing():
    """
    Test if check_ndim_easting_northing works as expected
    """
    # Easting and northing as 1d arrays
    easting, northing = scatter_points((-5, 5, 0, 4), 50, random_state=42)
    assert get_ndim_horizontal_coords(easting, northing) == 1
    # Easting and northing as 2d arrays
    easting, northing = grid_coordinates((-5, 5, 0, 4), spacing=1)
    assert get_ndim_horizontal_coords(easting, northing) == 2
    # Check if error is raised after easting and northing with different ndims
    easting = np.linspace(0, 5, 6)
    northing = np.linspace(-5, 5, 16).reshape(4, 4)
    with pytest.raises(ValueError):
        get_ndim_horizontal_coords(easting, northing)


def test_minmax_nans():
    """
    Test minmax handles nans correctly
    """
    assert tuple(map(float, minmax((-1, 100, 1, 2, np.nan)))) == (-1, 100)
    assert tuple(map(float, minmax((np.nan, -3.2, -1, -2, 3.1)))) == (-3.2, 3.1)
    assert np.all(np.isnan(minmax((np.nan, -3, -1, 3), nan=False)))


def test_minmax_percentile():
    """
    Test minmax with percentile option
    """
    data = np.arange(0, 101)

    # test generic functionality
    result = tuple(map(float, minmax(data, min_percentile=0, max_percentile=100)))
    assert result == (0, 100)
    result = tuple(map(float, minmax(data, min_percentile=0.0, max_percentile=100.0)))
    assert result == (0, 100)
    result = tuple(map(float, minmax(data, min_percentile=10, max_percentile=90)))
    assert pytest.approx(result, 0.1) == (10, 90)
    result = tuple(map(float, minmax(data, min_percentile=10.0, max_percentile=90.0)))
    assert pytest.approx(result, 0.1) == (10, 90)

    # test with nans
    data_with_nans = np.append(data, np.nan)
    result = tuple(
        map(float, minmax(data_with_nans, min_percentile=0, max_percentile=100))
    )
    assert result == (0, 100)
    result = tuple(
        map(float, minmax(data_with_nans, min_percentile=10, max_percentile=90))
    )
    assert pytest.approx(result, 0.1) == (10, 90)
    result = tuple(
        map(
            float,
            minmax(data_with_nans, min_percentile=10, max_percentile=90, nan=True),
        )
    )
    assert pytest.approx(result, 0.1) == (10, 90)
    result = minmax(data_with_nans, min_percentile=10, max_percentile=90, nan=False)
    assert np.all(np.isnan(result))

    # test with varying array sizes
    result = tuple(
        map(
            float,
            minmax(
                [0, 1, 2, 3, 4], [[-2, 2], [0, 5]], min_percentile=0, max_percentile=100
            ),
        )
    )
    assert result == (-2, 5)
    result = tuple(
        map(
            float,
            minmax(
                [0, 1, 2, 3, 4], [[-2, 2], [0, 5]], min_percentile=1, max_percentile=99
            ),
        )
    )
    assert pytest.approx(result, 0.1) == (-1.84, 4.92)

    # test invalid percentile types
    with pytest.raises(TypeError):
        minmax(data, min_percentile=None)
    with pytest.raises(TypeError):
        minmax(data, max_percentile=[90])

    # test invalid percentile values
    msg = "'min_percentile'"
    with pytest.raises(ValueError, match=msg):
        minmax(data, min_percentile=99, max_percentile=90)
    msg = "Invalid value for 'min_percentile'"
    with pytest.raises(ValueError, match=msg):
        minmax(data, min_percentile=-10)
    msg = "Invalid value for 'max_percentile'"
    with pytest.raises(ValueError, match=msg):
        minmax(data, max_percentile=110)


def test_maxabs_nans():
    """
    Test maxabs handles nans correctly
    """
    assert float(maxabs((0, 100, 1, 2, np.nan))) == 100
    assert float(maxabs((np.nan, -3.2, -1, -2, 3.1))) == 3.2
    assert np.isnan(maxabs((np.nan, -3, -1, 3), nan=False))


def test_maxabs_percentile():
    """
    Test maxabs with percentile option
    """
    # test generic functionality
    data = np.arange(1, 101)
    assert float(maxabs(data, percentile=100)) == 100
    assert pytest.approx(float(maxabs(data, percentile=90)), 0.1) == 90
    assert pytest.approx(float(maxabs(data, percentile=50)), 0.1) == 50

    # test with nans
    data_with_nans = np.append(data, np.nan)
    assert float(maxabs(data_with_nans, percentile=100)) == 100
    assert pytest.approx(float(maxabs(data_with_nans, percentile=90)), 0.1) == 90
    assert pytest.approx(float(maxabs(data_with_nans, percentile=50)), 0.1) == 50
    assert (
        pytest.approx(float(maxabs(data_with_nans, percentile=90, nan=True)), 0.1) == 90
    )
    assert np.isnan(float(maxabs(data_with_nans, percentile=90, nan=False)))

    # test with varying array sizes
    assert (
        pytest.approx(
            float(maxabs([0, 1, 2, 3, 4], [[-2, 2], [0, 5]], percentile=80)), 0.1
        )
        == 3.4
    )

    # test invalid percentile types
    msg = "Invalid 'percentile' of type"
    with pytest.raises(TypeError, match=msg):
        maxabs(data, percentile="90")
    msg = "Invalid 'percentile' of type"
    with pytest.raises(TypeError, match=msg):
        maxabs(data, percentile=[90])
    msg = "Invalid 'percentile' of type"
    with pytest.raises(TypeError, match=msg):
        maxabs(data, percentile=None)

    # test invalid percentile values
    msg = "Invalid 'percentile' value of"
    with pytest.raises(ValueError, match=msg):
        maxabs(data, percentile=-10)
    msg = "Invalid 'percentile' value of"
    with pytest.raises(ValueError, match=msg):
        maxabs(data, percentile=110)
