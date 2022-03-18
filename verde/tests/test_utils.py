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
import pytest
import xarray as xr
from scipy.spatial import cKDTree

from .. import utils
from ..coordinates import grid_coordinates, scatter_points
from ..utils import (
    dummy_jit,
    get_ndim_horizontal_coords,
    grid_to_table,
    kdtree,
    make_xarray_grid,
    meshgrid_from_1d,
    meshgrid_to_1d,
    parse_engine,
    partition_by_sum,
)


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
