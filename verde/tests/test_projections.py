# Copyright (c) 2017 The Verde Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test the projection functions.
"""
import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr

from ..projections import project_grid
from ..scipygridder import ScipyGridder


def projection(longitude, latitude):
    "Dummy projection"
    return longitude**2, latitude**2


@pytest.mark.parametrize(
    "method",
    ["nearest", "linear", "cubic", ScipyGridder("nearest")],
    ids=["nearest", "linear", "cubic", "gridder"],
)
def test_project_grid(method):
    "Use a simple projection to test that the output is as expected"
    shape = (50, 40)
    lats = np.linspace(2, 10, shape[1])
    lons = np.linspace(-10, 2, shape[0])
    data = np.ones(shape, dtype="float")
    grid = xr.DataArray(data, coords=[lons, lats], dims=("latitude", "longitude"))
    proj = project_grid(grid, projection, method=method)
    assert proj.dims == ("northing", "easting")
    assert proj.name == "scalars"
    assert proj.shape == shape
    # Check the grid spacing is constant
    spacing_east = proj.easting[1:] - proj.easting[0:-1]
    npt.assert_allclose(spacing_east, spacing_east[0])
    spacing_north = proj.northing[1:] - proj.northing[0:-1]
    npt.assert_allclose(spacing_north, spacing_north[0])
    # Check that the values are all 1
    npt.assert_allclose(proj.values[~np.isnan(proj.values)], 1)


def test_project_grid_name():
    "Check that grid name is kept"
    shape = (50, 40)
    lats = np.linspace(2, 10, shape[1])
    lons = np.linspace(-10, 2, shape[0])
    data = np.ones(shape, dtype="float")
    grid = xr.DataArray(
        data, coords=[lons, lats], dims=("latitude", "longitude"), name="yara"
    )
    proj = project_grid(grid, projection)
    assert proj.name == "yara"
    assert proj.dims == ("northing", "easting")
    assert proj.shape == shape
    # Check the grid spacing is constant
    spacing_east = proj.easting[1:] - proj.easting[0:-1]
    npt.assert_allclose(spacing_east, spacing_east[0])
    spacing_north = proj.northing[1:] - proj.northing[0:-1]
    npt.assert_allclose(spacing_north, spacing_north[0])
    # Check that the values are all 1
    npt.assert_allclose(proj.values[~np.isnan(proj.values)], 1)


@pytest.mark.parametrize("antialias", [True, False])
def test_project_grid_antialias(antialias):
    "Check if antialias is being used"
    shape = (50, 40)
    lats = np.linspace(2, 10, shape[1])
    lons = np.linspace(-10, 2, shape[0])
    data = np.ones(shape, dtype="float")
    grid = xr.DataArray(data, coords=[lons, lats], dims=("latitude", "longitude"))
    proj = project_grid(grid, projection, antialias=antialias)
    if antialias:
        assert "BlockReduce" in proj.attrs["metadata"]
    else:
        assert "BlockReduce" not in proj.attrs["metadata"]
    assert proj.dims == ("northing", "easting")
    assert proj.name == "scalars"
    assert proj.shape == shape
    # Check the grid spacing is constant
    spacing_east = proj.easting[1:] - proj.easting[0:-1]
    npt.assert_allclose(spacing_east, spacing_east[0])
    spacing_north = proj.northing[1:] - proj.northing[0:-1]
    npt.assert_allclose(spacing_north, spacing_north[0])
    # Check that the values are all 1
    npt.assert_allclose(proj.values[~np.isnan(proj.values)], 1)


def test_project_grid_fails_dataset():
    "Should raise an exception when given a Datatset"
    shape = (50, 40)
    lats = np.linspace(2, 10, shape[1])
    lons = np.linspace(-10, 2, shape[0])
    data = np.ones(shape, dtype="float")
    grid = xr.DataArray(data, coords=[lons, lats], dims=("latitude", "longitude"))
    grid = grid.to_dataset(name="scalars")
    with pytest.raises(ValueError):
        project_grid(grid, projection)


@pytest.mark.parametrize("ndims", [1, 3])
def test_project_grid_fails_dimensions(ndims):
    "Should raise an exception when given more or less than 2 dimensions"
    shape = (10, 20, 12)
    coords = [
        np.linspace(-10, 2, shape[0]),
        np.linspace(2, 10, shape[1]),
        np.linspace(0, 100, shape[2]),
    ]
    dims = ("height", "latitude", "longitude")
    data = np.ones(shape[:ndims], dtype="float")
    grid = xr.DataArray(data, coords=coords[:ndims], dims=dims[:ndims])
    with pytest.raises(ValueError):
        project_grid(grid, projection)
