# Copyright (c) 2017 The Verde Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test the base classes and their utility functions.
"""
import warnings

import numpy as np
import numpy.testing as npt
import pytest

from ..base.base_classes import (
    BaseBlockCrossValidator,
    BaseGridder,
    get_instance_region,
)
from ..base.least_squares import least_squares
from ..base.utils import check_coordinates, check_fit_input
from ..coordinates import grid_coordinates, scatter_points


def test_check_coordinates():
    "Should raise a ValueError is the coordinates have different shapes."
    # Should not raise an error
    check_coordinates([np.arange(10), np.arange(10)])
    check_coordinates([np.arange(10).reshape((5, 2)), np.arange(10).reshape((5, 2))])
    # Should raise an error
    with pytest.raises(ValueError):
        check_coordinates([np.arange(10), np.arange(10).reshape((5, 2))])
    with pytest.raises(ValueError):
        check_coordinates(
            [np.arange(10).reshape((2, 5)), np.arange(10).reshape((5, 2))]
        )


def test_get_dims():
    "Tests that get_dims returns the expected results"
    gridder = BaseGridder()
    assert gridder._get_dims(dims=None) == ("northing", "easting")
    assert gridder._get_dims(dims=("john", "paul")) == ("john", "paul")
    gridder.dims = ("latitude", "longitude")
    assert gridder._get_dims(dims=None) == ("latitude", "longitude")


def test_get_data_names():
    "Tests that get_data_names returns the expected results"
    data1 = (np.arange(10),)
    data2 = tuple([np.arange(10)] * 2)
    data3 = tuple([np.arange(10)] * 3)
    # Test the default names
    gridder = BaseGridder()
    assert gridder._get_data_names(data1, data_names=None) == ("scalars",)
    assert gridder._get_data_names(data2, data_names=None) == (
        "east_component",
        "north_component",
    )
    assert gridder._get_data_names(data3, data_names=None) == (
        "east_component",
        "north_component",
        "vertical_component",
    )
    # Test custom names
    assert gridder._get_data_names(data1, data_names=("a",)) == ("a",)
    assert gridder._get_data_names(data2, data_names=("a", "b")) == ("a", "b")
    assert gridder._get_data_names(data3, data_names=("a", "b", "c")) == ("a", "b", "c")


def test_get_data_names_fails():
    "Check if fails for invalid data types"
    gridder = BaseGridder()
    with pytest.raises(ValueError):
        gridder._get_data_names(tuple([np.arange(5)] * 4), data_names=None)
    with pytest.raises(ValueError):
        gridder._get_data_names(tuple([np.arange(5)] * 2), data_names=("meh",))


def test_get_instance_region():
    "Check if get_instance_region finds the correct region"
    grd = BaseGridder()
    assert get_instance_region(grd, region=(1, 2, 3, 4)) == (1, 2, 3, 4)
    with pytest.raises(ValueError):
        get_instance_region(grd, region=None)
    grd.region_ = (5, 6, 7, 8)
    assert get_instance_region(grd, region=None) == (5, 6, 7, 8)
    assert get_instance_region(grd, region=(1, 2, 3, 4)) == (1, 2, 3, 4)


class PolyGridder(BaseGridder):
    "A test gridder"

    def __init__(self, degree=1):
        super().__init__()
        self.degree = degree

    def fit(self, coordinates, data, weights=None):  # noqa: U100
        "Fit an easting polynomial"
        ndata = data.size
        nparams = self.degree + 1
        jac = np.zeros((ndata, nparams))
        for j in range(nparams):
            jac[:, j] = coordinates[0].ravel() ** j
        self.coefs_ = np.linalg.solve(jac.T.dot(jac), jac.T.dot(data.ravel()))
        return self

    def predict(self, coordinates):
        "Predict the data"
        return sum(cof * coordinates[0] ** deg for deg, cof in enumerate(self.coefs_))


def test_basegridder():
    "Test basic functionality of BaseGridder"

    with pytest.raises(NotImplementedError):
        BaseGridder().predict(None)
    with pytest.raises(NotImplementedError):
        BaseGridder().fit(None, None)

    grd = PolyGridder()
    assert repr(grd) == "PolyGridder()"
    grd.degree = 2
    assert repr(grd) == "PolyGridder(degree=2)"

    region = (0, 10, -10, -5)
    shape = (50, 30)
    angular, linear = 2, 100
    coordinates = scatter_points(region, 1000, random_state=0)
    data = angular * coordinates[0] + linear
    grd = PolyGridder().fit(coordinates, data)

    with pytest.raises(ValueError):
        # A region should be given because it hasn't been assigned
        grd.grid()

    npt.assert_allclose(grd.coefs_, [linear, angular])

    # Check predictions by comparing against expected values
    coordinates_true = grid_coordinates(region, shape=shape)
    data_true = angular * coordinates_true[0] + linear
    # Grid passing region and shape
    grids = []
    grids.append(grd.grid(region=region, shape=shape))
    # Grid passing grid coordinates
    grids.append(grd.grid(coordinates=coordinates_true))
    # Grid passing grid coordinates as 1d arrays
    grids.append(grd.grid(coordinates=tuple(np.unique(c) for c in coordinates_true)))
    # Grid on profile
    prof = grd.profile((0, -10), (10, -10), 30)
    # Grid on scatter
    scat = grd.scatter(region=region, size=1000, random_state=0)

    for grid in grids:
        npt.assert_allclose(grid.scalars.values, data_true)
        npt.assert_allclose(grid.easting.values, coordinates_true[0][0, :])
        npt.assert_allclose(grid.northing.values, coordinates_true[1][:, 0])
    npt.assert_allclose(scat.scalars, data)
    npt.assert_allclose(
        prof.scalars,
        angular * coordinates_true[0][0, :] + linear,
    )
    npt.assert_allclose(prof.easting, coordinates_true[0][0, :])
    npt.assert_allclose(prof.northing, coordinates_true[1][0, :])
    npt.assert_allclose(prof.distance, coordinates_true[0][0, :])


def test_basegridder_data_names():
    """
    Test default values for data_names
    """
    region = (0, 10, -10, -5)
    shape = (50, 30)
    angular, linear = 2, 100
    coordinates = scatter_points(region, 1000, random_state=0)
    data = angular * coordinates[0] + linear
    grd = PolyGridder().fit(coordinates, data)
    grid = grd.grid(region=region, shape=shape)
    prof = grd.profile((0, -10), (10, -10), 30)
    # Check if default name for data_names was applied correctly
    assert "scalars" in grid
    assert "scalars" in prof


def test_basegridder_projection():
    "Test basic functionality of BaseGridder when passing in a projection"

    # Lets say the projection is doubling the coordinates
    def proj(lon, lat, inverse=False):
        "Project from the new coordinates to the original"
        if inverse:
            return (lon / 2, lat / 2)
        return (lon * 2, lat * 2)

    # Values in "geographic" coordinates
    region = (0, 10, -10, -5)
    shape = (51, 31)
    angular, linear = 2, 100
    coordinates = scatter_points(region, 1000, random_state=0)
    data = angular * coordinates[0] + linear
    # Project before passing to our Cartesian gridder
    grd = PolyGridder().fit(proj(coordinates[0], coordinates[1]), data)

    # Check the estimated coefficients
    # The grid is estimated in projected coordinates (which are twice as large)
    # so the rate of change (angular) should be half to get the same values.
    npt.assert_allclose(grd.coefs_, [linear, angular / 2])

    # The actual values for a grid
    coordinates_true = grid_coordinates(region, shape)
    data_true = angular * coordinates_true[0] + linear

    # Check the scatter
    scat = grd.scatter(region, 1000, random_state=0, projection=proj)
    npt.assert_allclose(scat.scalars, data)
    npt.assert_allclose(scat.easting, coordinates[0])
    npt.assert_allclose(scat.northing, coordinates[1])

    # Check the grid
    grid = grd.grid(region=region, shape=shape, projection=proj)
    npt.assert_allclose(grid.scalars.values, data_true)
    npt.assert_allclose(grid.easting.values, coordinates_true[0][0, :])
    npt.assert_allclose(grid.northing.values, coordinates_true[1][:, 0])

    # Check the profile
    prof = grd.profile(
        (region[0], region[-1]), (region[1], region[-1]), shape[1], projection=proj
    )
    npt.assert_allclose(prof.scalars, data_true[-1, :])
    # Coordinates should still be evenly spaced since the projection is a
    # multiplication.
    npt.assert_allclose(prof.easting, coordinates_true[0][0, :])
    npt.assert_allclose(prof.northing, coordinates_true[1][-1, :])
    # Distance should still be in the projected coordinates. If the projection
    # is from geographic, we shouldn't be returning distances in degrees but in
    # projected meters. The distances will be evenly spaced in unprojected
    # coordinates.
    distance_true = np.linspace(region[0] * 2, region[1] * 2, shape[1])
    npt.assert_allclose(prof.distance, distance_true)


def test_basegridder_extra_coords():
    "Test if BaseGridder adds extra_coords to generated outputs"

    grd = PolyGridder()
    region = (0, 10, -10, -5)
    angular, linear = 2, 100
    coordinates = scatter_points(region, 1000, random_state=0)
    data = angular * coordinates[0] + linear
    grd = PolyGridder().fit(coordinates, data)

    # Test grid with a single extra coord
    extra_coords = 9
    grid = grd.grid(region=region, spacing=1, extra_coords=extra_coords)
    assert "extra_coord" in grid.coords
    assert grid.coords["extra_coord"].dims == ("northing", "easting")
    npt.assert_allclose(grid.coords["extra_coord"], extra_coords)

    # Test grid with multiple extra coords
    extra_coords = [9, 18, 27]
    grid = grd.grid(region=region, spacing=1, extra_coords=extra_coords)
    extra_coord_names = ["extra_coord", "extra_coord_1", "extra_coord_2"]
    for name, coord in zip(extra_coord_names, extra_coords):
        assert name in grid.coords
        assert grid.coords[name].dims == ("northing", "easting")
        npt.assert_allclose(grid.coords[name], coord)

    # Test scatter with a single extra coord
    extra_coords = 9
    scat = grd.scatter(region, 1000, extra_coords=extra_coords)
    assert "extra_coord" in scat.columns
    npt.assert_allclose(scat["extra_coord"], extra_coords)

    # Test scatter with multiple extra coord
    extra_coords = [9, 18, 27]
    scat = grd.scatter(region, 1000, extra_coords=extra_coords)
    extra_coord_names = ["extra_coord", "extra_coord_1", "extra_coord_2"]
    for name, coord in zip(extra_coord_names, extra_coords):
        assert name in scat.columns
        npt.assert_allclose(scat[name], coord)

    # Test profile with a single extra coord
    extra_coords = 9
    prof = grd.profile(
        (region[0], region[-1]),
        (region[1], region[-1]),
        51,
        extra_coords=extra_coords,
    )
    assert "extra_coord" in prof.columns
    npt.assert_allclose(prof["extra_coord"], extra_coords)

    # Test profile with multiple extra coord
    extra_coords = [9, 18, 27]
    prof = grd.profile(
        (region[0], region[-1]),
        (region[1], region[-1]),
        51,
        extra_coords=extra_coords,
    )
    extra_coord_names = ["extra_coord", "extra_coord_1", "extra_coord_2"]
    for name, coord in zip(extra_coord_names, extra_coords):
        assert name in prof.columns
        npt.assert_allclose(prof[name], coord)


def test_basegridder_projection_multiple_coordinates():
    "Test BaseGridder when passing in a projection with multiple coordinates"

    # Lets say the projection is doubling the coordinates
    def proj(lon, lat, inverse=False):
        "Project from the new coordinates to the original"
        if inverse:
            return (lon / 2, lat / 2)
        return (lon * 2, lat * 2)

    # Values in "geographic" coordinates
    region = (0, 10, -10, -5)
    shape = (51, 31)
    angular, linear = 2, 100
    coordinates = scatter_points(region, 1000, random_state=0, extra_coords=(1, 2))
    data = angular * coordinates[0] + linear
    # Project before passing to our Cartesian gridder
    proj_coordinates = proj(coordinates[0], coordinates[1]) + coordinates[2:]
    grd = PolyGridder().fit(proj_coordinates, data)

    # Check the estimated coefficients
    # The grid is estimated in projected coordinates (which are twice as large)
    # so the rate of change (angular) should be half to get the same values.
    npt.assert_allclose(grd.coefs_, [linear, angular / 2])

    # The actual values for a grid
    coordinates_true = grid_coordinates(region, shape=shape, extra_coords=(13, 17))
    data_true = angular * coordinates_true[0] + linear

    # Check the scatter
    scat = grd.scatter(
        region, 1000, random_state=0, projection=proj, extra_coords=(13, 17)
    )
    npt.assert_allclose(scat.scalars, data)
    npt.assert_allclose(scat.easting, coordinates[0])
    npt.assert_allclose(scat.northing, coordinates[1])

    # Check the grid
    grid = grd.grid(region=region, shape=shape, projection=proj, extra_coords=(13, 17))
    npt.assert_allclose(grid.scalars.values, data_true)
    npt.assert_allclose(grid.easting.values, coordinates_true[0][0, :])
    npt.assert_allclose(grid.northing.values, coordinates_true[1][:, 0])

    # Check the profile
    prof = grd.profile(
        (region[0], region[-1]),
        (region[1], region[-1]),
        shape[1],
        projection=proj,
        extra_coords=(13, 17),
    )
    npt.assert_allclose(prof.scalars, data_true[-1, :])
    # Coordinates should still be evenly spaced since the projection is a
    # multiplication.
    npt.assert_allclose(prof.easting, coordinates_true[0][0, :])
    npt.assert_allclose(prof.northing, coordinates_true[1][-1, :])
    # Distance should still be in the projected coordinates. If the projection
    # is from geographic, we shouldn't be returning distances in degrees but in
    # projected meters. The distances will be evenly spaced in unprojected
    # coordinates.
    distance_true = np.linspace(region[0] * 2, region[1] * 2, shape[1])
    npt.assert_allclose(prof.distance, distance_true)


def test_basegridder_grid_invalid_arguments():
    """
    Test if errors and warnings are raised on invalid arguments to grid method
    """
    region = (0, 10, -10, -5)
    angular, linear = 2, 100
    coordinates = scatter_points(region, 1000, random_state=0, extra_coords=(1, 2))
    data = angular * coordinates[0] + linear
    grd = PolyGridder().fit(coordinates, data)
    # Check error is raised if coordinates and shape are passed
    grid_coords = (np.linspace(*region[:2], 11), np.linspace(*region[2:], 7))
    with pytest.raises(ValueError):
        grd.grid(coordinates=grid_coords, shape=(30, 30))
    # Check error is raised if coordinates and spacing are passed
    with pytest.raises(ValueError):
        grd.grid(coordinates=grid_coords, spacing=10)
    # Check error is raised if both coordinates and region are passed
    with pytest.raises(ValueError):
        grd.grid(coordinates=grid_coords, region=region)
    # Check if FutureWarning is raised after passing region, spacing or shape
    with warnings.catch_warnings(record=True) as warns:
        grd.grid(region=region, shape=(4, 4))
        assert len(warns) == 1
        assert issubclass(warns[0].category, FutureWarning)
    with warnings.catch_warnings(record=True) as warns:
        grd.grid(region=region, spacing=1)
        assert len(warns) == 1
        assert issubclass(warns[0].category, FutureWarning)


def test_check_fit_input():
    "Make sure no exceptions are raised for standard cases"
    size = 20
    data = np.arange(size)
    coords = (np.arange(size), np.arange(size))
    weights = np.arange(size)
    check_fit_input(coords, data, None)
    check_fit_input(coords, data, weights)
    check_fit_input(coords, (data, data), None)
    check_fit_input(coords, (data, data), (weights, weights))
    check_fit_input(coords, (data, data), (None, None))
    check_fit_input(coords, (data,), (None,))
    check_fit_input(coords, (data,), (weights,))


def test_check_fit_input_fails_coordinates():
    "Test the failing conditions for check_fit_input"
    coords = (np.arange(20), np.arange(20))
    data = np.arange(30)
    with pytest.raises(ValueError):
        check_fit_input(coords, data, weights=None)


def test_check_fit_input_fails_weights():
    "Test the failing conditions for check_fit_input"
    data = np.arange(20)
    coords = (data, data)
    weights = np.arange(30)
    with pytest.raises(ValueError):
        check_fit_input(coords, data, weights)
    with pytest.raises(ValueError):
        check_fit_input(coords, (data, data), weights)


class DummyCrossValidator(BaseBlockCrossValidator):
    """
    Dummy class to test the base cross-validator.
    """

    def _iter_test_indices(self, X=None, y=None, groups=None):  # noqa: U100,N803
        """
        Yields a list of indices for the entire X.
        """
        yield list(range(X.shape[0]))


def test_baseblockedcrossvalidator_n_splits():
    "Make sure get_n_splits returns the correct value"
    cv = DummyCrossValidator(spacing=1, n_splits=14)
    assert cv.get_n_splits() == 14


def test_baseblockedcrossvalidator_fails_spacing_shape():
    "Should raise an exception if not given spacing or shape."
    with pytest.raises(ValueError):
        DummyCrossValidator()


def test_baseblockedcrossvalidator_fails_data_shape():
    "Should raise an exception if the X array doesn't have 2 columns."
    cv = DummyCrossValidator(spacing=1)
    with pytest.raises(ValueError):
        next(cv.split(np.ones(shape=(10, 4))))
    with pytest.raises(ValueError):
        next(cv.split(np.ones(shape=(10, 1))))


def test_least_squares_copy_jacobian():
    """
    Test if Jacobian matrix is copied or scaled inplace
    """
    jacobian = np.identity(5)
    original_jacobian = jacobian.copy()
    data = np.array([1, 2, 3, 4, 5], dtype=float)
    least_squares(jacobian, data, weights=None, copy_jacobian=True)
    npt.assert_allclose(jacobian, original_jacobian)
    least_squares(jacobian, data, weights=None)
    assert not np.allclose(jacobian, original_jacobian)
