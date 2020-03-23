# pylint: disable=unused-argument,too-many-locals,protected-access
"""
Test the base classes and their utility functions.
"""
import numpy as np
import numpy.testing as npt
import pytest

from ..base.utils import check_fit_input, check_coordinates
from ..base.base_classes import (
    BaseGridder,
    get_data_names,
    get_instance_region,
)
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
    assert get_data_names(data1, data_names=None) == ("scalars",)
    assert get_data_names(data2, data_names=None) == (
        "east_component",
        "north_component",
    )
    assert get_data_names(data3, data_names=None) == (
        "east_component",
        "north_component",
        "vertical_component",
    )
    # Test custom names
    assert get_data_names(data1, data_names=("a",)) == ("a",)
    assert get_data_names(data2, data_names=("a", "b")) == ("a", "b")
    assert get_data_names(data3, data_names=("a", "b", "c")) == ("a", "b", "c")


def test_get_data_names_fails():
    "Check if fails for invalid data types"
    with pytest.raises(ValueError):
        get_data_names(tuple([np.arange(5)] * 4), data_names=None)
    with pytest.raises(ValueError):
        get_data_names(tuple([np.arange(5)] * 2), data_names=("meh",))


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

    def fit(self, coordinates, data, weights=None):
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
    assert repr(grd) == "PolyGridder(degree=1)"
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

    coordinates_true = grid_coordinates(region, shape)
    data_true = angular * coordinates_true[0] + linear
    grid = grd.grid(region, shape)
    prof = grd.profile((0, -10), (10, -10), 30)

    npt.assert_allclose(grd.coefs_, [linear, angular])
    npt.assert_allclose(grid.scalars.values, data_true)
    npt.assert_allclose(grid.easting.values, coordinates_true[0][0, :])
    npt.assert_allclose(grid.northing.values, coordinates_true[1][:, 0])
    npt.assert_allclose(grd.scatter(region, 1000, random_state=0).scalars, data)
    npt.assert_allclose(
        prof.scalars, angular * coordinates_true[0][0, :] + linear,
    )
    npt.assert_allclose(prof.easting, coordinates_true[0][0, :])
    npt.assert_allclose(prof.northing, coordinates_true[1][0, :])
    npt.assert_allclose(prof.distance, coordinates_true[0][0, :])


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
    grid = grd.grid(region, shape, projection=proj)
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
