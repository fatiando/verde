# pylint: disable=unused-argument,too-many-locals
"""
Test the base classes and their utility functions.
"""
import pytest
import numpy as np
import numpy.testing as npt

from ..base import BaseGridder
from ..base.gridder import get_dims, get_data_names, get_region
from .. import grid_coordinates, scatter_points


def test_get_dims():
    "Tests that get_dims returns the expected results"
    grd = BaseGridder()
    assert get_dims(grd, dims=None) == ('northing', 'easting')
    assert get_dims(grd, dims=('john', 'paul')) == ('john', 'paul')

    grd.coordinate_system = 'geographic'
    assert get_dims(grd, dims=None) == ('latitude', 'longitude')
    assert get_dims(grd, dims=('john', 'paul')) == ('john', 'paul')

    grd.coordinate_system = 'cartesian'
    assert get_dims(grd, dims=None) == ('northing', 'easting')
    assert get_dims(grd, dims=('john', 'paul')) == ('john', 'paul')

    # Make sure the given dims is returned no matter what
    grd.coordinate_system = 'an invalid system'
    assert get_dims(grd, dims=('john', 'paul')) == ('john', 'paul')


def test_get_dims_fails():
    "Check if fails for invalid coordinate system"
    grd = BaseGridder()
    with pytest.raises(ValueError):
        grd.coordinate_system = 'Cartesian'
        get_dims(grd, dims=None)
    with pytest.raises(ValueError):
        grd.coordinate_system = 'Geographic'
        get_dims(grd, dims=None)
    with pytest.raises(ValueError):
        grd.coordinate_system = 'some totally not valid name'
        get_dims(grd, dims=None)


def test_get_data_names():
    "Tests that get_data_names returns the expected results"
    grd = BaseGridder()
    assert get_data_names(grd, data_names=None) == ('scalars',)
    assert get_data_names(grd, data_names=('a', 'b')) == ('a', 'b')

    grd.data_type = 'scalar'
    assert get_data_names(grd, data_names=None) == ('scalars',)
    assert get_data_names(grd, data_names=('a', 'b')) == ('a', 'b')

    grd.data_type = 'vector2d'
    assert get_data_names(grd, data_names=None) == ('east_component',
                                                    'north_component')
    assert get_data_names(grd, data_names=('a', 'b')) == ('a', 'b')

    grd.data_type = 'vector3d'
    assert get_data_names(grd, data_names=None) == ('east_component',
                                                    'north_component',
                                                    'vertical_component')
    assert get_data_names(grd, data_names=('a', 'b')) == ('a', 'b')

    # Make sure the given dims is returned no matter what
    grd.data_type = 'an invalid type'
    assert get_data_names(grd, data_names=('a', 'b')) == ('a', 'b')


def test_get_data_names_fails():
    "Check if fails for invalid data types"
    grd = BaseGridder()
    with pytest.raises(ValueError):
        grd.data_type = 'Scalars'
        get_data_names(grd, data_names=None)
    with pytest.raises(ValueError):
        grd.data_type = 'Vector3d'
        get_data_names(grd, data_names=None)
    with pytest.raises(ValueError):
        grd.data_type = 'some totally not valid name'
        get_data_names(grd, data_names=None)


def test_get_region():
    "Check if get_region finds the correct region"
    grd = BaseGridder()
    assert get_region(grd, region=(1, 2, 3, 4)) == (1, 2, 3, 4)
    with pytest.raises(ValueError):
        get_region(grd, region=None)
    grd.region_ = (5, 6, 7, 8)
    assert get_region(grd, region=None) == (5, 6, 7, 8)
    assert get_region(grd, region=(1, 2, 3, 4)) == (1, 2, 3, 4)


class PolyGridder(BaseGridder):
    "A test gridder"

    def __init__(self, degree=1):
        self.degree = degree

    def fit(self, easting, northing, data):
        "Fit an easting polynomial"
        ndata = data.size
        nparams = self.degree + 1
        jac = np.zeros((ndata, nparams))
        for j in range(nparams):
            jac[:, j] = easting.ravel()**j
        self.coefs_ = np.linalg.solve(jac.T.dot(jac), jac.T.dot(data.ravel()))
        return self

    def predict(self, easting, northing):
        "Predict the data"
        return np.sum(cof*easting**deg for deg, cof in enumerate(self.coefs_))


def test_basegridder():
    "Test basic functionality of BaseGridder"

    with pytest.raises(NotImplementedError):
        BaseGridder().predict(None, None)

    grd = PolyGridder()
    assert repr(grd) == 'PolyGridder(degree=1)'
    grd.degree = 2
    assert repr(grd) == 'PolyGridder(degree=2)'

    region = (0, 10, -10, -5)
    shape = (50, 30)
    angular, linear = 2, 100
    east, north = scatter_points(region, 1000, random_state=0)
    data = angular*east + linear
    grd = PolyGridder().fit(east, north, data)

    with pytest.raises(ValueError):
        # A region should be given because it hasn't been assigned
        grd.grid()

    east_true, north_true = grid_coordinates(region, shape)
    data_true = angular*east_true + linear
    grid = grd.grid(region, shape)

    npt.assert_allclose(grd.coefs_, [linear, angular])
    npt.assert_allclose(grid.scalars.values, data_true)
    npt.assert_allclose(grid.easting.values, east_true[0, :])
    npt.assert_allclose(grid.northing.values, north_true[:, 0])
    npt.assert_allclose(grd.scatter(region, 1000, random_state=0).scalars,
                        data)
    npt.assert_allclose(grd.profile((0, 0), (10, 0), 30).scalars,
                        angular*east_true[0, :] + linear)


def test_basegridder_projection():
    "Test basic functionality of BaseGridder when passing in a projection"

    region = (0, 10, -10, -5)
    shape = (50, 30)
    angular, linear = 2, 100
    east, north = scatter_points(region, 1000, random_state=0)
    data = angular*east + linear
    east_true, north_true = grid_coordinates(region, shape)
    data_true = angular*east_true + linear
    grd = PolyGridder().fit(east, north, data)

    # Lets say we want to specify the region for a grid using a coordinate
    # system that is lon/2, lat/2.
    def proj(lon, lat):
        "Project from the new coordinates to the original"
        return (lon*2, lat*2)

    proj_region = [i/2 for i in region]
    grid = grd.grid(proj_region, shape, projection=proj)
    scat = grd.scatter(proj_region, 1000, random_state=0, projection=proj)
    prof = grd.profile((0, 0), (5, 0), 30, projection=proj)

    npt.assert_allclose(grd.coefs_, [linear, angular])
    npt.assert_allclose(grid.scalars.values, data_true)
    npt.assert_allclose(grid.easting.values, east_true[0, :]/2)
    npt.assert_allclose(grid.northing.values, north_true[:, 0]/2)
    npt.assert_allclose(scat.scalars, data)
    npt.assert_allclose(prof.scalars, angular*east_true[0, :] + linear)
