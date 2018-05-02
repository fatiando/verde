"""
Test the base classes and their utility functions.
"""
import pytest

from ..base import BaseGridder
from ..base.gridder import get_dims, get_data_names, get_region


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
