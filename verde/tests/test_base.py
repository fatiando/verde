"""
Test the base classes and their utility functions.
"""
import pytest

from ..base import get_dimensions, BaseGridder


def test_get_dimensions():
    "Tests that get_dimensions returns the expected results"
    grd = BaseGridder()
    assert get_dimensions(grd) == ('northing', 'easting')
    grd.coordinate_system = 'geographic'
    assert get_dimensions(grd) == ('latitude', 'longitude')
    grd.coordinate_system = 'cartesian'
    assert get_dimensions(grd) == ('northing', 'easting')


def test_get_dimensions_fails():
    "Check if fails for invalid coordinate system"
    grd = BaseGridder()
    with pytest.raises(ValueError):
        grd.coordinate_system = 'Cartesian'
        get_dimensions(grd)
    with pytest.raises(ValueError):
        grd.coordinate_system = 'Geographic'
        get_dimensions(grd)
    with pytest.raises(ValueError):
        grd.coordinate_system = 'some totally not valid name'
        get_dimensions(grd)
