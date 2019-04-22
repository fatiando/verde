"""
Test the coordinate generation functions
"""
import numpy as np
import numpy.testing as npt
import pytest

from ..coordinates import (
    check_region,
    spacing_to_shape,
    profile_coordinates,
    grid_coordinates,
    inside,
    _latlon_continuity,
)


def test_spacing_to_shape():
    "Check that correct spacing and region are returned"
    region = (-10, 0, 0, 5)

    shape, new_region = spacing_to_shape(region, spacing=2.5, adjust="spacing")
    npt.assert_allclose(shape, (3, 5))
    npt.assert_allclose(new_region, region)

    shape, new_region = spacing_to_shape(region, spacing=(2.5, 2), adjust="spacing")
    npt.assert_allclose(shape, (3, 6))
    npt.assert_allclose(new_region, region)

    shape, new_region = spacing_to_shape(region, spacing=2.6, adjust="spacing")
    npt.assert_allclose(shape, (3, 5))
    npt.assert_allclose(new_region, region)

    shape, new_region = spacing_to_shape(region, spacing=2.4, adjust="spacing")
    npt.assert_allclose(shape, (3, 5))
    npt.assert_allclose(new_region, region)

    shape, new_region = spacing_to_shape(region, spacing=(2.4, 1.9), adjust="spacing")
    npt.assert_allclose(shape, (3, 6))
    npt.assert_allclose(new_region, region)

    shape, new_region = spacing_to_shape(region, spacing=2.6, adjust="region")
    npt.assert_allclose(shape, (3, 5))
    npt.assert_allclose(new_region, (-10, 0.4, 0, 5.2))

    shape, new_region = spacing_to_shape(region, spacing=(2.6, 2.4), adjust="region")
    npt.assert_allclose(shape, (3, 5))
    npt.assert_allclose(new_region, (-10, -0.4, 0, 5.2))


def test_spacing_to_shape_fails():
    "Should fail if more than 2 spacings are given"
    with pytest.raises(ValueError):
        spacing_to_shape((0, 1, 0, 1), (1, 2, 3), adjust="region")


def test_grid_coordinates_fails():
    "Check failures for invalid arguments"
    region = (0, 1, 0, 10)
    shape = (10, 20)
    spacing = 0.5
    # Make sure it doesn't fail for these parameters
    grid_coordinates(region, shape)
    grid_coordinates(region, spacing=spacing)

    with pytest.raises(ValueError):
        grid_coordinates(region, shape=shape, spacing=spacing)
    with pytest.raises(ValueError):
        grid_coordinates(region, shape=None, spacing=None)
    with pytest.raises(ValueError):
        grid_coordinates(region, spacing=spacing, adjust="invalid adjust")


def test_check_region():
    "Make sure an exception is raised for bad regions"
    with pytest.raises(ValueError):
        check_region([])
    with pytest.raises(ValueError):
        check_region([1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        check_region([1, 2, 3])
    with pytest.raises(ValueError):
        check_region([1, 2, 3, 1])
    with pytest.raises(ValueError):
        check_region([2, 1, 3, 4])
    with pytest.raises(ValueError):
        check_region([-1, -2, -4, -3])
    with pytest.raises(ValueError):
        check_region([-2, -1, -2, -3])


def test_profile_coordiantes_fails():
    "Should raise an exception for invalid input"
    with pytest.raises(ValueError):
        profile_coordinates((0, 1), (1, 2), size=0)
    with pytest.raises(ValueError):
        profile_coordinates((0, 1), (1, 2), size=-10)


def test_latlon_continuity():
    "Test continuous boundary conditions in geographic coordinates."
    # Define longitude coordinates around the globe for [0, 360) and [-180, 180)
    longitude_360 = np.linspace(0, 350, 36)
    longitude_180 = np.hstack((longitude_360[:18], longitude_360[18:] - 360))
    # Check w, e in [0, 360)
    w, e = 10, 20
    for longitude in [longitude_360, longitude_180]:
        w_new, e_new, longitude_new = _latlon_continuity(w, e, longitude)
        assert w_new == 10
        assert e_new == 20
        npt.assert_allclose(longitude_new, longitude_360)
    # Check w, e in [-180, 180)
    w, e = -20, 20
    for longitude in [longitude_360, longitude_180]:
        w_new, e_new, longitude_new = _latlon_continuity(w, e, longitude)
        assert w_new == -20
        assert e_new == 20
        npt.assert_allclose(longitude_new, longitude_180)
    # Check angle greater than 180
    w, e = 0, 200
    for longitude in [longitude_360, longitude_180]:
        w_new, e_new, longitude_new = _latlon_continuity(w, e, longitude)
        assert w_new == 0
        assert e_new == 200
        npt.assert_allclose(longitude_new, longitude_360)
    w, e = -160, 160
    for longitude in [longitude_360, longitude_180]:
        w_new, e_new, longitude_new = _latlon_continuity(w, e, longitude)
        assert w_new == -160
        assert e_new == 160
        npt.assert_allclose(longitude_new, longitude_180)
    # Check overlapping regions
    w, e = -200, 200
    for longitude in [longitude_360, longitude_180]:
        w_new, e_new, longitude_new = _latlon_continuity(w, e, longitude)
        assert w_new == 160
        assert e_new == 200
        npt.assert_allclose(longitude_new, longitude_360)
    w, e = 200, -200
    for longitude in [longitude_360, longitude_180]:
        w_new, e_new, longitude_new = _latlon_continuity(w, e, longitude)
        assert w_new == -160
        assert e_new == 160
        npt.assert_allclose(longitude_new, longitude_180)


def test_inside_latlon_0_360():
    "Check if inside gets points properly with geographic coordinates on [0, 360]"
    # Define longitude coordinates on 0, 360
    longitude = np.linspace(0, 350, 36)
    latitude = np.linspace(-90, 90, 19)
    longitude, latitude = np.meshgrid(longitude, latitude)
    # Check region longitudes in 0, 360
    region = 20, 40, -10, 10
    are_inside = inside([longitude, latitude], region, latlon=True)
    longitude_cut, latitude_cut = longitude[are_inside], latitude[are_inside]
    assert longitude_cut.size == 9
    assert latitude_cut.size == 9
    assert set(longitude_cut) == set([20, 30, 40])
    assert set(latitude_cut) == set([-10, 0, 10])
    # Check region longitudes in -180, 180
    region = 170, -170, -10, 10
    are_inside = inside([longitude, latitude], region, latlon=True)
    longitude_cut, latitude_cut = longitude[are_inside], latitude[are_inside]
    assert longitude_cut.size == 9
    assert latitude_cut.size == 9
    assert set(longitude_cut) == set([170, 180, 190])
    assert set(latitude_cut) == set([-10, 0, 10])
    # Check region longitudes around zero meridian
    region = -10, 10, -10, 10
    are_inside = inside([longitude, latitude], region, latlon=True)
    longitude_cut, latitude_cut = longitude[are_inside], latitude[are_inside]
    assert longitude_cut.size == 9
    assert latitude_cut.size == 9
    assert set(longitude_cut) == set([0, 10, 350])
    assert set(latitude_cut) == set([-10, 0, 10])
    # Check region longitudes greater than 360
    region = 380, 400, -10, 10
    are_inside = inside([longitude, latitude], region, latlon=True)
    longitude_cut, latitude_cut = longitude[are_inside], latitude[are_inside]
    assert longitude_cut.size == 9
    assert latitude_cut.size == 9
    assert set(longitude_cut) == set([20, 30, 40])
    assert set(latitude_cut) == set([-10, 0, 10])


def test_inside_latlon_180_180():
    "Check if inside gets points properly with geographic coordinates on [-180, 180]"
    # Define longitude coordinates on -180, 180
    longitude = np.linspace(-170, 180, 36)
    latitude = np.linspace(-90, 90, 19)
    longitude, latitude = np.meshgrid(longitude, latitude)
    # Check region longitudes in 0, 360
    region = 20, 40, -10, 10
    are_inside = inside([longitude, latitude], region, latlon=True)
    longitude_cut, latitude_cut = longitude[are_inside], latitude[are_inside]
    assert longitude_cut.size == 9
    assert latitude_cut.size == 9
    assert set(longitude_cut) == set([20, 30, 40])
    assert set(latitude_cut) == set([-10, 0, 10])
    # Check region longitudes in 0, 360 around 180
    region = 170, 190, -10, 10
    are_inside = inside([longitude, latitude], region, latlon=True)
    longitude_cut, latitude_cut = longitude[are_inside], latitude[are_inside]
    assert longitude_cut.size == 9
    assert latitude_cut.size == 9
    assert set(longitude_cut) == set([170, 180, -170])
    assert set(latitude_cut) == set([-10, 0, 10])
    # Check region longitudes in -180, 180
    region = 170, -170, -10, 10
    are_inside = inside([longitude, latitude], region, latlon=True)
    longitude_cut, latitude_cut = longitude[are_inside], latitude[are_inside]
    assert longitude_cut.size == 9
    assert latitude_cut.size == 9
    assert set(longitude_cut) == set([170, 180, -170])
    assert set(latitude_cut) == set([-10, 0, 10])
    # Check region longitudes around zero meridian
    region = -10, 10, -10, 10
    are_inside = inside([longitude, latitude], region, latlon=True)
    longitude_cut, latitude_cut = longitude[are_inside], latitude[are_inside]
    assert longitude_cut.size == 9
    assert latitude_cut.size == 9
    assert set(longitude_cut) == set([-10, 0, 10])
    assert set(latitude_cut) == set([-10, 0, 10])
    # Check region longitudes greater than 360
    region = 380, 400, -10, 10
    are_inside = inside([longitude, latitude], region, latlon=True)
    longitude_cut, latitude_cut = longitude[are_inside], latitude[are_inside]
    assert longitude_cut.size == 9
    assert latitude_cut.size == 9
    assert set(longitude_cut) == set([20, 30, 40])
    assert set(latitude_cut) == set([-10, 0, 10])
