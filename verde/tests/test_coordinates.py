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
    check_region([0, 360, -90, 90], latlon=True)
    with pytest.raises(ValueError):
        check_region([-200, 0, -10, 10], latlon=True)
    with pytest.raises(ValueError):
        check_region([0, 400, -10, 10], latlon=True)
    with pytest.raises(ValueError):
        check_region([-200, -190, -10, 10], latlon=True)
    with pytest.raises(ValueError):
        check_region([-45, 45, -100, 0], latlon=True)
    with pytest.raises(ValueError):
        check_region([-45, 45, -100, 0], latlon=True)
    with pytest.raises(ValueError):
        check_region([-45, 45, 0, 100], latlon=True)
    with pytest.raises(ValueError):
        check_region([-100, 260.5, -30, 30], latlon=True)


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
    latitude = np.linspace(-90, 90, 36)
    s, n = -90, 90
    # Check w, e in [0, 360)
    w, e = 10.5, 20.3
    for longitude in [longitude_360, longitude_180]:
        coordinates = [longitude, latitude]
        coordinates_new, region_new = _latlon_continuity(coordinates, (w, e, s, n))
        w_new, e_new = region_new[:2]
        assert w_new == w
        assert e_new == e
        npt.assert_allclose(coordinates_new[0], longitude_360)
    # Check w, e in [-180, 180)
    w, e = -20, 20
    for longitude in [longitude_360, longitude_180]:
        coordinates = [longitude, latitude]
        coordinates_new, region_new = _latlon_continuity(coordinates, (w, e, s, n))
        w_new, e_new = region_new[:2]
        assert w_new == -20
        assert e_new == 20
        npt.assert_allclose(coordinates_new[0], longitude_180)
    # Check region around the globe
    for w, e in [[0, 360], [-180, 180], [-20, 340]]:
        for longitude in [longitude_360, longitude_180]:
            coordinates = [longitude, latitude]
            coordinates_new, region_new = _latlon_continuity(coordinates, (w, e, s, n))
            w_new, e_new = region_new[:2]
            assert w_new == 0
            assert e_new == 360
            npt.assert_allclose(coordinates_new[0], longitude_360)
    # Check w == e
    w, e = 20, 20
    for longitude in [longitude_360, longitude_180]:
        coordinates = [longitude, latitude]
        coordinates_new, region_new = _latlon_continuity(coordinates, (w, e, s, n))
        w_new, e_new = region_new[:2]
        assert w_new == 20
        assert e_new == 20
        npt.assert_allclose(coordinates_new[0], longitude_360)
    # Check angle greater than 180
    w, e = 0, 200
    for longitude in [longitude_360, longitude_180]:
        coordinates = [longitude, latitude]
        coordinates_new, region_new = _latlon_continuity(coordinates, (w, e, s, n))
        w_new, e_new = region_new[:2]
        assert w_new == 0
        assert e_new == 200
        npt.assert_allclose(coordinates_new[0], longitude_360)
    w, e = -160, 160
    for longitude in [longitude_360, longitude_180]:
        coordinates = [longitude, latitude]
        coordinates_new, region_new = _latlon_continuity(coordinates, (w, e, s, n))
        w_new, e_new = region_new[:2]
        assert w_new == -160
        assert e_new == 160
        npt.assert_allclose(coordinates_new[0], longitude_180)


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


def test_inside_latlon_around_poles():
    "Test inside function when region is around the poles"
    longitude, latitude = grid_coordinates([0, 350, -90, 90], spacing=10.0)
    # North Pole all around the globe
    regions = [[0, 360, 70, 90], [-180, 180, 70, 90]]
    for region in regions:
        are_inside = inside([longitude, latitude], region, latlon=True)
        assert longitude[are_inside].size == 3 * 36
        assert latitude[are_inside].size == 3 * 36
        assert set(longitude[are_inside]) == set(np.unique(longitude))
        assert set(latitude[are_inside]) == set([70, 80, 90])
    # South Pole all around the globe
    regions = [[0, 360, -90, -70], [-180, 180, -90, -70]]
    for region in regions:
        are_inside = inside([longitude, latitude], region, latlon=True)
        assert longitude[are_inside].size == 3 * 36
        assert latitude[are_inside].size == 3 * 36
        assert set(longitude[are_inside]) == set(np.unique(longitude))
        assert set(latitude[are_inside]) == set([-90, -80, -70])
    # Section at the North Pole
    region = [40, 90, 70, 90]
    are_inside = inside([longitude, latitude], region, latlon=True)
    assert longitude[are_inside].size == 3 * 6
    assert latitude[are_inside].size == 3 * 6
    assert set(longitude[are_inside]) == set([40, 50, 60, 70, 80, 90])
    assert set(latitude[are_inside]) == set([70, 80, 90])
    region = [-90, -40, 70, 90]
    are_inside = inside([longitude, latitude], region, latlon=True)
    assert longitude[are_inside].size == 3 * 6
    assert latitude[are_inside].size == 3 * 6
    assert set(longitude[are_inside]) == set([270, 280, 290, 300, 310, 320])
    assert set(latitude[are_inside]) == set([70, 80, 90])
    # Section at the South Pole
    region = [40, 90, -90, -70]
    are_inside = inside([longitude, latitude], region, latlon=True)
    assert longitude[are_inside].size == 3 * 6
    assert latitude[are_inside].size == 3 * 6
    assert set(longitude[are_inside]) == set([40, 50, 60, 70, 80, 90])
    assert set(latitude[are_inside]) == set([-90, -80, -70])
    region = [-90, -40, -90, -70]
    are_inside = inside([longitude, latitude], region, latlon=True)
    assert longitude[are_inside].size == 3 * 6
    assert latitude[are_inside].size == 3 * 6
    assert set(longitude[are_inside]) == set([270, 280, 290, 300, 310, 320])
    assert set(latitude[are_inside]) == set([-90, -80, -70])
