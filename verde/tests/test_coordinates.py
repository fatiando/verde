"""
Test the coordinate generation functions
"""
import warnings

import numpy as np
import numpy.testing as npt
import pytest

from ..coordinates import (
    check_region,
    spacing_to_shape,
    profile_coordinates,
    grid_coordinates,
    longitude_continuity,
    rolling_window,
)


def test_rolling_window_invalid_coordinate_shapes():
    "Shapes of input coordinates must all be the same"
    coordinates = [np.arange(10), np.arange(10).reshape((5, 2))]
    with pytest.raises(ValueError):
        rolling_window(coordinates, size=2, spacing=1)


def test_rolling_window_empty():
    "Make sure empty windows return an empty index"
    coords = grid_coordinates((-5, -1, 6, 10), spacing=1)
    # Use a larger region to make sure the first window is empty
    # Doing this will raise a warning for non-overlapping windows. Capture it
    # so it doesn't pollute the test log.
    with warnings.catch_warnings(record=True):
        windows = rolling_window(coords, size=0.001, spacing=1, region=(-7, 1, 4, 12))[
            1
        ]
    assert windows[0, 0][0].size == 0 and windows[0, 0][1].size == 0
    # Make sure we can still index with an empty array
    assert coords[0][windows[0, 0]].size == 0


def test_rolling_window_warnings():
    "Should warn users if the windows don't overlap"
    coords = grid_coordinates((-5, -1, 6, 10), spacing=1)
    # For exact same size there will be 1 point overlapping so should not warn
    with warnings.catch_warnings(record=True) as warn:
        rolling_window(coords, size=2, spacing=2)
        assert not any(issubclass(w.category, UserWarning) for w in warn)
    args = [dict(spacing=3), dict(spacing=(4, 1)), dict(shape=(1, 2))]
    for arg in args:
        with warnings.catch_warnings(record=True) as warn:
            rolling_window(coords, size=2, **arg)
            # Filter out the user warnings from some deprecation warnings that
            # might be thrown by other packages.
            userwarnings = [w for w in warn if issubclass(w.category, UserWarning)]
            assert len(userwarnings) == 1
            assert issubclass(userwarnings[-1].category, UserWarning)
            assert str(userwarnings[-1].message).split()[0] == "Rolling"


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


def test_longitude_continuity():
    "Test continuous boundary conditions in geographic coordinates."
    # Define longitude around the globe for [0, 360) and [-180, 180)
    longitude_360 = np.linspace(0, 350, 36)
    longitude_180 = np.hstack((longitude_360[:18], longitude_360[18:] - 360))
    latitude = np.linspace(-90, 90, 36)
    s, n = -90, 90
    # Check w, e in [0, 360)
    w, e = 10.5, 20.3
    for longitude in [longitude_360, longitude_180]:
        coordinates = [longitude, latitude]
        coordinates_new, region_new = longitude_continuity(coordinates, (w, e, s, n))
        w_new, e_new = region_new[:2]
        assert w_new == w
        assert e_new == e
        npt.assert_allclose(coordinates_new[0], longitude_360)
    # Check w, e in [-180, 180)
    w, e = -20, 20
    for longitude in [longitude_360, longitude_180]:
        coordinates = [longitude, latitude]
        coordinates_new, region_new = longitude_continuity(coordinates, (w, e, s, n))
        w_new, e_new = region_new[:2]
        assert w_new == -20
        assert e_new == 20
        npt.assert_allclose(coordinates_new[0], longitude_180)
    # Check region around the globe
    for w, e in [[0, 360], [-180, 180], [-20, 340]]:
        for longitude in [longitude_360, longitude_180]:
            coordinates = [longitude, latitude]
            coordinates_new, region_new = longitude_continuity(
                coordinates, (w, e, s, n)
            )
            w_new, e_new = region_new[:2]
            assert w_new == 0
            assert e_new == 360
            npt.assert_allclose(coordinates_new[0], longitude_360)
    # Check w == e
    w, e = 20, 20
    for longitude in [longitude_360, longitude_180]:
        coordinates = [longitude, latitude]
        coordinates_new, region_new = longitude_continuity(coordinates, (w, e, s, n))
        w_new, e_new = region_new[:2]
        assert w_new == 20
        assert e_new == 20
        npt.assert_allclose(coordinates_new[0], longitude_360)
    # Check angle greater than 180
    w, e = 0, 200
    for longitude in [longitude_360, longitude_180]:
        coordinates = [longitude, latitude]
        coordinates_new, region_new = longitude_continuity(coordinates, (w, e, s, n))
        w_new, e_new = region_new[:2]
        assert w_new == 0
        assert e_new == 200
        npt.assert_allclose(coordinates_new[0], longitude_360)
    w, e = -160, 160
    for longitude in [longitude_360, longitude_180]:
        coordinates = [longitude, latitude]
        coordinates_new, region_new = longitude_continuity(coordinates, (w, e, s, n))
        w_new, e_new = region_new[:2]
        assert w_new == -160
        assert e_new == 160
        npt.assert_allclose(coordinates_new[0], longitude_180)


def test_invalid_geographic_region():
    "Check if invalid region in longitude_continuity raises a ValueError"
    # Region with latitude over boundaries
    w, e = -10, 10
    for s, n in [[-200, 90], [-90, 200]]:
        with pytest.raises(ValueError):
            longitude_continuity(None, [w, e, s, n])
    # Region with longitude over boundaries
    s, n = -10, 10
    for w, e in [[-200, 0], [0, 380]]:
        with pytest.raises(ValueError):
            longitude_continuity(None, [w, e, s, n])
    # Region with longitudinal difference greater than 360
    w, e, s, n = -180, 200, -10, 10
    with pytest.raises(ValueError):
        longitude_continuity(None, [w, e, s, n])


def test_invalid_geographic_coordinates():
    "Check if invalid coordinates in longitude_continuity raises a ValueError"
    boundaries = [0, 360, -90, 90]
    spacing = 10
    region = [-20, 20, -20, 20]
    # Region with longitude point over boundaries
    longitude, latitude = grid_coordinates(boundaries, spacing=spacing)
    longitude[0] = -200
    with pytest.raises(ValueError):
        longitude_continuity([longitude, latitude], region)
    longitude[0] = 400
    with pytest.raises(ValueError):
        longitude_continuity([longitude, latitude], region)
    # Region with latitude point over boundaries
    longitude, latitude = grid_coordinates(boundaries, spacing=spacing)
    latitude[0] = -100
    with pytest.raises(ValueError):
        longitude_continuity([longitude, latitude], region)
    latitude[0] = 100
    with pytest.raises(ValueError):
        longitude_continuity([longitude, latitude], region)
