"""
Test data fetching routines.
"""
import os

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import pytest

from ..datasets.sample_data import (
    locate,
    fetch_baja_bathymetry,
    setup_baja_bathymetry_map,
    fetch_rio_magnetic,
    setup_rio_magnetic_map,
    fetch_california_gps,
    setup_california_gps_map,
    fetch_texas_wind,
    setup_texas_wind_map,
)


def test_datasets_locate():
    "Make sure the data cache location has the right package name"
    path = locate()
    assert os.path.exists(path)
    # This is the most we can check in a platform independent way without
    # testing appdirs itself.
    assert "verde" in path


def test_fetch_texas_wind():
    "Make sure the data are loaded properly"
    data = fetch_texas_wind()
    assert data.size == 1116
    assert data.shape == (186, 6)
    assert all(
        data.columns
        == [
            "station_id",
            "longitude",
            "latitude",
            "air_temperature_c",
            "wind_speed_east_knots",
            "wind_speed_north_knots",
        ]
    )


@pytest.mark.mpl_image_compare
def test_setup_texas_wind():
    "Test the map setup"
    fig = plt.figure()
    ax = plt.subplot(111, projection=ccrs.Mercator())
    setup_texas_wind_map(ax)
    return fig


def test_fetch_baja_bathymetry():
    "Make sure the data are loaded properly"
    data = fetch_baja_bathymetry()
    assert data.size == 248910
    assert data.shape == (82970, 3)
    assert all(data.columns == ["longitude", "latitude", "bathymetry_m"])


@pytest.mark.mpl_image_compare
def test_setup_baja_bathymetry():
    "Test the map setup"
    fig = plt.figure()
    ax = plt.subplot(111, projection=ccrs.Mercator())
    setup_baja_bathymetry_map(ax)
    return fig


def test_fetch_rio_magnetic():
    "Make sure the data are loaded properly"
    data = fetch_rio_magnetic()
    assert data.size == 226308
    assert data.shape == (37718, 6)
    assert all(
        data.columns
        == [
            "longitude",
            "latitude",
            "total_field_anomaly_nt",
            "height_ell_m",
            "line_type",
            "line_number",
        ]
    )


@pytest.mark.mpl_image_compare
def test_setup_rio_magnetic():
    "Test the map setup"
    fig = plt.figure()
    ax = plt.subplot(111, projection=ccrs.Mercator())
    setup_rio_magnetic_map(ax)
    return fig


def test_fetch_california_gps():
    "Make sure the data are loaded properly"
    data = fetch_california_gps()
    assert data.size == 22122
    assert data.shape == (2458, 9)
    columns = [
        "latitude",
        "longitude",
        "height",
        "velocity_north",
        "velocity_east",
        "velocity_up",
        "std_north",
        "std_east",
        "std_up",
    ]
    assert all(data.columns == columns)


@pytest.mark.mpl_image_compare
def test_setup_california_gps():
    "Test the map setup"
    fig = plt.figure()
    ax = plt.subplot(111, projection=ccrs.Mercator())
    setup_california_gps_map(ax)
    return fig
