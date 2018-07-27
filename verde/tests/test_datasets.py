# pylint: disable=wrong-import-position
"""
Test data fetching routines.
"""
# Import matplotlib and set the backend before anything else to make sure no windows are
# created and there are no problem with TravisCI running in headless mode.
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import pytest

from ..datasets.sample_data import (
    fetch_baja_bathymetry,
    setup_baja_bathymetry_map,
    fetch_rio_magnetic,
    setup_rio_magnetic_map,
    fetch_california_gps,
    setup_california_gps_map,
)


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
    assert data.size == 150884
    assert data.shape == (37721, 4)
    assert all(
        data.columns
        == ["longitude", "latitude", "total_field_anomaly_nt", "height_ell_m"]
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
