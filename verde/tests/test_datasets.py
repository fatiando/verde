"""
Test data fetching routines.
"""
import os
import warnings
from requests.exceptions import HTTPError

import pytest

from ..datasets.download import fetch_data
from ..datasets.sample_data import fetch_baja_bathymetry, \
    fetch_rio_magnetic_anomaly, fetch_california_gps


def compare_tiny_data(datapath):
    """
    Check if the file exists and compare its content with what we know it
    should be.
    """
    assert os.path.exists(datapath)
    with open(datapath) as datafile:
        content = datafile.read()
    true_content = "\n".join([
        '# A tiny data file for test purposes only',
        '1  2  3  4  5  6'])
    assert content.strip() == true_content


# Has to go first in order to guarantee that the file has been downloaded
def test_fetch_data_from_remote():
    "Download data from Github to the data directory"
    with warnings.catch_warnings(record=True) as warn:
        datapath = fetch_data('tiny-data.txt', force_download=True)
        assert len(warn) == 1
        assert issubclass(warn[-1].category, UserWarning)
        assert str(warn[-1].message).split()[0] == "Downloading"
    compare_tiny_data(datapath)


def test_fetch_data():
    "Make sure the file exists when not being downloaded"
    with warnings.catch_warnings(record=True) as warn:
        datapath = fetch_data('tiny-data.txt')
        assert not warn
    compare_tiny_data(datapath)


def test_fetch_data_from_store_remote_fail():
    "Should raise an exception if the remote 404s"
    with warnings.catch_warnings(record=True) as warn:
        with pytest.raises(HTTPError):
            fetch_data('invalid remote file name')
        assert len(warn) == 1
        assert issubclass(warn[-1].category, UserWarning)
        assert str(warn[-1].message).split()[0] == "Downloading"


def test_fetch_baja_bathymetry():
    "Make sure the data are loaded properly"
    data = fetch_baja_bathymetry()
    assert data.size == 248910
    assert data.shape == (82970, 3)
    assert all(data.columns == ['longitude', 'latitude', 'bathymetry_m'])


def test_fetch_rio_magnetic_anomaly():
    "Make sure the data are loaded properly"
    data = fetch_rio_magnetic_anomaly()
    assert data.size == 150884
    assert data.shape == (37721, 4)
    assert all(data.columns == ['longitude', 'latitude',
                                'total_field_anomaly_nt', 'height_ell_m'])


def test_fetch_california_gps():
    "Make sure the data are loaded properly"
    data = fetch_california_gps()
    assert data.size == 22122
    assert data.shape == (2458, 9)
    columns = ['latitude', 'longitude', 'height', 'velocity_north',
               'velocity_east', 'velocity_up', 'std_north', 'std_east',
               'std_up']
    assert all(data.columns == columns)
