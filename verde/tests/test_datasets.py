"""
Test data fetching routines.
"""
import os
import warnings
from urllib.error import HTTPError

import pytest

from ..datasets.download import fetch_data


# Has to go first in order to guarantee that the file has been downloaded
def test_fetch_data_from_remote():
    "Download data from Github to the data directory"
    with warnings.catch_warnings(record=True) as warn:
        datapath = fetch_data('baja-california-bathymetry.csv.xz',
                              force_download=True)
        assert len(warn) == 1
        assert issubclass(warn[-1].category, UserWarning)
        assert str(warn[-1].message).split()[0] == "Downloading"
    assert os.path.exists(datapath)


def test_fetch_data():
    "Make sure the file exists when not being downloaded"
    with warnings.catch_warnings(record=True) as warn:
        datapath = fetch_data('baja-california-bathymetry.csv.xz')
        assert len(warn) == 0
    assert os.path.exists(datapath)


def test_fetch_data_from_store_remote_fail():
    "Should raise an exception if the remote 404s"
    with warnings.catch_warnings(record=True) as warn:
        with pytest.raises(HTTPError):
            fetch_data('invalid remote file name')
        assert len(warn) == 1
        assert issubclass(warn[-1].category, UserWarning)
        assert str(warn[-1].message).split()[0] == "Downloading"
