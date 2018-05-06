"""
Test data fetching routines.
"""
import os
from pathlib import Path
import warnings
from urllib.error import HTTPError

import pytest

from ..datasets.download import fetch_data_from_store


VERDE_DATA_DIR = Path(__file__).parent.parent.parent.joinpath('data')


def test_fetch_data_from_store_local():
    "Move data from the repository to the data directory"
    with warnings.catch_warnings(record=True) as warn:
        datapath = fetch_data_from_store('baja-california-bathymetry.csv.xz',
                                         data_store=VERDE_DATA_DIR)
        assert len(warn) == 1
        assert issubclass(warn[-1].category, UserWarning)
        assert str(warn[-1].message).split()[0] == "Caching"
    assert os.path.exists(datapath)


def test_fetch_data_from_store_local_fail():
    "An exception should be raised for invalid file names"
    with pytest.raises(FileNotFoundError):
        fetch_data_from_store('an invalid file name',
                              data_store=VERDE_DATA_DIR)


def test_fetch_data_from_store_local_env():
    "Use an environment variable to set data path"
    os.environ['VERDE_DATA_STORE'] = str(VERDE_DATA_DIR)
    try:
        with warnings.catch_warnings(record=True) as warn:
            datapath = fetch_data_from_store(
                'baja-california-bathymetry.csv.xz', data_store=None)
            assert len(warn) == 1
            assert issubclass(warn[-1].category, UserWarning)
            assert str(warn[-1].message).split()[0] == "Caching"
        assert os.path.exists(datapath)
    finally:
        del os.environ['VERDE_DATA_STORE']


def test_fetch_data_from_store_remote():
    "Download data from Github to the data directory"
    with warnings.catch_warnings(record=True) as warn:
        datapath = fetch_data_from_store('baja-california-bathymetry.csv.xz',
                                         data_store=None)
        assert len(warn) == 1
        assert issubclass(warn[-1].category, UserWarning)
        assert str(warn[-1].message).split()[0] == "Downloading"
    assert os.path.exists(datapath)


def test_fetch_data_from_store_remote_fail():
    "Should raise an exception if the remote 404s"
    with warnings.catch_warnings(record=True) as warn:
        with pytest.raises(HTTPError):
            fetch_data_from_store('invalid remote file name', data_store=None)
        assert len(warn) == 1
        assert issubclass(warn[-1].category, UserWarning)
        assert str(warn[-1].message).split()[0] == "Downloading"
