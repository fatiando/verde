"""
Test data fetching routines.
"""
import os
from pathlib import Path
import warnings
from urllib.error import HTTPError
from contextlib import contextmanager

import pytest

from ..datasets.download import fetch_data_from_store


REPO_DATA_DIR = Path(__file__).parent.parent.parent.joinpath('data')


@contextmanager
def backup_store(path=None):
    """
    Backup the VERDE_DATA_STORE environment and remove it or replace with path.
    Restore it when the exiting the context.
    Use this to test functions without letting user configuration get in the
    way.
    """
    backup = os.environ.get('VERDE_DATA_STORE', None)
    if path is not None:
        os.environ['VERDE_DATA_STORE'] = str(path)
    else:
        if backup is not None:
            del os.environ['VERDE_DATA_STORE']
    try:
        yield
    finally:
        if backup is not None:
            os.environ['VERDE_DATA_STORE'] = backup


def test_fetch_data_from_store_remote():
    "Download data from Github to the data directory"
    with warnings.catch_warnings(record=True) as warn:
        with backup_store():
            datapath = fetch_data_from_store(
                'baja-california-bathymetry.csv.xz')
        assert len(warn) == 1
        assert issubclass(warn[-1].category, UserWarning)
        assert str(warn[-1].message).split()[0] == "Downloading"
    assert os.path.exists(datapath)


def test_fetch_data_from_store_remote_fail():
    "Should raise an exception if the remote 404s"
    with warnings.catch_warnings(record=True) as warn:
        with pytest.raises(HTTPError):
            with backup_store():
                fetch_data_from_store('invalid remote file name')
        assert len(warn) == 1
        assert issubclass(warn[-1].category, UserWarning)
        assert str(warn[-1].message).split()[0] == "Downloading"


def test_fetch_data_from_store_local():
    "Use an environment variable to set data path"
    with warnings.catch_warnings(record=True) as warn:
        with backup_store(str(REPO_DATA_DIR)):
            datapath = fetch_data_from_store(
                'baja-california-bathymetry.csv.xz')
        assert len(warn) == 1
        assert issubclass(warn[-1].category, UserWarning)
        assert str(warn[-1].message).split()[0] == "Caching"
    assert os.path.exists(datapath)


def test_fetch_data_from_store_local_fail():
    "An exception should be raised for invalid file names"
    with pytest.raises(FileNotFoundError):
        with backup_store(str(REPO_DATA_DIR)):
            fetch_data_from_store('an invalid file name')
