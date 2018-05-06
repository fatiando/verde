"""
Functions to download, verify, and update a sample dataset.
"""
import os
from warnings import warn
from urllib.request import urlopen

from ..utils import get_data_dir


VERDE_DATA_STORE_URL = 'https://github.com/fatiando/verde/raw/master/data'


def fetch_data(filename, force_download=False):
    """
    Get the path to a data file in the Verde data directory.

    If it doesn't exist, download it from the remote data store.

    Parameters
    ----------
    filename : str
        The name of the data file to fetch.
    force_download : bool
        If True, will download the file even if it already exists.

    Return
    ------
    data_path : str
        A full path to the local file in the data cache directory.

    """
    data_dir = get_data_dir()
    data_path = os.path.join(data_dir, filename)
    if not os.path.exists(data_path) or force_download:
        data_src = '/'.join([VERDE_DATA_STORE_URL, filename])
        warn("Downloading data file '{}' from remote data store '{}' to '{}'."
             .format(filename, VERDE_DATA_STORE_URL, data_dir))
        with urlopen(data_src) as request:
            with open(data_path, 'wb') as dest:
                dest.write(request.read())
    return data_path
