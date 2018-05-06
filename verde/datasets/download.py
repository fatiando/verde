"""
Functions to download, verify, and update a sample dataset.
"""
import os
import shutil
from warnings import warn
from urllib.request import urlopen

from ..utils import get_data_dir


VERDE_DATA_STORE_URL = 'https://github.com/fatiando/verde/raw/master/data'


def fetch_data_from_store(filename, data_store=None):
    """
    Download data from the Verde data store.

    If the environment variable ``VERDE_DATA_STORE`` is set, will fetch data
    files from there instead. This must be a local path.

    Overwrites the file if it already exists in the Verde data directory.

    """
    data_dest = os.path.join(get_data_dir(), filename)
    # Get data from a local data store if the environment variable is defined
    if data_store is None:
        data_store = os.environ.get('VERDE_DATA_STORE', None)
    if data_store is not None:
        data_src = os.path.join(data_store, filename)
        if not os.path.exists(data_src):
            raise FileNotFoundError(
                "Unable to find data file '{}' at local data store '{}'."
                .format(filename, data_store))
        warn("Caching data file '{}' from local data store '{}' to '{}'."
             .format(filename, data_store, get_data_dir()))
        shutil.copy2(data_src, data_dest)
        return data_dest
    # Download the data from the remote data store
    data_src = '/'.join([VERDE_DATA_STORE_URL, filename])
    warn("Downloading data file '{}' from remote data store '{}' to '{}'."
         .format(filename, VERDE_DATA_STORE_URL, get_data_dir()))
    with urlopen(data_src) as request:
        with open(data_dest, 'wb') as dest:
            dest.write(request.read())
    return data_dest
