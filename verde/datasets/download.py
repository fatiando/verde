"""
Functions to download, verify, and update a sample dataset.
"""
import os
import shutil
from urllib.request import urlopen

from ..utils import get_data_dir


VERDE_DATA_STORE_URL = 'https://github.com/fatiando/verde/raw/master/data'



def fetch_data_from_store(filename, data_store=None):
    """
    Download data from the Verde data store.

    If the environment variable ``VERDE_DATA_STORE`` is set, will fetch data
    files from there instead. This must be a local path.

    Overwrites the file in the Verde data directory if it already exists.

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
        shutil.copy2(data_src, data_dest)
        return data_dest
    # Download the data from the remote data store
    data_src = '/'.join(VERDE_DATA_STORE_URL, filename)
    with urlopen(data_src) as request:
        with open(data_dest, 'wb') as dest:
            dest.write(request.read())
    return data_dest
