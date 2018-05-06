"""
Functions to load sample data
"""
import pandas as pd

from .download import fetch_data


def fetch_baja_bathymetry(force_download=False):
    """
    Fetch sample bathymetry data from Baja California.

    This is the ``@tut_ship.xyz`` sample data from the `GMT
    <http://gmt.soest.hawaii.edu/>`__ tutorial.

    If the file isn't already in your data directory (``$HOME/.verde/data`` by
    default), it will be downloaded.

    Parameters
    ----------
    force_download : bool
        If True, will download the file even if it already exists.

    Returns
    -------
    data : pandas.DataFrame
        The bathymetry data. Columns are longitude, latitude, and bathymetry
        (in meters) for each data point.

    """
    data_file = fetch_data('baja-california-bathymetry.csv.xz',
                           force_download=force_download)
    data = pd.read_csv(data_file, compression='xz')
    return data
