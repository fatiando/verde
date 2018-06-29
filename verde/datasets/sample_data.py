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
    data_file = fetch_data(
        "baja-california-bathymetry.csv.xz", force_download=force_download
    )
    data = pd.read_csv(data_file, compression="xz")
    return data


def fetch_rio_magnetic_anomaly(force_download=False):
    """
    Fetch sample total-field magnetic anomaly data from Rio de Janeiro, Brazil.

    These data were cropped from the northwestern part of an airborne survey of
    Rio de Janeiro, Brazil, conducted in 1978. The data are made available by
    the Geological Survey of Brazil (CPRM) through their `GEOSGB portal
    <http://geosgb.cprm.gov.br/>`__.

    The anomaly is calculated with respect to the IGRF field parameters listed
    on the table below. See the original data for more processing information.

    +----------+-----------+----------------+-------------+-------------+
    |               IGRF for year 1978.3 at 500 m height                |
    +----------+-----------+----------------+-------------+-------------+
    | Latitude | Longitude | Intensity (nT) | Declination | Inclination |
    +==========+===========+================+=============+=============+
    |  -22º15' |  -42º15'  |     23834      |   -19º19'   |   -27º33'   |
    +----------+-----------+----------------+-------------+-------------+

    If the file isn't already in your data directory (``$HOME/.verde/data`` by
    default), it will be downloaded.

    Parameters
    ----------
    force_download : bool
        If True, will download the file even if it already exists.

    Returns
    -------
    data : pandas.DataFrame
        The magnetic anomaly data. Columns are longitude, latitude, total-field
        magnetic anomaly (nanoTesla), and observation height above the
        ellipsoid (in meters) for each data point.

    """
    data_file = fetch_data(
        "rio-de-janeiro-magnetic.csv.xz", force_download=force_download
    )
    data = pd.read_csv(data_file, compression="xz")
    return data


def fetch_california_gps(force_download=False):
    """
    Fetch sample GPS velocity data from California (the U.S. West coast).

    Velocities and their standard deviations are in meters/year. Height is
    geometric height above WGS84 in meters. Velocities are referenced to the
    North American tectonic plate (NAM08). The average velocities were released
    on 2017-12-27.

    This material is based on EarthScope Plate Boundary Observatory data
    services provided by UNAVCO through the GAGE Facility with support from the
    National Science Foundation (NSF) and National Aeronautics and Space
    Administration (NASA) under NSF Cooperative Agreement No. EAR-1261833.

    If the file isn't already in your data directory (``$HOME/.verde/data`` by
    default), it will be downloaded.

    Parameters
    ----------
    force_download : bool
        If True, will download the file even if it already exists.

    Returns
    -------
    data : pandas.DataFrame
        The GPS velocity data. Columns are longitude, latitude, height
        (geometric, in meters), East velocity (meter/year), North velocity
        (meter/year), upward velocity (meter/year), standard deviation of East
        velocity (meter/year), standard deviation of North velocity
        (meter/year), standard deviation of upward velocity (meter/year).

    """
    data_file = fetch_data("california-gps.csv.xz", force_download=force_download)
    data = pd.read_csv(data_file, compression="xz")
    return data
