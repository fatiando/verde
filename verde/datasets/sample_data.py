"""
Functions to load sample data
"""
import numpy as np
import pandas as pd

from .download import fetch_data


def _setup_map(ax, xticks, yticks, crs, region, land=None, ocean=None):
    """
    Setup a Cartopy map with land and ocean features and proper tick labels.
    """
    import cartopy.feature as cfeature
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

    if land is not None:
        ax.add_feature(cfeature.LAND, facecolor=land)
    if ocean is not None:
        ax.add_feature(cfeature.OCEAN, facecolor=ocean)
    ax.set_extent(region, crs=crs)
    # Set the proper ticks for a Cartopy map
    ax.set_xticks(xticks, crs=crs)
    ax.set_yticks(yticks, crs=crs)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())


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

    See also
    --------
    setup_baja_bathymetry_map: Utility function to help setup a Cartopy map.

    """
    data_file = fetch_data("baja-bathymetry.csv.xz", force_download=force_download)
    data = pd.read_csv(data_file, compression="xz")
    return data


def setup_baja_bathymetry_map(
    ax, region=(245.0, 254.705, 20.0, 29.99), land="gray", ocean=None
):
    """
    Setup a Cartopy map for the Baja California bathymetry dataset.

    Parameters
    ----------
    ax : matplotlib Axes
        The axes where the map is being plotted.
    region : list = [W, E, S, N]
        The boundaries of the map region in the coordinate system of the data.
    land : str or None
        The name of the color of the land feature or None to omit it.
    ocean : str or None
        The name of the color of the ocean feature or None to omit it.

    See also
    --------
    fetch_baja_bathymetry: Sample bathymetry data from Baja California.

    """
    import cartopy.crs as ccrs

    _setup_map(
        ax,
        xticks=np.arange(-114, -105, 2),
        yticks=np.arange(21, 30, 2),
        land=land,
        ocean=ocean,
        region=region,
        crs=ccrs.PlateCarree(),
    )


def fetch_rio_magnetic(force_download=False):
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

    See also
    --------
    setup_rio_magnetic_map: Utility function to help setup a Cartopy map.

    """
    data_file = fetch_data("rio-magnetic.csv.xz", force_download=force_download)
    data = pd.read_csv(data_file, compression="xz")
    return data


def setup_rio_magnetic_map(ax, region=(-42.6, -42, -22.5, -22)):
    """
    Setup a Cartopy map for the Rio de Janeiro magnetic anomaly dataset.

    Parameters
    ----------
    ax : matplotlib Axes
        The axes where the map is being plotted.
    region : list = [W, E, S, N]
        The boundaries of the map region in the coordinate system of the data.
    land : str or None
        The name of the color of the land feature or None to omit it.
    ocean : str or None
        The name of the color of the ocean feature or None to omit it.

    See also
    --------
    fetch_rio_magnetic: Sample magnetic anomaly data from Rio de Janeiro, Brazil.

    """
    import cartopy.crs as ccrs

    _setup_map(
        ax,
        xticks=np.arange(-42.5, -42, 0.1),
        yticks=np.arange(-22.5, -21.99, 0.1),
        land=None,
        ocean=None,
        region=region,
        crs=ccrs.PlateCarree(),
    )


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

    See also
    --------
    setup_california_gps_map: Utility function to help setup a Cartopy map.

    """
    data_file = fetch_data("california-gps.csv.xz", force_download=force_download)
    data = pd.read_csv(data_file, compression="xz")
    return data


def setup_california_gps_map(
    ax, region=(235.2, 245.3, 31.9, 42.3), land="gray", ocean="skyblue"
):
    """
    Setup a Cartopy map for the California GPS velocity dataset.

    Parameters
    ----------
    ax : matplotlib Axes
        The axes where the map is being plotted.
    region : list = [W, E, S, N]
        The boundaries of the map region in the coordinate system of the data.
    land : str or None
        The name of the color of the land feature or None to omit it.
    ocean : str or None
        The name of the color of the ocean feature or None to omit it.

    See also
    --------
    fetch_california_gps: Sample GPS velocity data from California.

    """
    import cartopy.crs as ccrs

    _setup_map(
        ax,
        xticks=np.arange(-124, -115, 4),
        yticks=np.arange(33, 42, 2),
        land=land,
        ocean=ocean,
        region=region,
        crs=ccrs.PlateCarree(),
    )
