"""
Functions to load sample data
"""
import warnings

import pkg_resources
import numpy as np
import pandas as pd
import pooch

try:
    import cartopy.feature as cfeature
    import cartopy.crs as ccrs
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
except ImportError:
    pass

from ..version import full_version


# Otherwise, DeprecationWarning won't be shown, kind of defeating the purpose.
warnings.simplefilter("default")

REGISTRY = pooch.create(
    path=pooch.os_cache("verde"),
    base_url="https://github.com/fatiando/verde/raw/{version}/data/",
    version=full_version,
    version_dev="master",
    env="VERDE_DATA_DIR",
)
with pkg_resources.resource_stream("verde.datasets", "registry.txt") as registry_file:
    REGISTRY.load_registry(registry_file)


def locate():
    r"""
    The absolute path to the sample data storage location on disk.

    This is where the data are saved on your computer. The location is
    dependent on the operating system. The folder locations are defined by the
    ``appdirs``  package (see the `appdirs documentation
    <https://github.com/ActiveState/appdirs>`__).

    The location can be overwritten by the ``VERDE_DATA_DIR`` environment
    variable to the desired destination.

    Returns
    -------
    path : str
        The local data storage location.

    """
    return str(REGISTRY.abspath)


def _setup_map(
    ax, xticks, yticks, crs, region, land=None, ocean=None, borders=None, states=None
):
    """
    Setup a Cartopy map with land and ocean features and proper tick labels.
    """

    if land is not None:
        ax.add_feature(cfeature.LAND, facecolor=land)
    if ocean is not None:
        ax.add_feature(cfeature.OCEAN, facecolor=ocean)
    if borders is not None:
        ax.add_feature(cfeature.BORDERS, linewidth=borders)
    if states is not None:
        ax.add_feature(cfeature.STATES, linewidth=states)
    ax.set_extent(region, crs=crs)
    # Set the proper ticks for a Cartopy map
    ax.set_xticks(xticks, crs=crs)
    ax.set_yticks(yticks, crs=crs)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())


def fetch_baja_bathymetry():
    """
    Fetch sample bathymetry data from Baja California.

    This is the ``@tut_ship.xyz`` sample data from the `GMT
    <http://gmt.soest.hawaii.edu/>`__ tutorial.

    If the file isn't already in your data directory, it will be downloaded
    automatically.

    Returns
    -------
    data : :class:`pandas.DataFrame`
        The bathymetry data. Columns are longitude, latitude, and bathymetry
        (in meters) for each data point.

    See also
    --------
    setup_baja_bathymetry_map: Utility function to help setup a Cartopy map.

    """
    data_file = REGISTRY.fetch("baja-bathymetry.csv.xz")
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
    _setup_map(
        ax,
        xticks=np.arange(-114, -105, 2),
        yticks=np.arange(21, 30, 2),
        land=land,
        ocean=ocean,
        region=region,
        crs=ccrs.PlateCarree(),
    )


def fetch_rio_magnetic():
    """
    Fetch total-field magnetic anomaly data from Rio de Janeiro, Brazil.

    .. warning::

        **The Rio magnetic anomaly dataset is deprecated and will be removed in
        Verde v2.0.0** (functions :func:`verde.datasets.fetch_rio_magnetic` and
        :func:`verde.datasets.setup_rio_magnetic_map`). Please use another
        dataset instead.

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

    If the file isn't already in your data directory, it will be downloaded
    automatically.

    Returns
    -------
    data : :class:`pandas.DataFrame`
        The magnetic anomaly data. Columns are longitude, latitude, total-field
        magnetic anomaly (nanoTesla), observation height above the WGS84
        ellipsoid (in meters), flight line type (LINE or TIE), and flight line
        number for each data point.

    See also
    --------
    setup_rio_magnetic_map: Utility function to help setup a Cartopy map.

    """
    warnings.warn(
        "The Rio magnetic anomaly dataset is deprecated and will be removed "
        "in Verde v2.0.0. Use a different dataset instead.",
        DeprecationWarning,
    )
    data_file = REGISTRY.fetch("rio-magnetic.csv.xz")
    data = pd.read_csv(data_file, compression="xz")
    return data


def setup_rio_magnetic_map(ax, region=(-42.6, -42, -22.5, -22)):
    """
    Setup a Cartopy map for the Rio de Janeiro magnetic anomaly dataset.

    .. warning::

        **The Rio magnetic anomaly dataset is deprecated and will be removed in
        Verde v2.0.0** (functions :func:`verde.datasets.fetch_rio_magnetic` and
        :func:`verde.datasets.setup_rio_magnetic_map`). Please use another
        dataset instead.

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
    fetch_rio_magnetic: Magnetic anomaly data from Rio de Janeiro, Brazil.

    """
    warnings.warn(
        "The Rio magnetic anomaly dataset is deprecated and will be removed "
        "in Verde v2.0.0. Use a different dataset instead.",
        DeprecationWarning,
    )
    _setup_map(
        ax,
        xticks=np.arange(-42.5, -42, 0.1),
        yticks=np.arange(-22.5, -21.99, 0.1),
        land=None,
        ocean=None,
        region=region,
        crs=ccrs.PlateCarree(),
    )


def fetch_california_gps():
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

    If the file isn't already in your data directory, it will be downloaded
    automatically.

    Returns
    -------
    data : :class:`pandas.DataFrame`
        The GPS velocity data. Columns are longitude, latitude, height
        (geometric, in meters), East velocity (meter/year), North velocity
        (meter/year), upward velocity (meter/year), standard deviation of East
        velocity (meter/year), standard deviation of North velocity
        (meter/year), standard deviation of upward velocity (meter/year).

    See also
    --------
    setup_california_gps_map: Utility function to help setup a Cartopy map.

    """
    data_file = REGISTRY.fetch("california-gps.csv.xz")
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
    _setup_map(
        ax,
        xticks=np.arange(-124, -115, 4),
        yticks=np.arange(33, 42, 2),
        land=land,
        ocean=ocean,
        region=region,
        crs=ccrs.PlateCarree(),
    )


def fetch_texas_wind():
    """
    Fetch sample wind speed and air temperature data for Texas, USA.

    Data are average wind speed and air temperature for data for February 26
    2018. The original data was downloaded from `Iowa State University
    <https://mesonet.agron.iastate.edu/request/download.phtml>`__.

    If the file isn't already in your data directory, it will be downloaded
    automatically.

    Returns
    -------
    data : :class:`pandas.DataFrame`
        Columns are the station ID, longitude, latitude, air temperature in C,
        east component of wind speed in knots, and north component of wind
        speed in knots.

    See also
    --------
    setup_texas_wind_map: Utility function to help setup a Cartopy map.

    """
    data_file = REGISTRY.fetch("texas-wind.csv")
    data = pd.read_csv(data_file)
    return data


def setup_texas_wind_map(
    ax, region=(-107, -93, 25.5, 37), land="#dddddd", borders=0.5, states=0.1
):
    """
    Setup a Cartopy map for the Texas wind speed and air temperature dataset.

    Parameters
    ----------
    ax : matplotlib Axes
        The axes where the map is being plotted.
    region : list = [W, E, S, N]
        The boundaries of the map region in the coordinate system of the data.
    land : str or None
        The name of the color of the land feature or None to omit it.
    borders : float or None
        Line width of the country borders.
    states : float or None
        Line width of the state borders.

    See also
    --------
    fetch_texas_wind: Sample wind speed and air temperature data for Texas.

    """

    _setup_map(
        ax,
        xticks=np.arange(-106, -92, 3),
        yticks=np.arange(27, 38, 3),
        land=land,
        ocean=None,
        region=region,
        borders=borders,
        states=states,
        crs=ccrs.PlateCarree(),
    )
