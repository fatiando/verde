# Copyright (c) 2017 The Verde Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Functions to load sample data
"""
import warnings

import numpy as np
import pandas as pd
import pkg_resources
import pooch

try:
    import cartopy.crs as ccrs
    from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
except ImportError:
    pass

from .._version import __version__

REGISTRY = pooch.create(
    path=pooch.os_cache("verde"),
    base_url="https://github.com/fatiando/verde/raw/{version}/data/",
    version=__version__,
    version_dev="main",
    env="VERDE_DATA_DIR",
)
with pkg_resources.resource_stream("verde.datasets", "registry.txt") as registry_file:
    REGISTRY.load_registry(registry_file)


def _datasets_deprecation_warning():
    warnings.warn(
        "All sample datasets in Verde are deprecated and will be "
        "removed in Verde v2.0.0. The tutorials/examples will transition "
        "to using Ensaio (https://www.fatiando.org/ensaio/) instead.",
        DeprecationWarning,
    )


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
    ax,
    xticks,
    yticks,
    crs,
    region,
    coastlines=False,
):
    """
    Setup a Cartopy map with coastlines and proper tick labels.
    """
    if coastlines:
        ax.coastlines()
    ax.set_extent(region, crs=crs)
    # Set the proper ticks for a Cartopy map
    ax.set_xticks(xticks, crs=crs)
    ax.set_yticks(yticks, crs=crs)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())


def fetch_baja_bathymetry():
    """
    Fetch sample bathymetry data from Baja California.

    .. warning::

        All sample datasets in Verde are deprecated and will be
        **removed in Verde v2.0.0**.
        The tutorials/examples will transition to using
        `Ensaio <https://www.fatiando.org/ensaio/>`__ instead.

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
    _datasets_deprecation_warning()
    data_file = REGISTRY.fetch("baja-bathymetry.csv.xz")
    data = pd.read_csv(data_file, compression="xz")
    return data


def setup_baja_bathymetry_map(
    ax, region=(245.0, 254.705, 20.0, 29.99), coastlines=True, **kwargs
):
    """
    Setup a Cartopy map for the Baja California bathymetry dataset.

    .. warning::

        All sample datasets in Verde are deprecated and will be
        **removed in Verde v2.0.0**.
        The tutorials/examples will transition to using
        `Ensaio <https://www.fatiando.org/ensaio/>`__ instead.

    Parameters
    ----------
    ax : matplotlib Axes
        The axes where the map is being plotted.
    region : list = [W, E, S, N]
        The boundaries of the map region in the coordinate system of the data.
    coastlines : bool
        If True the coastlines will be added to the plot.
    kwargs :
        All additional key-word arguments will be ignored. ``kwargs`` are
        accepted to guarantee backward compatibility.

    See also
    --------
    fetch_baja_bathymetry: Sample bathymetry data from Baja California.

    """
    if kwargs:
        warnings.warn(
            "All kwargs are being ignored. They are accepted to "
            + "guarantee backward compatibility."
        )
    _setup_map(
        ax,
        xticks=np.arange(-114, -105, 2),
        yticks=np.arange(21, 30, 2),
        coastlines=coastlines,
        region=region,
        crs=ccrs.PlateCarree(),
    )


def fetch_rio_magnetic():
    """
    Fetch total-field magnetic anomaly data from Rio de Janeiro, Brazil.

    .. warning::

        All sample datasets in Verde are deprecated and will be
        **removed in Verde v2.0.0**.
        The tutorials/examples will transition to using
        `Ensaio <https://www.fatiando.org/ensaio/>`__ instead.

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
        FutureWarning,
    )
    data_file = REGISTRY.fetch("rio-magnetic.csv.xz")
    data = pd.read_csv(data_file, compression="xz")
    return data


def setup_rio_magnetic_map(ax, region=(-42.6, -42, -22.5, -22)):
    """
    Setup a Cartopy map for the Rio de Janeiro magnetic anomaly dataset.

    .. warning::

        All sample datasets in Verde are deprecated and will be
        **removed in Verde v2.0.0**.
        The tutorials/examples will transition to using
        `Ensaio <https://www.fatiando.org/ensaio/>`__ instead.

    Parameters
    ----------
    ax : matplotlib Axes
        The axes where the map is being plotted.
    region : list = [W, E, S, N]
        The boundaries of the map region in the coordinate system of the data.

    See also
    --------
    fetch_rio_magnetic: Magnetic anomaly data from Rio de Janeiro, Brazil.

    """
    warnings.warn(
        "The Rio magnetic anomaly dataset is deprecated and will be removed "
        "in Verde v2.0.0. Use a different dataset instead.",
        FutureWarning,
    )
    _setup_map(
        ax,
        xticks=np.arange(-42.5, -42, 0.1),
        yticks=np.arange(-22.5, -21.99, 0.1),
        region=region,
        crs=ccrs.PlateCarree(),
    )


def fetch_california_gps():
    """
    Fetch sample GPS velocity data from California (the U.S. West coast).

    .. warning::

        All sample datasets in Verde are deprecated and will be
        **removed in Verde v2.0.0**.
        The tutorials/examples will transition to using
        `Ensaio <https://www.fatiando.org/ensaio/>`__ instead.

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
    _datasets_deprecation_warning()
    data_file = REGISTRY.fetch("california-gps.csv.xz")
    data = pd.read_csv(data_file, compression="xz")
    return data


def setup_california_gps_map(
    ax, region=(235.2, 245.3, 31.9, 42.3), coastlines=True, **kwargs
):
    """
    Setup a Cartopy map for the California GPS velocity dataset.

    .. warning::

        All sample datasets in Verde are deprecated and will be
        **removed in Verde v2.0.0**.
        The tutorials/examples will transition to using
        `Ensaio <https://www.fatiando.org/ensaio/>`__ instead.

    Parameters
    ----------
    ax : matplotlib Axes
        The axes where the map is being plotted.
    region : list = [W, E, S, N]
        The boundaries of the map region in the coordinate system of the data.
    coastlines : bool
        If True the coastlines will be added to the plot.
    kwargs :
        All additional key-word arguments will be ignored. ``kwargs`` are
        accepted to guarantee backward compatibility.

    See also
    --------
    fetch_california_gps: Sample GPS velocity data from California.

    """
    if kwargs:
        warnings.warn(
            "All kwargs are being ignored. They are accepted to "
            + "guarantee backward compatibility."
        )
    _setup_map(
        ax,
        xticks=np.arange(-124, -115, 4),
        yticks=np.arange(33, 42, 2),
        coastlines=coastlines,
        region=region,
        crs=ccrs.PlateCarree(),
    )


def fetch_texas_wind():
    """
    Fetch sample wind speed and air temperature data for Texas, USA.

    .. warning::

        All sample datasets in Verde are deprecated and will be
        **removed in Verde v2.0.0**.
        The tutorials/examples will transition to using
        `Ensaio <https://www.fatiando.org/ensaio/>`__ instead.

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
    _datasets_deprecation_warning()
    data_file = REGISTRY.fetch("texas-wind.csv")
    data = pd.read_csv(data_file)
    return data


def setup_texas_wind_map(ax, region=(-107, -93, 25.5, 37), coastlines=True, **kwargs):
    """
    Setup a Cartopy map for the Texas wind speed and air temperature dataset.

    .. warning::

        All sample datasets in Verde are deprecated and will be
        **removed in Verde v2.0.0**.
        The tutorials/examples will transition to using
        `Ensaio <https://www.fatiando.org/ensaio/>`__ instead.

    Parameters
    ----------
    ax : matplotlib Axes
        The axes where the map is being plotted.
    region : list = [W, E, S, N]
        The boundaries of the map region in the coordinate system of the data.
    coastlines : bool
        If True the coastlines will be added to the plot.
    kwargs :
        All additional key-word arguments will be ignored. ``kwargs`` are
        accepted to guarantee backward compatibility.

    See also
    --------
    fetch_texas_wind: Sample wind speed and air temperature data for Texas.

    """
    if kwargs:
        warnings.warn(
            "All kwargs are being ignored. They are accepted to "
            + "guarantee backward compatibility."
        )
    _setup_map(
        ax,
        xticks=np.arange(-106, -92, 3),
        yticks=np.arange(27, 38, 3),
        coastlines=coastlines,
        region=region,
        crs=ccrs.PlateCarree(),
    )
