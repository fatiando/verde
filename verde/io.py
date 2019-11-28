"""
Functions for input and output of grids in less common formats.
"""
import numpy as np
import xarray as xr


def load_surfer(fname, dtype="float64"):
    """
    Read data from a Surfer ASCII grid file.

    This function reads a Surfer grid file, masks any NaNs in the data,
    and outputs a :class:`xarray.DataArray` that contains easting and northing
    coordinates, data values, and associated file metadata.

    Surfer is a contouring, griding and surface mapping software
    from GoldenSoftware. The names and logos for Surfer and Golden
    Software are registered trademarks of Golden Software, Inc.

    http://www.goldensoftware.com/products/surfer

    Parameters
    ----------
    fname : str or file-like object
        Name or path of the Surfer grid file or an open file (or file-like)
        object.
    dtype : str or numpy.dtype
        The type of variable used for the data. Default is ``float64``. Use
        ``float32`` if the data are large and precision is not an issue.

    Returns
    ----------
    data : :class:`xarray.DataArray`
        A 2D grid with the data.

    """
    # Surfer ASCII grid structure
    # DSAA            Surfer ASCII GRD ID
    # nCols nRows     number of columns and rows
    # xMin xMax       X min max
    # yMin yMax       Y min max
    # zMin zMax       Z min max
    # z11 z21 z31 ... List of Z values

    # Only open a file if given a path instead of a file-like object
    ispath = not hasattr(fname, "readline")
    if ispath:
        input_file = open(fname, "r")
    else:
        input_file = fname
    try:
        grid_id, shape, region, data_range = _read_surfer_header(input_file)
        field = np.loadtxt(input_file, dtype=dtype)
        nans = field >= 1.70141e38
        if np.any(nans):
            field = np.ma.masked_where(nans, field)
        _check_surfer_integrity(field, shape, data_range)
        attrs = {"gridID": grid_id}
        if ispath:
            attrs["file"] = fname
        dims = ("northing", "easting")
        coords = {
            "northing": np.linspace(*region[2:], shape[0]),
            "easting": np.linspace(*region[:2], shape[1]),
        }
        data = xr.DataArray(field, coords=coords, dims=dims, attrs=attrs)
    finally:
        if ispath:
            input_file.close()
    return data


def _read_surfer_header(input_file):
    """
    Parse the header record of the grid file.

    The header contains information on the grid shape, region, and the minimum
    and maximum data values.

    Parameters
    ----------
    input_file : file-like object
        An open file to read from.

    Returns
    -------
    grid_id : str
        The ID of the Surfer ASCII grid.
    shape : tuple = (n_northing, n_easting)
        The number of grid points in the northing and easting dimension,
        respectively.
    region : tuple = (west, east, south, north)
        The grid region.
    data_range : list = [min, max]
        The minimum and maximum data values.

    """
    # DSAA is a Surfer ASCII GRD ID
    grid_id = input_file.readline().strip()
    # Read the grid shape (n_northing, n_easting)
    shape = tuple(int(i.strip()) for i in input_file.readline().split())
    # Our x points North, so the first thing we read north-south.
    south, north = [float(i.strip()) for i in input_file.readline().split()]
    west, east = [float(i.strip()) for i in input_file.readline().split()]
    region = (west, east, south, north)
    # The min and max of the data values (used for validation)
    data_range = [float(i.strip()) for i in input_file.readline().split()]
    return grid_id, shape, region, data_range


def _check_surfer_integrity(field, shape, data_range):
    """
    Check that the grid matches the information from the header.
    """
    if field.shape != shape:
        raise IOError(
            "Grid shape {} doesn't match shape read from header {}.".format(
                field.shape, shape
            )
        )
    field_range = [field.min(), field.max()]
    if not np.allclose(field_range, data_range):
        raise IOError(
            "Grid data range {} doesn't match range read from header {}.".format(
                field_range, data_range
            )
        )
