"""
Operations with projections for grids, regions, etc.
"""
import numpy as np

from .coordinates import grid_coordinates, get_region, shape_to_spacing, check_region
from .utils import grid_to_table
from .scipygridder import ScipyGridder
from .blockreduce import BlockReduce
from .chain import Chain
from .mask import convexhull_mask


def project_region(region, projection):
    """
    Calculate the bounding box of a region in projected coordinates.

    Parameters
    ----------
    region : list = [W, E, S, N]
        The boundaries of a given region in Cartesian or geographic
        coordinates.
    projection : callable
        Should be a callable object (like a function) ``projection(easting,
        northing) -> (proj_easting, proj_northing)`` that takes in easting and
        northing coordinate arrays and returns projected northing and easting
        coordinate arrays.

    Returns
    -------
    proj_region : list = [W, E, S, N]
        The bounding box of the projected region.

    Examples
    --------

    >>> def projection(x, y):
    ...     return (2*x, -1*y)
    >>> project_region((3, 5, -9, -4), projection)
    (6.0, 10.0, 4.0, 9.0)

    """
    east, north = grid_coordinates(region, shape=(101, 101))
    east, north = projection(east.ravel(), north.ravel())
    return (east.min(), east.max(), north.min(), north.max())


def project_grid(grid, projection, method="linear", antialias=True, **kwargs):
    """
    Apply the given map projection to a grid and re-sample it.

    Creates a new grid in the projected coordinates by interpolating the
    original values using the chosen *method* (linear by default). Before
    interpolation, apply a blocked mean operation (:class:`~verde.BlockReduce`)
    to avoid aliasing when the projected coordinates become oversampled in some
    regions (which would cause the interpolation to down-sample the original
    data). For example, applying a polar projection results in oversampled data
    close to the pole.

    Points that fall outside the convex hull of the original data will be
    masked (see :func:`~verde.convexhull_mask`) since they are not constrained
    by any data points.

    Any arguments that can be passed to the
    :meth:`~verde.base.BaseGridder.grid` method of Verde gridders can be passed
    to this function as well. Use this to set a region and spacing (or shape)
    for the projected grid. The region and spacing must be in *projected
    coordinates*.

    If no region is provided, the bounding box of the projected data will be
    used. If no spacing or shape is provided, the shape of the input *grid*
    will be used for the projected grid.

    By default, the ``data_names`` argument will be set to the name of the data
    variable of the input *grid* (if it has been set).

    .. note::

        The interpolation methods are limited to what is available in Verde and
        there is only support for single 2D grids. For more sophisticated use
        cases, you might want to try
        `pyresample <https://github.com/pytroll/pyresample>`__ instead.

    Parameters
    ----------
    grid : :class:`xarray.DataArray`
        A single 2D grid of values. The first dimension is assumed to be the
        northing/latitude dimension while the second is assumed to be the
        easting/longitude dimension.
    projection : callable
        Should be a callable object (like a function) ``projection(easting,
        northing) -> (proj_easting, proj_northing)`` that takes in easting and
        northing coordinate arrays and returns projected northing and easting
        coordinate arrays.
    method : string or Verde gridder
        If a string, will use it to create a :class:`~verde.ScipyGridder` with
        the corresponding method (nearest, linear, or cubic). Otherwise, should
        be a gridder/estimator object, like :class:`~verde.Spline`. Default is
        ``"linear"``.
    antialias : bool
        If True, will run a :class:`~verde.BlockReduce` with a mean function to
        avoid aliasing when the projection results in oversampling of the data
        in some areas (for example, in polar projections). If False, will not
        run the blocked mean.

    Returns
    -------
    projected_grid : :class:`xarray.DataArray`
        The projected grid, interpolated with the given parameters.

    """
    if hasattr(grid, "data_vars"):
        raise ValueError(
            "Projecting xarray.Dataset is not currently supported. "
            "Please provide a DataArray instead."
        )
    if len(grid.dims) != 2:
        raise ValueError(
            "Projecting grids with number of dimensions other than 2 is not "
            "currently supported (dimensions of the given DataArray: {}).".format(
                len(grid.dims)
            )
        )

    # Can be set to None for some data arrays depending on how they are created
    # so we can't just rely on the default value for getattr.
    name = getattr(grid, "name", None)
    if name is None:
        name = "scalars"

    data = grid_to_table(grid).dropna()
    coordinates = projection(data[grid.dims[1]].values, data[grid.dims[0]].values)
    data_region = get_region(coordinates)

    region = kwargs.pop("region", data_region)
    shape = kwargs.pop("shape", grid.shape)
    spacing = kwargs.pop("spacing", shape_to_spacing(region, shape))

    check_region(region)

    steps = []
    if antialias:
        steps.append(
            ("mean", BlockReduce(np.mean, spacing=spacing, region=data_region))
        )
    if isinstance(method, str):
        steps.append(("spline", ScipyGridder(method)))
    else:
        steps.append(("spline", method))
    interpolator = Chain(steps)
    interpolator.fit(coordinates, data[name])

    projected = interpolator.grid(
        region=region,
        spacing=spacing,
        data_names=kwargs.pop("data_names", [name]),
        **kwargs
    )
    if method not in ["linear", "cubic"]:
        projected = convexhull_mask(coordinates, grid=projected)
    return projected[name]
