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
    projection : callable or None
        If not None, then should be a callable object (like a function)
        ``projection(easting, northing) -> (proj_easting, proj_northing)`` that
        takes in easting and northing coordinate arrays and returns projected
        northing and easting coordinate arrays.

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
    Transform a grid using the given map projection.

    """
    if hasattr(grid, "data_vars"):
        raise ValueError("No Datasets!")
    if len(grid.dims) != 2:
        raise ValueError("Only 2D grids!")

    name = getattr(grid, "name", "scalars")

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
