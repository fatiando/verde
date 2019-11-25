"""
Distance calculations between points.
"""
import numpy as np

from .utils import kdtree
from .base.utils import n_1d_arrays


def median_distance(coordinates, k_nearest=1, projection=None):
    """
    Median distance between the *k* nearest neighbors of each point.

    For each point specified in *coordinates*, calculate the median of the
    distance to its *k_nearest* neighbors among the other points in the
    dataset. Sparse uniformly spaced datasets can use *k_nearest* of 1.
    Datasets with points clustered into tight groups (e.g., densely sampled
    along a flight line or ship treck) will have very small distances to the
    closest neighbors, which is not representative of the actual median spacing
    because it doesn't take the spacing between lines into account. In these
    cases, a median of the 10 or 20 nearest neighbors might be more
    representative.

    The distances calculated are Cartesian (l2-norms) and horizontal (only the
    first two coordinate arrays are used). If the coordinates are in geodetic
    latitude and longitude, provide a *projection* function to convert them to
    Cartesian before doing the computations.

    .. note::

        If installed, package ``pykdtree`` will be used instead of
        :class:`scipy.spatial.cKDTree` for better performance.

    Parameters
    ----------
    coordinates : tuple of arrays
        Arrays with the coordinates of each data point. Should be in the
        following order: (easting, northing, vertical, ...). Only the first two
        coordinates (assumed to be the horizontal) will be used in the distance
        computations.
    k_nearest : int
        Will calculate the median of the *k* nearest neighbors of each point. A
        value of 1 will result in the distance to nearest neighbor of each data
        point.
    projection : callable or None
        If not None, then should be a callable object (like a function)
        ``projection(easting, northing) -> (proj_easting, proj_northing)`` that
        takes in easting and northing coordinate arrays and returns projected
        northing and easting coordinate arrays.

    Returns
    -------
    distances : array
        An array with the median distances to the *k* nearest neighbors of each
        data point. The array will have the same shape as the input coordinate
        arrays.

    Examples
    --------

    >>> import verde as vd
    >>> import numpy as np
    >>> coords = vd.grid_coordinates((5, 10, -20, -17), spacing=1)
    >>> # The nearest neighbor distance should be the grid spacing
    >>> distance = median_distance(coords, k_nearest=1)
    >>> np.allclose(distance, 1)
    True
    >>> # The distance has the same shape as the coordinate arrays
    >>> print(distance.shape, coords[0].shape)
    (4, 6) (4, 6)
    >>> # The 2 nearest points should also all be at a distance of 1
    >>> distance = median_distance(coords, k_nearest=2)
    >>> np.allclose(distance, 1)
    True
    >>> # The 3 nearest points are at a distance of 1 but on the corners they
    >>> # are [1, 1, sqrt(2)] away. The median for these points is also 1.
    >>> distance = median_distance(coords, k_nearest=3)
    >>> np.allclose(distance, 1)
    True
    >>> # The 4 nearest points are at a distance of 1 but on the corners they
    >>> # are [1, 1, sqrt(2), 2] away.
    >>> distance = median_distance(coords, k_nearest=4)
    >>> print("{:.2f}".format(np.median([1, 1, np.sqrt(2), 2])))
    1.21
    >>> for line in distance:
    ...     print(" ".join(["{:.2f}".format(i) for i in line]))
    1.21 1.00 1.00 1.00 1.00 1.21
    1.00 1.00 1.00 1.00 1.00 1.00
    1.00 1.00 1.00 1.00 1.00 1.00
    1.21 1.00 1.00 1.00 1.00 1.21

    """
    shape = np.broadcast(*coordinates[:2]).shape
    coords = n_1d_arrays(coordinates, n=2)
    if projection is not None:
        coords = projection(*coords)
    tree = kdtree(coords)
    # The k=1 nearest point is going to be the point itself (with a distance of
    # zero) because we don't remove each point from the dataset in turn. We
    # don't care about that distance so start with the second closest. Only get
    # the first element returned (the distance) and ignore the rest (the
    # neighbor indices).
    k_distances = tree.query(np.transpose(coords), k=k_nearest + 1)[0][:, 1:]
    distances = np.median(k_distances, axis=1)
    return distances.reshape(shape)
