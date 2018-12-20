"""
Test distance calculation functions.
"""
import numpy as np
import numpy.testing as npt

from ..distances import median_distance
from ..coordinates import grid_coordinates


def test_distance_nearest():
    "On a regular grid, distances should be straight forward for 1st neighbor"
    spacing = 0.5
    coords = grid_coordinates((5, 10, -20, -17), spacing=spacing)

    distance = median_distance(coords, k_nearest=1)
    # The nearest neighbor distance should be the grid spacing
    npt.assert_allclose(distance, spacing)
    assert distance.shape == coords[0].shape

    # The shape should be the same even if we ravel the coordinates
    coords = tuple(coord.ravel() for coord in coords)
    distance = median_distance(coords, k_nearest=1)
    # The nearest neighbor distance should be the grid spacing
    npt.assert_allclose(distance, spacing)
    assert distance.shape == coords[0].shape


def test_distance_k_nearest():
    "Check the median results for k nearest neighbors"
    coords = grid_coordinates((5, 10, -20, -17), spacing=1)

    # The 2 nearest points should also all be at a distance of 1 spacing
    distance = median_distance(coords, k_nearest=2)
    npt.assert_allclose(distance, 1)

    # The 3 nearest points are at a distance of 1 but on the corners they are
    # [1, 1, sqrt(2)] away. The median for these points is also 1.
    distance = median_distance(coords, k_nearest=3)
    npt.assert_allclose(distance, np.median([1, 1, np.sqrt(2)]))

    # The 4 nearest points are at a distance of 1 but on the corners they are
    # [1, 1, sqrt(2), 2] away.
    distance = median_distance(coords, k_nearest=4)
    true = np.ones_like(coords[0])
    corners = np.median([1, 1, np.sqrt(2), 2])
    true[0, 0] = true[0, -1] = true[-1, 0] = true[-1, -1] = corners
    npt.assert_allclose(distance, true)


def test_distance_nearest_projection():
    "Use a simple projection to test the mechanics"
    spacing = 0.3
    coords = grid_coordinates((5, 10, -20, -17), spacing=spacing, adjust="region")
    # The projection multiplies by 2 so the nearest should be 2*spacing
    distance = median_distance(
        coords, k_nearest=1, projection=lambda i, j: (i * 2, j * 2)
    )
    npt.assert_allclose(distance, spacing * 2)
    assert distance.shape == coords[0].shape
