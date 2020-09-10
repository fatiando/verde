"""
Test the grid masking functions
"""
import numpy as np
import numpy.testing as npt
import xarray as xr
import pytest

from ..mask import distance_mask, convexhull_mask
from ..coordinates import grid_coordinates


def test_convexhull_mask():
    "Check that the mask works for basic input"
    region = (0, 5, -10, -4)
    coords = grid_coordinates(region, spacing=1)
    data_coords = ((2, 3, 2, 3), (-9, -9, -6, -6))
    mask = convexhull_mask(data_coords, coordinates=coords)
    true = [
        [False, False, False, False, False, False],
        [False, False, True, True, False, False],
        [False, False, True, True, False, False],
        [False, False, True, True, False, False],
        [False, False, True, True, False, False],
        [False, False, False, False, False, False],
        [False, False, False, False, False, False],
    ]
    assert mask.tolist() == true


def test_convexhull_mask_projection():
    "Check that the mask works when given a projection"
    region = (0, 5, -10, -4)
    coords = grid_coordinates(region, spacing=1)
    data_coords = ((2, 3, 2, 3), (-9, -9, -6, -6))
    # For a linear projection, the result should be the same since there is no
    # area change in the data.
    mask = convexhull_mask(
        data_coords, coordinates=coords, projection=lambda e, n: (10 * e, 10 * n),
    )
    true = [
        [False, False, False, False, False, False],
        [False, False, True, True, False, False],
        [False, False, True, True, False, False],
        [False, False, True, True, False, False],
        [False, False, True, True, False, False],
        [False, False, False, False, False, False],
        [False, False, False, False, False, False],
    ]
    assert mask.tolist() == true


def test_distance_mask():
    "Check that the mask works for basic input"
    region = (0, 5, -10, -4)
    coords = grid_coordinates(region, spacing=1)
    mask = distance_mask((2.5, -7.5), maxdist=2, coordinates=coords)
    true = [
        [False, False, False, False, False, False],
        [False, False, True, True, False, False],
        [False, True, True, True, True, False],
        [False, True, True, True, True, False],
        [False, False, True, True, False, False],
        [False, False, False, False, False, False],
        [False, False, False, False, False, False],
    ]
    assert mask.tolist() == true


def test_distance_mask_projection():
    "Check that the mask works when given a projection"
    region = (0, 5, -10, -4)
    coords = grid_coordinates(region, spacing=1)
    mask = distance_mask(
        (2.5, -7.5),
        maxdist=20,
        coordinates=coords,
        projection=lambda e, n: (10 * e, 10 * n),
    )
    true = [
        [False, False, False, False, False, False],
        [False, False, True, True, False, False],
        [False, True, True, True, True, False],
        [False, True, True, True, True, False],
        [False, False, True, True, False, False],
        [False, False, False, False, False, False],
        [False, False, False, False, False, False],
    ]
    assert mask.tolist() == true


def test_distance_mask_grid():
    "Check that the mask works for grid input"
    region = (0, 5, -10, -4)
    shape = (7, 6)
    east, north = grid_coordinates(region, shape=shape)
    coords = {"easting": east[0, :], "northing": north[:, 0]}
    data_vars = {"scalars": (["northing", "easting"], np.ones(shape))}
    grid = xr.Dataset(data_vars, coords=coords)
    masked = distance_mask((2.5, -7.5), maxdist=2, grid=grid)
    true = [
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, 1, 1, np.nan, np.nan],
        [np.nan, 1, 1, 1, 1, np.nan],
        [np.nan, 1, 1, 1, 1, np.nan],
        [np.nan, np.nan, 1, 1, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    ]
    npt.assert_array_equal(true, masked.scalars.values)


def test_distance_mask_missing_args():
    "Check that the function fails if no coordinates or grid are given"
    with pytest.raises(ValueError):
        distance_mask((2.5, -7.5), maxdist=2)


def test_distance_mask_wrong_shapes():
    "Check that the function fails if coordinates have different shapes"
    coords = np.ones((10, 3)), np.zeros((2, 4))
    with pytest.raises(ValueError):
        distance_mask((2.5, -7.5), maxdist=2, coordinates=coords)
