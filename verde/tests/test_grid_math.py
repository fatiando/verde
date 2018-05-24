"""
Test the grid math functions
"""
import numpy as np
import numpy.testing as npt

from ..coordinates import grid_coordinates, scatter_points
from ..grid_math import BlockReduce


def test_block_reduce():
    "Try reducing constant values in a regular grid"
    region = (-5, 0, 5, 10)
    east, north = grid_coordinates(region, spacing=0.1, pixel_register=True)
    data = 20*np.ones_like(east)
    reducer = BlockReduce(np.mean, spacing=1)
    block_coords, block_data = reducer.filter((east, north), data)
    assert len(block_coords[0]) == 25
    assert len(block_coords[1]) == 25
    assert len(block_data) == 25
    npt.assert_allclose(block_data, 20)
    npt.assert_allclose(block_coords[0][:5], np.linspace(-4.5, -0.5, 5))
    npt.assert_allclose(block_coords[1][::5], np.linspace(5.5, 9.5, 5))


def test_block_reduce_scatter():
    "Try reducing constant values in a dense enough scatter"
    region = (-5, 0, 5, 10)
    east, north = scatter_points(region, size=10000, random_state=0)
    data = 20*np.ones_like(east)
    block_coords, block_data = BlockReduce(
        np.mean, 1, region=region, center_coordinates=True).filter(
            (east, north), data)
    assert len(block_coords[0]) == 25
    assert len(block_coords[1]) == 25
    assert len(block_data) == 25
    npt.assert_allclose(block_data, 20)
    npt.assert_allclose(block_coords[0][:5], np.linspace(-4.5, -0.5, 5))
    npt.assert_allclose(block_coords[1][::5], np.linspace(5.5, 9.5, 5))


def test_block_reduce_weights():
    "Average with an outlier and zero weight should ignore the outlier"
    region = (-5, 0, 5, 10)
    size = 10000
    coords = scatter_points(region, size=size, random_state=0)
    data = 20*np.ones(size)
    weights = np.ones_like(data)
    outlier = 1000
    data[outlier] = 10000
    weights[outlier] = 0
    block_coords, block_data = BlockReduce(
        np.average, 1, region=region).filter(coords, data, weights)
    assert len(block_coords[0]) == 25
    assert len(block_coords[1]) == 25
    assert len(block_data) == 25
    npt.assert_allclose(block_data, 20)


def test_block_reduce_multiple_components():
    "Try reducing multiple components in a regular grid"
    region = (-5, 0, 5, 10)
    coords = grid_coordinates(region, spacing=0.1, pixel_register=True)
    data = 20*np.ones_like(coords[0]), -13*np.ones_like(coords[0])
    reducer = BlockReduce(np.mean, spacing=1)
    block_coords, block_data = reducer.filter(coords, data)
    assert len(block_coords[0]) == 25
    assert len(block_coords[1]) == 25
    npt.assert_allclose(block_coords[0][:5], np.linspace(-4.5, -0.5, 5))
    npt.assert_allclose(block_coords[1][::5], np.linspace(5.5, 9.5, 5))
    assert isinstance(block_data, tuple)
    assert len(block_data) == 2
    assert all(len(i) == 25 for i in block_data)
    npt.assert_allclose(block_data[0], 20)
    npt.assert_allclose(block_data[1], -13)


def test_block_reduce_multiple_weights():
    "Try reducing multiple components with weights"
    region = (-5, 0, 5, 10)
    size = 10000
    coords = scatter_points(region, size=size, random_state=10)
    data = 20*np.ones(size), -13*np.ones(size)
    outlier1 = 1000
    outlier2 = 3000
    data[0][outlier1] = 10000
    data[1][outlier2] = -10000
    weights = (np.ones(size), np.ones(size))
    weights[0][outlier1] = 0
    weights[1][outlier2] = 0
    reducer = BlockReduce(np.average, spacing=1)
    block_coords, block_data = reducer.filter(coords, data, weights)
    assert len(block_coords[0]) == 25
    assert len(block_coords[1]) == 25
    assert isinstance(block_data, tuple)
    assert len(block_data) == 2
    assert all(len(i) == 25 for i in block_data)
    npt.assert_allclose(block_data[0], 20)
    npt.assert_allclose(block_data[1], -13)
