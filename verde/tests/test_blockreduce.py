# pylint: disable=protected-access
"""
Test the grid math functions
"""
import pandas as pd
import numpy as np
import numpy.testing as npt
import pytest

from ..coordinates import grid_coordinates, scatter_points
from ..blockreduce import BlockReduce, BlockMean


def test_block_reduce():
    "Try reducing constant values in a regular grid"
    region = (-5, 0, 5, 10)
    east, north = grid_coordinates(region, spacing=0.1, pixel_register=True)
    data = 20 * np.ones_like(east)
    reducer = BlockReduce(np.mean, spacing=1)
    block_coords, block_data = reducer.filter((east, north), data)
    assert len(block_coords[0]) == 25
    assert len(block_coords[1]) == 25
    assert len(block_data) == 25
    npt.assert_allclose(block_data, 20)
    npt.assert_allclose(block_coords[0][:5], np.linspace(-4.5, -0.5, 5))
    npt.assert_allclose(block_coords[1][::5], np.linspace(5.5, 9.5, 5))


def test_block_reduce_shape():
    "Try reducing constant values in a regular grid using shape"
    region = (-5, 0, 5, 10)
    east, north = grid_coordinates(region, spacing=0.1, pixel_register=True)
    data = 20 * np.ones_like(east)
    reducer = BlockReduce(np.mean, shape=(5, 5))
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
    coordinates = scatter_points(region, size=10000, random_state=0)
    data = 20 * np.ones_like(coordinates[0])
    block_coords, block_data = BlockReduce(
        np.mean, 1, region=region, center_coordinates=True
    ).filter(coordinates, data)
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
    data = 20 * np.ones(size)
    weights = np.ones_like(data)
    outlier = 1000
    data[outlier] = 10000
    weights[outlier] = 0
    block_coords, block_data = BlockReduce(np.average, 1, region=region).filter(
        coords, data, weights
    )
    assert len(block_coords[0]) == 25
    assert len(block_coords[1]) == 25
    assert len(block_data) == 25
    npt.assert_allclose(block_data, 20)


def test_block_reduce_drop_coords():
    "Try reducing constant values in a regular grid dropping extra coordinates"
    region = (-5, 0, 5, 10)
    east, north, down, time = grid_coordinates(
        region, spacing=0.1, pixel_register=True, extra_coords=[70, 1]
    )
    data = 20 * np.ones_like(east)
    reducer = BlockReduce(np.mean, spacing=1, drop_coords=True)
    block_coords, block_data = reducer.filter((east, north, down, time), data)
    assert len(block_coords) == 2
    assert len(block_coords[0]) == 25
    assert len(block_coords[1]) == 25
    assert len(block_data) == 25
    npt.assert_allclose(block_data, 20)
    npt.assert_allclose(block_coords[0][:5], np.linspace(-4.5, -0.5, 5))
    npt.assert_allclose(block_coords[1][::5], np.linspace(5.5, 9.5, 5))


def test_block_reduce_multiple_coordinates():
    "Reduce constant values in a regular grid with n-dimensional coordinates"
    region = (-5, 0, 5, 10)
    east, north, down, time = grid_coordinates(
        region, spacing=0.1, pixel_register=True, extra_coords=[70, 1]
    )
    data = 20 * np.ones_like(east)
    reducer = BlockReduce(np.mean, spacing=1, drop_coords=False)
    block_coords, block_data = reducer.filter((east, north, down, time), data)
    assert len(block_coords) == 4
    assert len(block_coords[0]) == 25
    assert len(block_coords[1]) == 25
    assert len(block_coords[2]) == 25
    assert len(block_coords[3]) == 25
    assert len(block_data) == 25
    npt.assert_allclose(block_data, 20)
    npt.assert_allclose(block_coords[0][:5], np.linspace(-4.5, -0.5, 5))
    npt.assert_allclose(block_coords[1][::5], np.linspace(5.5, 9.5, 5))
    npt.assert_allclose(block_coords[2][::5], 70 * np.ones(5))
    npt.assert_allclose(block_coords[3][::5], np.ones(5))


def test_block_reduce_scatter_multiple_coordinates():
    "Reduce constant values in a dense scatter with n-dimensional coords"
    region = (-5, 0, 5, 10)
    coordinates = scatter_points(
        region, size=10000, random_state=0, extra_coords=[70, 1]
    )
    data = 20 * np.ones_like(coordinates[0])
    block_coords, block_data = BlockReduce(
        np.mean, 1, region=region, center_coordinates=True, drop_coords=False
    ).filter(coordinates, data)
    assert len(block_coords) == 4
    assert len(block_coords[0]) == 25
    assert len(block_coords[1]) == 25
    assert len(block_coords[2]) == 25
    assert len(block_coords[3]) == 25
    assert len(block_data) == 25
    npt.assert_allclose(block_data, 20)
    npt.assert_allclose(block_coords[0][:5], np.linspace(-4.5, -0.5, 5))
    npt.assert_allclose(block_coords[1][::5], np.linspace(5.5, 9.5, 5))
    npt.assert_allclose(block_coords[2][::5], 70 * np.ones(5))
    npt.assert_allclose(block_coords[3][::5], np.ones(5))


def test_block_reduce_multiple_components():
    "Try reducing multiple components in a regular grid"
    region = (-5, 0, 5, 10)
    coords = grid_coordinates(region, spacing=0.1, pixel_register=True)
    data = 20 * np.ones_like(coords[0]), -13 * np.ones_like(coords[0])
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
    data = 20 * np.ones(size), -13 * np.ones(size)
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


def test_blockmean_noweights():
    "Try blockmean with no weights"
    region = (-5, 0, 5, 10)
    east, north = grid_coordinates(region, spacing=0.1, pixel_register=True)
    data = 20 * np.ones_like(east)
    reducer = BlockMean(spacing=1)
    block_coords, block_data, block_weights = reducer.filter((east, north), data)
    assert len(block_coords[0]) == 25
    assert len(block_coords[1]) == 25
    assert len(block_data) == 25
    assert len(block_weights) == 25
    npt.assert_allclose(block_data, 20)
    npt.assert_allclose(block_weights, 1)
    npt.assert_allclose(block_coords[0][:5], np.linspace(-4.5, -0.5, 5))
    npt.assert_allclose(block_coords[1][::5], np.linspace(5.5, 9.5, 5))


def test_blockmean_noweights_multiple_components():
    "Try blockmean with no weights and multiple data components"
    region = (-5, 0, 5, 10)
    east, north = grid_coordinates(region, spacing=0.1, pixel_register=True)
    data = 20 * np.ones_like(east)
    reducer = BlockMean(spacing=1)
    block_coords, block_data, block_weights = reducer.filter(
        (east, north), (data, data)
    )
    assert len(block_coords[0]) == 25
    assert len(block_coords[1]) == 25
    npt.assert_allclose(block_coords[0][:5], np.linspace(-4.5, -0.5, 5))
    npt.assert_allclose(block_coords[1][::5], np.linspace(5.5, 9.5, 5))
    for datai, weighti in zip(block_data, block_weights):
        assert len(datai) == 25
        assert len(weighti) == 25
        npt.assert_allclose(datai, 20)
        npt.assert_allclose(weighti, 1)


def test_blockmean_noweights_table():
    "Try blockmean with no weights using a known blocked data table"
    reducer = BlockMean(spacing=1)
    table = pd.DataFrame(dict(data0=[1, 2, 10, 20, 5, 5], block=[1, 1, 2, 2, 3, 3]))
    mean, variance = reducer._blocked_mean_variance(table, 1)
    npt.assert_allclose(mean[0], [1.5, 15, 5])
    # The variance is calculated with 1 degree-of-freedom so it's divided by
    # N-1 instead of N because this is a sample variance, not a population
    # variance.
    npt.assert_allclose(variance[0], [0.5, 50, 0])


def test_blockmean_uncertainty_weights():
    "Try blockmean with uncertainty weights"
    region = (-2, 0, 6, 8)
    # This will be a 4x4 data grid that will be split into 2x2 blocks
    coords = grid_coordinates(region, spacing=0.5, pixel_register=True)
    data = 102.4 * np.ones_like(coords[0])
    uncertainty = np.ones_like(data)
    # Set a higher uncertainty for the first block
    uncertainty[:2, :2] = 2
    weights = 1 / uncertainty ** 2
    reducer = BlockMean(spacing=1, uncertainty=True)
    # Uncertainty propagation can only work if weights are given
    with pytest.raises(ValueError):
        reducer.filter(coords, data)
    block_coords, block_data, block_weights = reducer.filter(coords, data, weights)
    assert len(block_coords[0]) == 4
    assert len(block_coords[1]) == 4
    assert len(block_data) == 4
    assert len(block_weights) == 4
    npt.assert_allclose(block_data, 102.4)
    npt.assert_allclose(block_weights, [0.25, 1, 1, 1])
    npt.assert_allclose(block_coords[0][:2], [-1.5, -0.5])
    npt.assert_allclose(block_coords[1][::2], [6.5, 7.5])


def test_blockmean_variance_weights():
    "Try blockmean with variance weights"
    region = (-2, 0, 6, 8)
    # This will be a 4x4 data grid that will be split into 2x2 blocks
    coords = grid_coordinates(region, spacing=0.5, pixel_register=True)
    data = 102.4 * np.ones_like(coords[0])
    uncertainty = np.ones_like(data)
    # Set a higher uncertainty for the first block
    uncertainty[:2, :2] = 2
    weights = 1 / uncertainty ** 2
    reducer = BlockMean(spacing=1, uncertainty=False)
    block_coords, block_data, block_weights = reducer.filter(coords, data, weights)
    assert len(block_coords[0]) == 4
    assert len(block_coords[1]) == 4
    assert len(block_data) == 4
    assert len(block_weights) == 4
    npt.assert_allclose(block_data, 102.4)
    # The uncertainty in the first block shouldn't matter because the variance
    # is still zero, so the weights should be 1
    npt.assert_allclose(block_weights, [1, 1, 1, 1])
    npt.assert_allclose(block_coords[0][:2], [-1.5, -0.5])
    npt.assert_allclose(block_coords[1][::2], [6.5, 7.5])
