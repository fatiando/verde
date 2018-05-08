"""
Test the misc utilities in verde
"""
import pytest
import numpy as np
import numpy.testing as npt

from ..utils import check_region, spacing_to_shape
from .. import profile_coordinates, grid_coordinates, block_reduce, \
    scatter_points


def test_block_reduce():
    "Try reducing constant values in a regular grid"
    region = (-5, 0, 5, 10)
    east, north = grid_coordinates(region, spacing=0.1)
    data = 20*np.ones_like(east)
    re_east, re_north, re_data = block_reduce(
        east, north, data, np.mean, spacing=1)
    assert len(re_east) == len(re_north) == len(re_data) == 25
    npt.assert_allclose(re_data, 20)
    npt.assert_allclose(re_east[:5], np.linspace(-4.5, -0.5, 5))
    npt.assert_allclose(re_north[::5], np.linspace(5.5, 9.5, 5))


def test_block_reduce_scatter():
    "Try reducing constant values in a dense enough scatter"
    region = (-5, 0, 5, 10)
    east, north = scatter_points(region, size=10000, random_state=0)
    data = 20*np.ones_like(east)
    re_east, re_north, re_data = block_reduce(
        east, north, data, np.mean, spacing=1, region=region,
        center_coordinates=True)
    assert len(re_east) == len(re_north) == len(re_data) == 25
    npt.assert_allclose(re_data, 20)
    npt.assert_allclose(re_east[:5], np.linspace(-4.5, -0.5, 5))
    npt.assert_allclose(re_north[::5], np.linspace(5.5, 9.5, 5))


def test_spacing_to_shape():
    "Check that correct spacing and region are returned"
    region = (-10, 0, 0, 5)

    shape, new_region = spacing_to_shape(region, spacing=2.5, adjust='spacing')
    npt.assert_allclose(shape, (3, 5))
    npt.assert_allclose(new_region, region)

    shape, new_region = spacing_to_shape(region, spacing=(2.5, 2),
                                         adjust='spacing')
    npt.assert_allclose(shape, (3, 6))
    npt.assert_allclose(new_region, region)

    shape, new_region = spacing_to_shape(region, spacing=2.6, adjust='spacing')
    npt.assert_allclose(shape, (3, 5))
    npt.assert_allclose(new_region, region)

    shape, new_region = spacing_to_shape(region, spacing=2.4, adjust='spacing')
    npt.assert_allclose(shape, (3, 5))
    npt.assert_allclose(new_region, region)

    shape, new_region = spacing_to_shape(region, spacing=(2.4, 1.9),
                                         adjust='spacing')
    npt.assert_allclose(shape, (3, 6))
    npt.assert_allclose(new_region, region)

    shape, new_region = spacing_to_shape(region, spacing=2.6, adjust='region')
    npt.assert_allclose(shape, (3, 5))
    npt.assert_allclose(new_region, (-10, 0.4, 0, 5.2))

    shape, new_region = spacing_to_shape(region, spacing=(2.6, 2.4),
                                         adjust='region')
    npt.assert_allclose(shape, (3, 5))
    npt.assert_allclose(new_region, (-10, -0.4, 0, 5.2))


def test_spacing_to_shape_fails():
    "Should fail if more than 2 spacings are given"
    with pytest.raises(ValueError):
        spacing_to_shape((0, 1, 0, 1), (1, 2, 3), adjust='region')


def test_grid_coordinates_fails():
    "Check failures for invalid arguments"
    region = (0, 1, 0, 10)
    shape = (10, 20)
    spacing = 0.5
    # Make sure it doesn't fail for these parameters
    grid_coordinates(region, shape)
    grid_coordinates(region, spacing=spacing)

    with pytest.raises(ValueError):
        grid_coordinates(region, shape=shape, spacing=spacing)
    with pytest.raises(ValueError):
        grid_coordinates(region, shape=None, spacing=None)
    with pytest.raises(ValueError):
        grid_coordinates(region, spacing=spacing, adjust='invalid adjust')


def test_check_region():
    "Make sure an exception is raised for bad regions"
    with pytest.raises(ValueError):
        check_region([])
    with pytest.raises(ValueError):
        check_region([1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        check_region([1, 2, 3])
    with pytest.raises(ValueError):
        check_region([1, 2, 3, 1])
    with pytest.raises(ValueError):
        check_region([2, 1, 3, 4])
    with pytest.raises(ValueError):
        check_region([-1, -2, -4, -3])
    with pytest.raises(ValueError):
        check_region([-2, -1, -2, -3])


def test_profile_coordiantes_fails():
    "Should raise an exception for invalid input"
    with pytest.raises(ValueError):
        profile_coordinates((0, 1), (1, 2), size=0)
    with pytest.raises(ValueError):
        profile_coordinates((0, 1), (1, 2), size=-10)
    with pytest.raises(ValueError):
        profile_coordinates((0, 1), (1, 2), size=10, coordinate_system="meh")
    with pytest.raises(NotImplementedError):
        profile_coordinates((0, 1), (1, 2), size=10,
                            coordinate_system="geographic")
