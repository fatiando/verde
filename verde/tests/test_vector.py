# pylint: disable=redefined-outer-name
"""
Test the vector data interpolators
"""
import pytest
import numpy as np
import numpy.testing as npt

from ..datasets.synthetic import CheckerBoard
from ..coordinates import grid_coordinates
from ..vector import Vector2D


@pytest.fixture
def data2d():
    "Make 2 component vector data"
    synth = CheckerBoard(region=(2, 8, -3, 3), w_east=20, w_north=20)
    coordinates = grid_coordinates(synth.region, spacing=0.5)
    data = tuple(synth.predict(coordinates) for i in range(2))
    return coordinates, data


def test_vector2d(data2d):
    "See if the exact solution works"
    coords, data = data2d
    spline = Vector2D().fit(coords, data)
    npt.assert_allclose(spline.score(coords, data), 1)
    npt.assert_allclose(spline.predict(coords), data, rtol=1e-3)
    # There should be 1 force per data point
    assert data[0].size == spline.force_coords_[0].size
    assert data[0].size * 2 == spline.force_.size
    npt.assert_allclose(spline.force_coords_, coords)


def test_vector2d_weights(data2d):
    "Use unit weights and a regular grid solution"
    coords, data = data2d
    weights = tuple(np.ones_like(data[0]) for i in range(2))
    spline = Vector2D(shape=(11, 11)).fit(coords, data, weights)
    npt.assert_allclose(spline.score(coords, data), 1, rtol=1e-3)
    npt.assert_allclose(spline.predict(coords), data, rtol=5e-2)
    assert spline.force_coords_[0].size == 11 * 11
    assert spline.force_.size == 2 * 11 * 11


def test_vector2d_fails(data2d):
    "Should fail if not given 2 data components"
    coords, data = data2d
    spline = Vector2D()
    with pytest.raises(ValueError):
        spline.fit(coords, data[0])
    with pytest.raises(ValueError):
        spline.fit(coords, data + coords)
