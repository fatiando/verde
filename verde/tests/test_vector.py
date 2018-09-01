# pylint: disable=redefined-outer-name
"""
Test the vector data interpolators
"""
import pytest
import numpy as np
import numpy.testing as npt

from ..datasets.synthetic import CheckerBoard
from ..coordinates import grid_coordinates
from ..vector import VectorSpline2D
from ..utils import n_1d_arrays
from .utils import requires_numba


@pytest.fixture
def data2d():
    "Make 2 component vector data"
    synth = CheckerBoard()
    coordinates = grid_coordinates(synth.region, shape=(30, 25))
    data = tuple(synth.predict(coordinates).ravel() for i in range(2))
    return tuple(i.ravel() for i in coordinates), data


def test_vector2d(data2d):
    "See if the exact solution works"
    coords, data = data2d
    spline = VectorSpline2D().fit(coords, data)
    npt.assert_allclose(spline.score(coords, data), 1)
    npt.assert_allclose(spline.predict(coords), data, rtol=1e-2, atol=1)
    # There should be 1 force per data point
    assert data[0].size == spline.force_coords[0].size
    assert data[0].size * 2 == spline.force_.size
    npt.assert_allclose(spline.force_coords, n_1d_arrays(coords, n=2))


def test_vector2d_weights(data2d):
    "Use unit weights and a regular grid solution"
    coords, data = data2d
    outlier = 100
    outlier_value = 100000
    data_outlier = tuple(i.copy() for i in data)
    data_outlier[0][outlier] += outlier_value
    data_outlier[1][outlier] += outlier_value
    weights = tuple(np.ones_like(data_outlier[0]) for i in range(2))
    weights[0][outlier] = 1e-10
    weights[1][outlier] = 1e-10
    spline = VectorSpline2D(damping=1e-8).fit(coords, data_outlier, weights)
    npt.assert_allclose(spline.score(coords, data), 1, atol=0.01)
    npt.assert_allclose(spline.predict(coords), data, rtol=1e-2, atol=5)


def test_vector2d_fails(data2d):
    "Should fail if not given 2 data components"
    coords, data = data2d
    spline = VectorSpline2D()
    with pytest.raises(ValueError):
        spline.fit(coords, data[0])
    with pytest.raises(ValueError):
        spline.fit(coords, data + coords)


@requires_numba
def test_vector2d_jacobian_implementations(data2d):
    "Compare the numba and numpy implementations."
    coords = data2d[0]
    jac_numpy = VectorSpline2D(engine="numpy").jacobian(coords, coords)
    jac_numba = VectorSpline2D(engine="numba").jacobian(coords, coords)
    npt.assert_allclose(jac_numpy, jac_numba)
