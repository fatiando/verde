# pylint: disable=redefined-outer-name
"""
Test the vector data interpolators
"""
import numpy as np
import numpy.testing as npt
import pytest

from ..datasets.synthetic import CheckerBoard
from ..coordinates import grid_coordinates
from ..trend import Trend
from ..base import n_1d_arrays
from .utils import requires_numba
from ..vector import VectorSpline2D, Vector


@pytest.fixture
def data2d():
    "Make 2 component vector data"
    synth = CheckerBoard()
    coordinates = grid_coordinates(synth.region, shape=(15, 20))
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
    outlier = 10
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


@requires_numba
def test_vector2d_predict_implementations(data2d):
    "Compare the numba and numpy implementations."
    coords, data = data2d
    pred_numpy = VectorSpline2D(engine="numpy").fit(coords, data).predict(coords)
    pred_numba = VectorSpline2D(engine="numba").fit(coords, data).predict(coords)
    npt.assert_allclose(pred_numpy, pred_numba, atol=1e-4)


###############################################################################
# Test the Vector meta-gridder


@pytest.fixture()
def simple_2d_model():
    "A single degree=1 2-component model"
    east, north = grid_coordinates((1000, 5000, -5000, -1000), shape=(50, 50))
    coefs = ([-20, 0.5, 3], [10, 2, -0.4])
    data = tuple(c[0] + c[1] * east + c[2] * north for c in coefs)
    return (east, north), coefs, data


@pytest.fixture()
def simple_3d_model():
    "A single degree=1 3-component model"
    east, north = grid_coordinates((1000, 5000, -5000, -1000), shape=(50, 50))
    coefs = ([-20, 0.5, 3], [10, 2, -0.4], [30, -10, -1.3])
    data = tuple(c[0] + c[1] * east + c[2] * north for c in coefs)
    return (east, north), coefs, data


def test_vector_trend(simple_2d_model):
    "Test the vector trend estimation on a simple problem"
    coords, coefs, data = simple_2d_model
    trend = Vector([Trend(degree=1), Trend(degree=1)]).fit(coords, data)
    for i, coef in enumerate(coefs):
        npt.assert_allclose(trend.components[i].coef_, coef)
        npt.assert_allclose(trend.predict(coords)[i], data[i])
        npt.assert_allclose(trend.score(coords, data), 1)


def test_vector_trend_3d(simple_3d_model):
    "Test the vector trend estimation on a simple problem"
    coords, coefs, data = simple_3d_model
    trend = Vector([Trend(degree=1), Trend(degree=1), Trend(degree=1)])
    trend.fit(coords, data)
    for i, coef in enumerate(coefs):
        npt.assert_allclose(trend.components[i].coef_, coef)
        npt.assert_allclose(trend.predict(coords)[i], data[i])
        npt.assert_allclose(trend.score(coords, data), 1)


def test_vector_trend_fails(simple_2d_model):
    "Test the vector trend estimation on a simple problem"
    coords, _, data = simple_2d_model
    trend = Vector([Trend(degree=1), Trend(degree=1)])
    with pytest.raises(ValueError):
        trend.fit(coords, list(data))
    with pytest.raises(ValueError):
        trend.fit(coords, data, weights=[np.ones_like(data)] * 2)


def test_vector_trend_weights(simple_2d_model):
    "Use weights to account for outliers"
    coords, coefs, data = simple_2d_model
    outlier = np.abs(data[0]).max() * 3
    data_out = tuple(i.copy() for i in data)
    weights = tuple(np.ones_like(i) for i in data)
    for i, coef in enumerate(coefs):
        data_out[i][20, 20] += outlier
        weights[i][20, 20] = 1e-10
    trend = Vector([Trend(degree=1), Trend(degree=1)])
    trend.fit(coords, data_out, weights)
    for i, coef in enumerate(coefs):
        npt.assert_allclose(trend.components[i].coef_, coef)
        npt.assert_allclose((data_out[i] - trend.predict(coords)[i])[20, 20], outlier)
        npt.assert_allclose(trend.predict(coords)[i], data[i])
