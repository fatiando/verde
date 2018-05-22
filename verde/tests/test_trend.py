# pylint: disable=redefined-outer-name
"""
Test the trend estimators.
"""
import numpy as np
import numpy.testing as npt
import pytest

from ..trend import Trend, polynomial_power_combinations, trend_jacobian, \
    VectorTrend
from ..coordinates import grid_coordinates


@pytest.fixture()
def simple_model():
    "A single degree=1 model"
    east, north = grid_coordinates((1000, 5000, -5000, -1000), shape=(50, 50))
    coefs = [10, 2, -0.4]
    data = coefs[0] + coefs[1]*east + coefs[2]*north
    return (east, north), coefs, data


@pytest.fixture()
def simple_2d_model():
    "A single degree=1 2-component model"
    east, north = grid_coordinates((1000, 5000, -5000, -1000), shape=(50, 50))
    coefs = ([-20, 0.5, 3], [10, 2, -0.4])
    data = tuple(c[0] + c[1]*east + c[2]*north for c in coefs)
    return (east, north), coefs, data


@pytest.fixture()
def simple_3d_model():
    "A single degree=1 3-component model"
    east, north = grid_coordinates((1000, 5000, -5000, -1000), shape=(50, 50))
    coefs = ([-20, 0.5, 3], [10, 2, -0.4], [30, -10, -1.3])
    data = tuple(c[0] + c[1]*east + c[2]*north for c in coefs)
    return (east, north), coefs, data


def test_trend(simple_model):
    "Test the trend estimation on a simple problem"
    coords, coefs, data = simple_model
    trend = Trend(degree=1).fit(coords, data)
    npt.assert_allclose(trend.coef_, coefs)
    npt.assert_allclose(trend.predict(coords), data)


def test_trend_weights(simple_model):
    "Use weights to account for outliers"
    coords, coefs, data = simple_model
    data_out = data.copy()
    outlier = data_out[20, 20]*50
    data_out[20, 20] += outlier
    weights = np.ones_like(data)
    weights[20, 20] = 1e-10
    trend = Trend(degree=1).fit(coords, data_out, weights)
    npt.assert_allclose(trend.coef_, coefs)
    npt.assert_allclose((data_out - trend.predict(coords))[20, 20], outlier)
    npt.assert_allclose(trend.predict(coords), data)


def test_polynomial_combinations_fails():
    "Test failing conditions for the combinations"
    with pytest.raises(ValueError):
        polynomial_power_combinations(degree=0)
    with pytest.raises(ValueError):
        polynomial_power_combinations(degree=-10)


def test_trend_jacobian_fails():
    "Test failing conditions for the trend jacobian builder"
    east, north = np.arange(50), np.arange(30)
    with pytest.raises(ValueError):
        trend_jacobian((east, north), degree=1)


def test_vector_trend(simple_2d_model):
    "Test the vector trend estimation on a simple problem"
    coords, coefs, data = simple_2d_model
    trend = VectorTrend(degree=1).fit(coords, data)
    for i, coef in enumerate(coefs):
        npt.assert_allclose(trend.component_[i].coef_, coef)
        npt.assert_allclose(trend.predict(coords)[i], data[i])
        npt.assert_allclose(trend.score(coords, data), 1)


def test_vector_trend_3d(simple_3d_model):
    "Test the vector trend estimation on a simple problem"
    coords, coefs, data = simple_3d_model
    trend = VectorTrend(degree=1).fit(coords, data)
    for i, coef in enumerate(coefs):
        npt.assert_allclose(trend.component_[i].coef_, coef)
        npt.assert_allclose(trend.predict(coords)[i], data[i])
        npt.assert_allclose(trend.score(coords, data), 1)


def test_vector_trend_fails(simple_2d_model):
    "Test the vector trend estimation on a simple problem"
    coords, _, data = simple_2d_model
    trend = VectorTrend(degree=1)
    with pytest.raises(ValueError):
        trend.fit(coords, list(data))
    with pytest.raises(ValueError):
        trend.fit(coords, data, weights=[np.ones_like(data)]*2)


def test_vector_trend_weights(simple_2d_model):
    "Use weights to account for outliers"
    coords, coefs, data = simple_2d_model
    outlier = np.abs(data[0]).max()*3
    data_out = tuple(i.copy() for i in data)
    weights = tuple(np.ones_like(i) for i in data)
    for i, coef in enumerate(coefs):
        data_out[i][20, 20] += outlier
        weights[i][20, 20] = 1e-10
    trend = VectorTrend(degree=1).fit(coords, data_out, weights)
    for i, coef in enumerate(coefs):
        npt.assert_allclose(trend.component_[i].coef_, coef)
        npt.assert_allclose((data_out[i] - trend.predict(coords)[i])[20, 20],
                            outlier)
        npt.assert_allclose(trend.predict(coords)[i], data[i])
