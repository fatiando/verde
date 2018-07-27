# pylint: disable=redefined-outer-name
"""
Test the Components meta-gridder.
"""
import numpy as np
import numpy.testing as npt
import pytest

from ..trend import Trend
from ..coordinates import grid_coordinates
from ..components import Components


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


def test_components_vector_trend(simple_2d_model):
    "Test the vector trend estimation on a simple problem"
    coords, coefs, data = simple_2d_model
    trend = Components([Trend(degree=1), Trend(degree=1)]).fit(coords, data)
    for i, coef in enumerate(coefs):
        npt.assert_allclose(trend.components[i].coef_, coef)
        npt.assert_allclose(trend.predict(coords)[i], data[i])
        npt.assert_allclose(trend.score(coords, data), 1)


def test_components_vector_trend_3d(simple_3d_model):
    "Test the vector trend estimation on a simple problem"
    coords, coefs, data = simple_3d_model
    trend = Components([Trend(degree=1), Trend(degree=1)]).fit(coords, data)
    for i, coef in enumerate(coefs):
        npt.assert_allclose(trend.components[i].coef_, coef)
        npt.assert_allclose(trend.predict(coords)[i], data[i])
        npt.assert_allclose(trend.score(coords, data), 1)


def test_components_vector_trend_fails(simple_2d_model):
    "Test the vector trend estimation on a simple problem"
    coords, _, data = simple_2d_model
    trend = Components([Trend(degree=1), Trend(degree=1)])
    with pytest.raises(ValueError):
        trend.fit(coords, list(data))
    with pytest.raises(ValueError):
        trend.fit(coords, data, weights=[np.ones_like(data)] * 2)


def test_components_vector_trend_weights(simple_2d_model):
    "Use weights to account for outliers"
    coords, coefs, data = simple_2d_model
    outlier = np.abs(data[0]).max() * 3
    data_out = tuple(i.copy() for i in data)
    weights = tuple(np.ones_like(i) for i in data)
    for i, coef in enumerate(coefs):
        data_out[i][20, 20] += outlier
        weights[i][20, 20] = 1e-10
    trend = Components([Trend(degree=1), Trend(degree=1)])
    trend.fit(coords, data_out, weights)
    for i, coef in enumerate(coefs):
        npt.assert_allclose(trend.components[i].coef_, coef)
        npt.assert_allclose((data_out[i] - trend.predict(coords)[i])[20, 20], outlier)
        npt.assert_allclose(trend.predict(coords)[i], data[i])
