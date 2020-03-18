# pylint: disable=redefined-outer-name
"""
Test the Gradient estimator
"""
import numpy as np
import numpy.testing as npt
import pytest

from ..gradient import Gradient
from ..trend import Trend
from ..vector import Vector
from ..chain import Chain
from ..coordinates import grid_coordinates


@pytest.fixture()
def polynomial():
    "A second degree polynomial data"
    region = (0, 5000, -10000, -5000)
    spacing = 50
    coordinates = grid_coordinates(region, spacing=spacing)
    baselevel = 100
    east_coef = -0.5
    north_coef = 1.5
    data = (
        baselevel + east_coef * coordinates[0] ** 2 + north_coef * coordinates[1] ** 2
    )
    east_deriv = 2 * east_coef * coordinates[0]
    north_deriv = 2 * north_coef * coordinates[1]
    return coordinates, data, east_deriv, north_deriv


def test_gradient(polynomial):
    "Check that the gradient of polynomial trend is estimated correctly"
    coordinates, data, true_east_deriv, true_north_deriv = polynomial

    trend = Trend(degree=2).fit(coordinates, data)
    east_deriv = Gradient(trend, step=10, direction=(1, 0)).predict(coordinates)
    north_deriv = Gradient(trend, step=10, direction=(0, 1)).predict(coordinates)

    npt.assert_allclose(true_east_deriv, east_deriv)
    npt.assert_allclose(true_north_deriv, north_deriv)


def test_gradient_fails_wrong_dimensions(polynomial):
    "An error should be raised if dimensions of coordinates != directions"
    coordinates, data = polynomial[:2]

    trend = Trend(degree=2).fit(coordinates, data)
    with pytest.raises(ValueError):
        Gradient(trend, step=10, direction=(1, 0, 1)).predict(coordinates)
    with pytest.raises(ValueError):
        Gradient(trend, step=10, direction=(1, 0)).predict((coordinates[0],))


def test_gradient_grid(polynomial):
    "Make sure the grid method works as expected"
    coordinates, data, true_east_deriv = polynomial[:3]

    trend = Trend(degree=2).fit(coordinates, data)
    deriv = Gradient(trend, step=10, direction=(1, 0)).grid(spacing=50)

    npt.assert_allclose(true_east_deriv, deriv.scalars.values)


def test_gradient_direction(polynomial):
    "Check that the gradient in a range of directions"
    coordinates, data, true_east_deriv, true_north_deriv = polynomial
    trend = Trend(degree=2).fit(coordinates, data)
    for azimuth in np.radians(np.linspace(0, 360, 60)):
        direction = (np.sin(azimuth), np.cos(azimuth))
        true_deriv = true_east_deriv * direction[0] + true_north_deriv * direction[1]
        deriv = Gradient(trend, step=10, direction=direction).predict(coordinates)
        npt.assert_allclose(true_deriv, deriv)


def test_gradient_fit(polynomial):
    "Check that calling fit on the Gradient works as expected"
    coordinates, data, true_east_deriv, true_north_deriv = polynomial

    trend = Trend(degree=2)
    east_deriv = (
        Gradient(trend, step=10, direction=(1, 0))
        .fit(coordinates, data)
        .predict(coordinates)
    )
    north_deriv = (
        Gradient(trend, step=10, direction=(0, 1))
        .fit(coordinates, data)
        .predict(coordinates)
    )

    npt.assert_allclose(true_east_deriv, east_deriv)
    npt.assert_allclose(true_north_deriv, north_deriv)


def test_gradient_vector(polynomial):
    "Make sure putting Gradients in a Vector works"
    coordinates, data, true_east_deriv, true_north_deriv = polynomial

    trend = Trend(degree=2)
    gradient = Vector(
        [
            Gradient(trend, step=10, direction=(1, 0)),
            Gradient(trend, step=10, direction=(0, 1)),
        ]
    )
    # This is wasteful because it fits the same trend twice so should not
    # really be done in practice.
    gradient.fit(coordinates, (data, data))
    deriv = gradient.grid(spacing=50)

    npt.assert_allclose(true_east_deriv, deriv.east_component.values)
    npt.assert_allclose(true_north_deriv, deriv.north_component.values)


def test_gradient_chain(polynomial):
    "Make sure putting Gradients in a Chain works"
    coordinates, data, true_east_deriv = polynomial[:3]

    trend = Trend(degree=2)
    gradient = Chain([("east", Gradient(trend, step=10, direction=(1, 0)))])
    gradient.fit(coordinates, data)
    deriv = gradient.predict(coordinates)
    npt.assert_allclose(true_east_deriv, deriv)
