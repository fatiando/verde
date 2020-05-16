"""
Utilities for calculating gradients of input data.
"""
import numpy as np

from .base import BaseGridder
from .base.utils import check_data


class Derivative(BaseGridder):
    """
    """

    def __init__(self, estimator, step, direction):
        super().__init__()
        self.estimator = estimator
        self.step = step
        self.direction = direction

    @property
    def region_(self):
        "The region of the data used to fit this estimator."
        return self.estimator.region_

    def predict(self, coordinates):
        """
        """
        direction = normalize_direction(self.direction)
        if len(coordinates) != len(direction):
            raise ValueError("Wrong number of dimensions")
        forward_coords = tuple(
            coordinate + component * self.step / 2
            for coordinate, component in zip(coordinates, direction)
        )
        backward_coords = tuple(
            coordinate - component * self.step / 2
            for coordinate, component in zip(coordinates, direction)
        )
        forward = check_data(self.estimator.predict(forward_coords))
        backward = check_data(self.estimator.predict(backward_coords))
        derivative = tuple(
            (fwd - back) / self.step for fwd, back in zip(forward, backward)
        )
        if len(derivative) == 1:
            derivative = derivative[0]
        return derivative

    def fit(self, *args, **kwargs):
        """
        Fit the estimator to
        """
        self.estimator.fit(*args, **kwargs)
        return self


def normalize_direction(direction):
    """
    Transform the direction into a numpy array and normalize it.

    Parameters
    ----------
    direction : list, tuple, or array
        A 1D array representing a vector.

    Returns
    -------
    normalized : array
        The array divided by its norm, making it a unit vector.

    Examples
    --------

    >>> print(normalize_direction((1, 0)))
    [1. 0.]
    >>> print(normalize_direction((1.5, 0)))
    [1. 0.]
    >>> print(normalize_direction((2, 0)))
    [1. 0.]
    >>> print(normalize_direction((0, 2)))
    [0. 1.]
    >>> print(normalize_direction((0, 2, 0)))
    [0. 1. 0.]
    >>> print("{:.3f} {:.3f}".format(*normalize_direction((1, 1))))
    0.707 0.707
    >>> print("{:.3f} {:.3f}".format(*normalize_direction((-2, 2))))
    -0.707 0.707

    """
    # Casting to float is required for the division to work if the original
    # numbers are integers. Since this is always small, it doesn't matter
    # much.
    direction = np.atleast_1d(direction).astype("float64")
    direction /= np.linalg.norm(direction)
    return direction
