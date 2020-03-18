"""
"""
import numpy as np

from .base import BaseGridder


class Gradient(BaseGridder):
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

    def _normalized_direction(self):
        """
        Transform the direction into a numpy array and normalize it.
        """
        # Casting to float is required for the division to work if the original
        # numbers are integers. Since this is always small, it doesn't matter
        # much.
        direction = np.atleast_1d(self.direction).astype("float64")
        direction /= np.linalg.norm(direction)
        return direction

    def predict(self, coordinates):
        """
        """
        direction = self._normalized_direction()
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
        forward = self.estimator.predict(forward_coords)
        backward = self.estimator.predict(backward_coords)
        derivative = (forward - backward) / self.step
        return derivative

    def fit(self, *args, **kwargs):
        """
        Fit the estimator to
        """
        self.estimator.fit(*args, **kwargs)
        return self
