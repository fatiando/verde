"""
Generators of synthetic datasets.
"""
import numpy as np
from sklearn.utils.validation import check_is_fitted

from ..base import BaseGridder
from ..utils import check_region


class CheckerBoard(BaseGridder):
    r"""
    Generate synthetic data in a checkerboard pattern.

    The mathematical model is:

    .. math::

        f(e, n) = a
        \sin\left(\frac{2\pi}{w_e} e\right)
        \cos\left(\frac{2\pi}{w_n} n\right)

    in which :math:`e` is the easting coordinate, :math:`n` is the northing
    coordinate, :math:`a` is the amplitude, and :math:`w_e` and :math:`w_n` are
    the wavelengths in the east and north directions, respectively.

    The model will evaluated on random points or on a regular grid depending on
    the value of *scatter*.

    Parameters
    ----------
    amplitude : float
        The amplitude of the checkerboard undulations.
    w_east : float
        The wavelength in the east direction. Defaults to half of the West-East
        size of the evaluating region.
    w_north : float
        The wavelength in the north direction. Defaults to half of the
        South-North size of the evaluating region.

    Examples
    --------

    """

    def __init__(self, amplitude=1000, w_east=None, w_north=None):
        self.amplitude = amplitude
        self.w_east = w_east
        self.w_north = w_north

    def fit(self, region=(0, 5000, 0, 5000)):
        """
        Set the region in which the checkerboard will be evaluated.

        If the wavelengths are not supplied, will set them to half of the
        region extent in each dimension.

        Parameters
        ----------
        region : list
            The boundaries (``[W, E, S, N]``) of a given region in Cartesian or
            geographic coordinates.

        """
        check_region(region)
        w, e, s, n = region
        if self.w_east is None:
            self.w_east = (e - w)/2
        if self.w_north is None:
            self.w_north = (n - s)/2
        self.region_ = region
        return self

    def predict(self, easting, northing):
        """
        Evaluate the checkerboard function on a given set of points.

        Parameters
        ----------
        easting : array
            The values of the West-East coordinates.
        northing : array
            The values of the South-North coordinates.

        Returns
        -------
        data : array
            The evaluate checkerboard function.

        """
        check_is_fitted(self, ['region_'])
        data = (self.amplitude *
                np.sin((2*np.pi/self.w_east)*easting) *
                np.cos((2*np.pi/self.w_north)*northing))
        return data
