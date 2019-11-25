# pylint: disable=abstract-method
"""
Generators of synthetic datasets.
"""
import numpy as np

from ..base import BaseGridder
from ..coordinates import check_region


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

    Parameters
    ----------
    amplitude : float
        The amplitude of the checkerboard undulations.
    region : tuple
        The boundaries (``[W, E, S, N]``) of the region used to generate the
        synthetic data.
    w_east : float
        The wavelength in the east direction. Defaults to half of the West-East
        size of the evaluating region.
    w_north : float
        The wavelength in the north direction. Defaults to half of the
        South-North size of the evaluating region.

    Examples
    --------

    >>> synth = CheckerBoard()
    >>> # Default values for the wavelengths are selected automatically
    >>> print(synth.w_east_, synth.w_north_)
    2500.0 2500.0
    >>> # CheckerBoard.grid produces an xarray.Dataset with data on a grid
    >>> grid = synth.grid(shape=(11, 6))
    >>> # scatter and profile generate pandas.DataFrame objects
    >>> table = synth.scatter()
    >>> print(sorted(table.columns))
    ['easting', 'northing', 'scalars']
    >>> profile = synth.profile(point1=(0, 0), point2=(2500, -2500), size=100)
    >>> print(sorted(profile.columns))
    ['distance', 'easting', 'northing', 'scalars']

    """

    def __init__(
        self, amplitude=1000, region=(0, 5000, -5000, 0), w_east=None, w_north=None
    ):
        super().__init__()
        self.amplitude = amplitude
        self.w_east = w_east
        self.w_north = w_north
        self.region = region

    @property
    def w_east_(self):
        "Use half of the E-W extent"
        if self.w_east is None:
            return (self.region[1] - self.region[0]) / 2
        return self.w_east

    @property
    def w_north_(self):
        "Use half of the N-S extent"
        if self.w_north is None:
            return (self.region[3] - self.region[2]) / 2
        return self.w_north

    @property
    def region_(self):
        "Used to fool the BaseGridder methods"
        check_region(self.region)
        return self.region

    def predict(self, coordinates):
        """
        Evaluate the checkerboard function on a given set of points.

        Parameters
        ----------
        coordinates : tuple of arrays
            Arrays with the coordinates of each data point. Should be in the
            following order: (easting, northing, vertical, ...). Only easting
            and northing will be used, all subsequent coordinates will be
            ignored.

        Returns
        -------
        data : array
            The evaluated checkerboard function.

        """
        easting, northing = coordinates[:2]
        data = (
            self.amplitude
            * np.sin((2 * np.pi / self.w_east_) * easting)
            * np.cos((2 * np.pi / self.w_north_) * northing)
        )
        return data
