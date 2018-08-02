"""
Overview
========

Verde provides classes and functions for processing spatial data, like bathymetry, GPS,
temperature, gravity, or anything else that is measured along the surface of the Earth.
The main focus is on methods for gridding such data (interpolating on a regular grid).
You'll also find other analysis methods that are often used in combination with
gridding, like trend removal and blocked operations.
"""

########################################################################################
# The library
# -----------
#
# Most classes and functions are available through the :mod:`verde` top level package.
# The only exceptions are the functions related to loading sample data, which are in
# :mod:`verde.datasets`. Throughout the documentation we'll use ``vd`` as the alias for
# :mod:`verde`.
import verde as vd

########################################################################################
# The gridder interface
# ---------------------
#

grd = vd.Spline()
print(grd)

# Cover fit, predict, score, grid, and profile on synthetic data. All cartesian and no

########################################################################################
# Conventions
# -----------
#
# Our naming conventions: region, coordinates, order of components, why no x and y.

########################################################################################
# Green's functions
# -----------------
#
# A bit about the math behind this approach. Explain the jacobian methods of gridders.
