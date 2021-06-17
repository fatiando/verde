# Copyright (c) 2017 The Verde Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Gridding on a preexisting grid
==============================

The ``grid`` method of any Verde gridder allows us to interpolate the data on
a regular grid. By passing the ``shape`` or ``spacing`` arguments, the method
will build the regular grid by itself.
Nevertheless, if we want to interpolate the data on a predefined regular grid,
we can pass the grid ``coordinates`` instead.
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pyproj
import numpy as np
import verde as vd

# Define a CheckerBoard gridder
region = (0, 5000, -5000, 0)
gridder = vd.datasets.CheckerBoard(region=region)

# Define a set of grid coordinates as 1d arrays
easting = np.linspace(*region[:2], 50)
northing = np.linspace(*region[2:], 50)
coordinates = (easting, northing)

# Use the .grid() method to predict on the grid coordinates
grid = gridder.grid(coordinates)

# Plot the grid
grid.scalars.plot()
plt.gca().set_aspect("equal")
plt.show()
