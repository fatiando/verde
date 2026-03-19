# Copyright (c) 2017 The Verde Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Gridding with a linear interpolator
===================================

Verde offers the :class:`verde.Linear` class for piecewise linear gridding.
It uses :class:`scipy.interpolate.LinearNDInterpolator` under the hood while
offering the convenience of Verde's gridder API.

The interpolation works on Cartesian data, so if we want to grid geographic
data (like our Baja California bathymetry) we need to project them into a
Cartesian system. We'll use `pyproj <https://github.com/jswhit/pyproj>`__ to
calculate a Mercator projection for the data.

For convenience, Verde still allows us to make geographic grids by passing the
``projection`` argument to :meth:`verde.Linear.grid` and the like. When
doing so, the grid will be generated using geographic coordinates which will be
projected prior to interpolation.
"""
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pyproj

import verde as vd

# We'll test this on the Baja California shipborne bathymetry data
data = vd.datasets.fetch_baja_bathymetry()

# Before gridding, we need to decimate the data to avoid aliasing because of
# the oversampling along the ship tracks. We'll use a blocked median with 1
# arc-minute blocks.
spacing = 1 / 60
reducer = vd.BlockReduce(reduction=np.median, spacing=spacing)
coordinates, bathymetry = reducer.filter(
    (data.longitude, data.latitude), data.bathymetry_m
)

# Project the data using pyproj so that we can use it as input for the gridder.
# We'll set the latitude of true scale to the mean latitude of the data.
projection = pyproj.Proj(proj="merc", lat_ts=data.latitude.mean())
proj_coordinates = projection(*coordinates)

# Now we can set up a gridder for the decimated data
grd = vd.Linear().fit(proj_coordinates, bathymetry)

# Get the grid region in geographic coordinates
region = vd.get_region((data.longitude, data.latitude))
print("Data region:", region)

# The 'grid' method can still make a geographic grid if we pass in a projection
# function that converts lon, lat into the easting, northing coordinates that
# we used in 'fit'. This can be any function that takes lon, lat and returns x,
# y. In our case, it'll be the 'projection' variable that we created above.
# We'll also set the names of the grid dimensions and the name the data
# variable in our grid (the default would be 'scalars', which isn't very
# informative).
grid = grd.grid(
    region=region,
    spacing=spacing,
    projection=projection,
    dims=["latitude", "longitude"],
    data_names="bathymetry_m",
)
print("Generated geographic grid:")
print(grid)

# Cartopy requires setting the coordinate reference system (CRS) of the
# original data through the transform argument. Their docs say to use
# PlateCarree to represent geographic data.
crs = ccrs.PlateCarree()

plt.figure(figsize=(7, 6))
# Make a Mercator map of our gridded bathymetry
ax = plt.axes(projection=ccrs.Mercator())
# Plot the gridded bathymetry
pc = grid.bathymetry_m.plot.pcolormesh(
    ax=ax, transform=crs, vmax=0, zorder=-1, add_colorbar=False
)
plt.colorbar(pc).set_label("meters")
# Plot the locations of the decimated data
ax.plot(*coordinates, ".k", markersize=0.1, transform=crs)
# Use an utility function to setup the tick labels and the land feature
vd.datasets.setup_baja_bathymetry_map(ax)
ax.set_title("Linear gridding of bathymetry")
plt.show()
