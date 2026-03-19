# Copyright (c) 2017 The Verde Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Gridding with a nearest-neighbors interpolator
==============================================

Verde offers the :class:`verde.KNeighbors` class for nearest-neighbor gridding.
The interpolation looks at the data values of the *k* nearest neighbors of a
interpolated point. If *k* is 1, then the data value of the closest neighbor is
assigned to the point. If *k* is greater than 1, the average value of the
closest *k* neighbors is assigned to the point.

The interpolation works on Cartesian data, so if we want to grid geographic
data (like our Baja California bathymetry) we need to project them into a
Cartesian system. We'll use `pyproj <https://github.com/jswhit/pyproj>`__ to
calculate a Mercator projection for the data.

For convenience, Verde still allows us to make geographic grids by passing the
``projection`` argument to :meth:`verde.KNeighbors.grid` and the like. When
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

# Data decimation using verde.BlockReduce is not necessary here since the
# averaging operation is already performed by the k nearest-neighbor
# interpolator.

# Project the data using pyproj so that we can use it as input for the gridder.
# We'll set the latitude of true scale to the mean latitude of the data.
projection = pyproj.Proj(proj="merc", lat_ts=data.latitude.mean())
proj_coordinates = projection(data.longitude, data.latitude)

# Now we can set up a gridder using the 10 nearest neighbors and averaging
# using using a median instead of a mean (the default). The median is better in
# this case since our data are expected to have sharp changes at ridges and
# faults.
grd = vd.KNeighbors(k=10, reduction=np.median)
grd.fit(proj_coordinates, data.bathymetry_m)

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
    spacing=1 / 60,
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
# Plot the locations of the data
ax.plot(data.longitude, data.latitude, ".k", markersize=0.1, transform=crs)
# Use an utility function to setup the tick labels and the land feature
vd.datasets.setup_baja_bathymetry_map(ax)
ax.set_title("Nearest-neighbor gridding of bathymetry")
plt.show()
