# Copyright (c) 2017 The Verde Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Gridding with on a preexisting grid
===================================

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

# We'll test this on the air temperature data from Texas
data = vd.datasets.fetch_texas_wind()
coordinates = (data.longitude.values, data.latitude.values)
region = vd.get_region(coordinates)

# Use a Mercator projection for our Cartesian gridder
projection = pyproj.Proj(proj="merc", lat_ts=data.latitude.mean())

# Let's assume we already have a predefined regular grid with a resolution of
# 30 arc-minutes
spacing = 30 / 60
grid_coordinates = vd.grid_coordinates(spacing=spacing, region=region)

# Now we can chain a blocked mean and spline together. The Spline can be regularized
# by setting the damping coefficient (should be positive). It's also a good idea to set
# the minimum distance to the average data spacing to avoid singularities in the spline.
chain = vd.Chain(
    [
        ("mean", vd.BlockReduce(np.mean, spacing=spacing * 111e3)),
        ("spline", vd.Spline(damping=1e-10, mindist=100e3)),
    ]
)
chain.fit(projection(*coordinates), data.air_temperature_c)
print(chain)

# Now we can create a geographic grid of air temperature by providing a projection
# function to the grid method and mask points that are too far from the observations
# Now we can use the chain predictor to interpolate on the preexisting grid
grid_full = chain.grid(
    coordinates=grid_coordinates,
    projection=projection,
    dims=["latitude", "longitude"],
    data_names="temperature",
)
grid = vd.distance_mask(
    coordinates, maxdist=3 * spacing * 111e3, grid=grid_full, projection=projection
)
print(grid)

# Plot the grid and the original data points
plt.figure(figsize=(8, 6))
ax = plt.axes(projection=ccrs.Mercator())
ax.set_title("Air temperature gridded with biharmonic spline")
ax.plot(*coordinates, ".k", markersize=1, transform=ccrs.PlateCarree())
tmp = grid.temperature.plot.pcolormesh(
    ax=ax, cmap="plasma", transform=ccrs.PlateCarree(), add_colorbar=False
)
plt.colorbar(tmp).set_label("Air temperature (C)")
# Use an utility function to add tick labels and land and ocean features to the map.
vd.datasets.setup_texas_wind_map(ax, region=region)
plt.show()
