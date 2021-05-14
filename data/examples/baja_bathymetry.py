# Copyright (c) 2017 The Verde Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Bathymetry data from Baja California
====================================

We provide sample bathymetry data from Baja California to test the gridding
methods. This is the ``@tut_ship.xyz`` sample data provided by `GMT
<https://www.generic-mapping-tools.org/>`__ for their tutorials and gallery.
The data is downloaded to a local directory if it's not there already.
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import verde as vd


# The data are in a pandas.DataFrame
data = vd.datasets.fetch_baja_bathymetry()
print(data.head())

# Make a Mercator map of the data using Cartopy
plt.figure(figsize=(7, 6))
ax = plt.axes(projection=ccrs.Mercator())
ax.set_title("Bathymetry from Baja California")
# Plot the bathymetry as colored circles. Cartopy requires setting the projection of the
# original data through the transform argument. Use PlateCarree for geographic data.
plt.scatter(
    data.longitude,
    data.latitude,
    c=data.bathymetry_m,
    s=0.1,
    transform=ccrs.PlateCarree(),
)
plt.colorbar().set_label("meters")
# Use an utility function to add tick labels and land and ocean features to the map.
vd.datasets.setup_baja_bathymetry_map(ax)
plt.show()
