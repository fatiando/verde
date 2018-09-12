"""
Vector Data
===========

Some datasets have multiple vector components measured for each location, like the East
and West components of wind speed or GPS velocities. For example, let's look at our
sample GPS velocity data from the U.S. West coast.
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import pyproj
import verde as vd

data = vd.datasets.fetch_california_gps()

# We'll need to project the geographic coordinates to work with our Cartesian
# classes/functions
projection = pyproj.Proj(proj="merc", lat_ts=data.latitude.mean())
proj_coords = projection(data.longitude.values, data.latitude.values)


plt.figure(figsize=(6, 8))
ax = plt.axes(projection=ccrs.Mercator())
crs = ccrs.PlateCarree()
tmp = ax.quiver(
    data.longitude.values,
    data.latitude.values,
    data.velocity_east.values,
    data.velocity_north.values,
    scale=0.3,
    transform=crs,
    width=0.002,
)
ax.quiverkey(tmp, 0.2, 0.15, 0.05, label="0.05 m/yr", coordinates="figure")
ax.set_title("GPS horizontal velocities")
vd.datasets.setup_california_gps_map(ax)
plt.tight_layout()
plt.show()


########################################################################################
# Verde classes and functions are equipped to deal with vector data natively or through
# the use of the :class:`verde.Vector` class. Function and classes that can take vector
# data as input will accept tuples as the ``data`` and ``weights`` arguments. Each
# element of the tuple must be an array with the data values for a component of the
# vector data. As with ``coordinates``, **the order of components must be**
# ``(east_component, north_component, up_component)``.
#
#
#
# Blocked reductions
# ------------------
#
# Show that it doesn't affect BlockReduce.

########################################################################################
# Trends
# ------
#
# How to use Vector.

########################################################################################
# Gridding
# --------
#
# Un-coupled using Components. Then coupled using VectorSpline2D.
