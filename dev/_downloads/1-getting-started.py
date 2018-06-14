"""
Getting Started
===============

This tutorial will help you get started with using Verde to grid some data.

.. attention::

    This is a work in progress. We're currently planning and prototyping what
    will be included in the tutorial.

Outline:

* The library and where to find functions
* Loading tests datasets (project it using pyproj)
* Preparing data for gridding (block median)
* The gridder classes, fitting, and their attributes
* Predicting data using ``predict`` (evaluate the misfit)
* Gridding (use Cartesian coordinates)
* Using ``projection`` to make a geographic grid
"""
###############################################################################
# First this we need to do is import the ``verde`` library. All of the main
# functions and classes are available through this single ``import`` (the verde
# *namespace*). We'll also need a few other libraries for plotting and
# projecting our data.

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pyproj
import numpy as np
import verde as vd

###############################################################################
# The first thing to do is load a test data set with which we can work. Verde
# offers functions for loading our packaged test data in :mod:`verde.datasets`.
# In this tutorial, we'll work with some bathymetry data from Baja California.

data = vd.datasets.fetch_baja_bathymetry()

###############################################################################
# The data are stored in a pandas.DataFrame object.

print("Data is of type:", type(data))
print(data.head())

###############################################################################
# Plot the data using matplotlib and Cartopy

crs = ccrs.PlateCarree()

plt.figure(figsize=(7, 6))
ax = plt.axes(projection=ccrs.Mercator())
ax.set_title('Bathymetry data from Baja California', pad=25)
# Plot the land as a solid color
ax.add_feature(cfeature.LAND, edgecolor='black')
# Plot the bathymetry as colored circles.
plt.scatter(data.longitude, data.latitude, c=data.bathymetry_m, s=0.1,
            transform=crs)
cb = plt.colorbar(pad=0.08)
cb.set_label('meters')
ax.gridlines(draw_labels=True)
plt.tight_layout()
plt.show()
