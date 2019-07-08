"""
.. _grid_coordinates:

Gridding coordinates
====================

Grid coordinates in Verde are used to create points on a regularly spaced grid
that are then used in the spline method to interpolate between sample data
points. The grid can be specified either by the number of points in each
dimension (the *shape*) or by the grid node spacing.

Creating regular grids in Verde is done using the :func:`verde.grid_coordinates`
function. The function uses :func:`numpy.linspace` and
:func:`numpy.meshgrid` to create a set of regulary spaced points in both the
west-east and south-north directions to create a two-dimensional spatial grid.
To this end, the function will adjust either the region bounds or the spacing
between grid nodes.

First let's create a region that is 1000 units west-east and 1000 units south-
north, and we will set an initial spacing to 100 units. This should create a
square grid with no adjustment of the region, or the spacing.
"""

import numpy as np
import matplotlib.pyplot as plt
import verde as vd
from matplotlib.patches import Rectangle

# create the bounds for the region and set the spacing for the grid points
spacing = 100
west, east, south, north = 0, 1000, 0, 1000
region = (west, east, south, north)

# create the grid
square_region_east, square_region_north = vd.grid_coordinates(
    region=region, spacing=spacing
)

########################################################################################
# We can check the dimensions of the grid to confirm that it is 11x11

print(square_region_east.shape, square_region_north.shape)

########################################################################################
# Now let's plot the grid to visualize the location of the grid points

plt.figure(figsize=(6, 6))
currentAxis = plt.gca()
currentAxis.add_patch(
    plt.Rectangle((west, south), east, north, fill=None, label="Region Bounds")
)

plt.vlines(
    square_region_east[0],
    ymin=min(square_region_north[0]),
    ymax=max(square_region_north[-1]),
    linestyles="dotted",
    zorder=0,
)
plt.hlines(
    square_region_north[:, 1],
    xmin=min(square_region_east[0]),
    xmax=max(square_region_east[-1]),
    linestyles="dotted",
    zorder=0,
)

plt.scatter(
    square_region_east,
    square_region_north,
    label="Square Region Grid Nodes",
    marker=".",
    color="black",
    s=100,
)

plt.ylim(-50, 1050)
plt.xlim(-50, 1050)
plt.xlabel("Easting")
plt.ylabel("Northing")
plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15))
plt.show()

########################################################################################
# How Grid Coordinates works
# --------------------------
# Now let's create a region that is 1555 units west-east, and 1250 units south-north
# with a spacing of 500 units. Because the range of the west-east and south-north
# boundaries are not multiples of 500, we must choose to change either:
#   - the boundaries of the region in order to fit the spacing, or
#   - the spacing in order to fit the region boundaries.

spacing = 500
west, east, south, north = 0, 1555, 0, 1250
region = (west, east, south, north)

########################################################################################
# Create the grid with adjust region
# ----------------------------------
# With a region and spacing defined, :func:`verde.grid_coordinates` can now create a regular
# grid. First let's pass our data to :func:`verde.grid_coordinates` to
# confirm that it is creating 4 west-east grid points and 3 south-north grid
# points, and set the adjust parameter to ``region``

region_east, region_north = vd.grid_coordinates(
    region=region, spacing=spacing, adjust="region"
)
print(region_east.shape, region_north.shape)

########################################################################################
# With the spacing set at 500 units and a 3 by 4 grid of regular dimensions,
# :func:`verde.grid_coordinates` then calculates the spatial location of each
# grid point and adjusts the region so that the maximum northing and maximum
# easting values are divisible by the spacing. In this example, the easting has
# 3 segments (4 nodes) that are each 500 units long, meaning the easting spans
# from 0 to 1500. The northing has 2 segments (3 nodes) that are each 500 units
# long, meaning the northing spans from 0 to 1000. Both dimensions are divisible
# by 500.

print(region_east)
print(region_north)

######################################################################################
# By default if adjust is not assigned to ``region`` or ``spacing``,
# :func:`verde.grid_coordinates` will adjust the spacing. With the adjust
# parameter set to ``spacing`` :func:`verde.grid_coordinates` creates grid nodes
# in a similar manner as when it adjusts the region. However, it doesn't readjust
# the region so that it is divisble by the spacing before creating the grid.
# This means the grid will have the same number of grid points no matter if
# the adjust parameter is set to ``region`` or ``spacing``.

########################################################################################
# Create the grid with adjust spacing
# -----------------------------------
#
# Now let's adjust the spacing of the grid points. Note that the number of grid
# points is still the same as above.

spacing_east, spacing_north = vd.grid_coordinates(
    region=region, spacing=500, adjust="spacing"
)
print(spacing_east.shape, spacing_north.shape)

######################################################################################
# However the regular spacing between the grid points is no longer 500 units.

print(spacing_east)
print(spacing_north)

######################################################################################
# Visualize the different adjustments
# -----------------------------------
# Let's visualize the difference between the original region bounds, the
# adjusted region grid nodes, and the adjusted spacing grid nodes. Note the
# adjusted spacing grid nodes keep the original region, while the adjusted
# region grid nodes on the north and east side of the region have moved.

plt.figure(figsize=(6, 6))
currentAxis = plt.gca()
currentAxis.add_patch(
    plt.Rectangle((west, south), east, north, fill=None, lw=2.0, label="Region Bounds")
)

plt.vlines(
    region_east[0],
    ymin=min(region_north[0]),
    ymax=max(region_north[-1]),
    linestyles="dotted",
    zorder=0,
)
plt.hlines(
    region_north[:, 1],
    xmin=min(region_east[0]),
    xmax=max(region_east[-1]),
    linestyles="dotted",
    zorder=0,
)

plt.scatter(
    region_east,
    region_north,
    label="Adjusted Region Grid Nodes",
    marker=">",
    color="blue",
    alpha=0.75,
    s=100,
)

plt.vlines(
    spacing_east[0],
    ymin=min(spacing_north[0]),
    ymax=max(spacing_north[-1]),
    linestyles="dashed",
    zorder=0,
)
plt.hlines(
    spacing_north[:, 1],
    xmin=min(spacing_east[0]),
    xmax=max(spacing_east[-1]),
    linestyles="dashed",
    zorder=0,
)
plt.scatter(
    spacing_east,
    spacing_north,
    label="Adjusted Spacing Grid Nodes",
    marker=">",
    color="orange",
    alpha=0.75,
    s=100,
)

plt.ylim(-50, 1300)
plt.xlim(-50, 1600)
plt.xlabel("Easting")
plt.ylabel("Northing")
plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18))
plt.show()

######################################################################################
# Pixel Registration
# ------------------
# Pixel registration locates the grid points in the middle of the grid segments
# rather than in the corner of each grid.
# First, let's take the 1000x1000 region with 100 unit spacing from the first example
# and set the ``pixel_register`` parameter to `True`. Without pixel registration our
# grid should have dimensions of 11x11. With pixel registration we expect the dimensions
# of the grid to be the dimensions of the non-registered grid minus one, or equal to
# the number of segments between the grid points in the non-registered grid (10x10).

spacing = 100
west, east, south, north = 0, 1000, 0, 1000
region = (west, east, south, north)

pixel_east, pixel_north = vd.grid_coordinates(
    region=region, spacing=spacing, pixel_register=True
)
print(pixel_east.shape, pixel_north.shape)

######################################################################################
# And we can check the coordinates for the grid points with pixel registration.

print(pixel_east)
print(pixel_north)

######################################################################################
# If we set ``pixel_register`` to ``False`` the function will return the grid
# with one more grid node in both west-east and south-north directions.

no_pixel_east, no_pixel_north = vd.grid_coordinates(
    region=region, spacing=spacing, pixel_register=False
)
print(no_pixel_east.shape, no_pixel_north.shape)

######################################################################################
# Again we can check the coordinates for grid points with spacing adjustment.

print(no_pixel_east)
print(no_pixel_north)

######################################################################################
# Lastly, we can plot up the pixel-registered grid points to see where they fall
# within the original region bounds.

plt.figure(figsize=(6, 6))
currentAxis = plt.gca()
currentAxis.add_patch(
    plt.Rectangle((west, south), east, north, fill=None, lw=2.0, label="Region Bounds")
)
plt.vlines(
    pixel_east[0],
    ymin=min(pixel_north[0]),
    ymax=max(pixel_north[-1]),
    linestyles="dotted",
    zorder=0,
)
plt.hlines(
    pixel_north[:, 1],
    xmin=min(pixel_east[0]),
    xmax=max(pixel_east[-1]),
    linestyles="dotted",
    zorder=0,
)

plt.scatter(
    pixel_east,
    pixel_north,
    label="Pixel Registered Grid Nodes",
    marker="o",
    color="blue",
    alpha=0.75,
    s=100,
)

plt.vlines(
    no_pixel_east[0],
    ymin=min(no_pixel_north[0]),
    ymax=max(no_pixel_north[-1]),
    linestyles="dashed",
    zorder=0,
)
plt.hlines(
    no_pixel_north[:, 1],
    xmin=min(no_pixel_east[0]),
    xmax=max(no_pixel_east[-1]),
    linestyles="dashed",
    zorder=0,
)
plt.scatter(
    no_pixel_east,
    no_pixel_north,
    label="Regular Registered Grid Nodes",
    marker="o",
    color="orange",
    alpha=0.75,
    s=100,
)

plt.ylim(-50, 1050)
plt.xlim(-50, 1050)
plt.xlabel("Easting")
plt.ylabel("Northing")
plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18))
plt.show()
