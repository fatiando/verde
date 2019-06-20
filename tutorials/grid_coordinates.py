"""
.. _grid_coordinates:

Gridding coordinates
====================

Grid coordinates in Verde are used to create points on a regularly spaced grid
that are  then used in the spline method to interpolate between sample data
points. The grid can be specified either by the number of points in each
dimension (the *shape*) or by the grid node spacing.

Creating regular grids in Verde is done using the :func:`verde.grid-coordinates`
function. The function uses :mod:`numpy.linspace` and
:mod:`numpy.meshgrid` to interpolate between points in both the west-east and
south-north directions and creates a two-dimensional spatial grid. To this
end, the function will adjust either the region bounds or the spacing between
grid nodes.

When we choose to adjust the region the function takes the original region, and
divides the south-north and west-east extents of the region by the spacing to
get the number of grid points in both directions. This value is then
rounded and converted to an integer. The function then adds one to the integer
to get the number of nodes rather than segments.The function then multiplies
the number of grid points by the chosen spacing to create the new region.

Let's create a region that is 1555 units west-east and 1250 units south-north
with a grid spacing every 500 units. For these dimensions and spacing we should
expect the west-east dimension to contain 4 grid points (1555/500 is 3.11 which
is rounded to 3 segments and 4 nodes), and the south-north dimensions to
contain 3 grid points (1250/500 is 2.5 which is rounded to 2 segments and 3
nodes. After :func:`verde.grid-coordinates` calculates the number of grid
points it then multiplies the number of grid points by the spacing and adds the
minimum values so that the points are spatially located.
"""

import numpy as np
import matplotlib.pyplot as plt
import verde as vd
from matplotlib.patches import Rectangle

# let's create some bounds for the region and set the spacing for the grid points
spacing = 500
west = 0
east = 1555
south = 0
north = 1250
region = (west, east, south, north)

########################################################################################
# Create the grid with adjust region
# ----------------------------------
#
# With a region and spacing defined, grid_coordinates can now create a regular
# grid. First let's pass a set shape to :func:`verde.grid_coordinates` to
# confirm that it is creating 4 west-east grid points and 3 south-north grid
# points, and set the adjust parameter to `region`

regioneast, regionnorth = vd.grid_coordinates(
    region=region, spacing=spacing, adjust="region"
)
print(regioneast.shape, regionnorth.shape)

########################################################################################
# With the spacing set at 500 units and a 3 by 4 grid of regular dimensions,
# :func:`verde.grid_coordinates` then calculates the spatial location of each
# grid point and adjusts the region so that the maximum northing and maximum
# easting values are divisible by the spacing. In this example, the easting has
# 3 segments (4 nodes) that are each 500 units long, meaning the easting spans
# from 0 to 1500. The northing has 2 segments (3 nodes) that are each 500 units
# long, meaning the northing spans from 0 to 1000. Both dimensions are divisble
# by 500.

print(regioneast)
print(regionnorth)

######################################################################################
# By default if adjust is not assigned to `region` or `spacing`,
# :func:`verde.grid_coordinates` will adjust the spacing. With the adjust
# parameter set to `spacing` :func:`verde.grid_coordinates` creates grid nodes
# in a similar manner as when it adjusts the region. However, it doesn't readjust
# the region so that it is divisble by the spacing before creating the grid.
# This means the grid will have the same number of grid points no matter if
# `region` or `spacing` are set to the adjust parameter.

########################################################################################
# Create the grid with adjust spacing
# -----------------------------------
#
# Now let's adjust the spacing of the grid points. Note that the number of grid
# points is still the same as above.

spacingeast, spacingnorth = vd.grid_coordinates(
    region=region, spacing=500, adjust="spacing"
)
print(spacingeast.shape, spacingnorth.shape)

######################################################################################
# However the regular spacing between the grid points is no longer 500 units.

print(spacingeast)
print(spacingnorth)

######################################################################################
# Visualize the different adjustments
# -----------------------------------
# Let's visualize the difference between the original region bounds, the
# adjusted region grid nodes, and the adjusted spacing grid nodes. Note the
# adjusted spacing grid nodes keep the original region, while the adjusted
# region grid nodes on the north and east side of the region have moved.

plt.figure(figsize=(6,6))
currentAxis = plt.gca()
currentAxis.add_patch(plt.Rectangle((west, south), east, north, fill=None, label='Region Bounds'))
plt.scatter(regioneast, regionnorth, label='Adjusted Region Grid Nodes', marker='>', color='blue')
plt.scatter(spacingeast, spacingnorth, label='Adjusted Spacing Grid Nodes', marker='>', color='orange')
plt.ylim(-50,1600)
plt.xlim(-50,1600)
plt.xlabel('Easting')
plt.ylabel('Northing')
plt.legend(loc='upper center')
plt.show()

######################################################################################
# Pixel Registration
# ------------------
# Pixel registration locates the grid points in the middle of the grid segments
# rather than in the corner of each grid.
# First, let's take the same region and grid, and set the `adjust` parameter to
#``region`` so that the function will adjust the region, and set
# ``pixel_register`` parameter to `true`. Without piexel registration our grid
# had dimensions of 3x4, with pixel registration we expect the dimensions of
# the grid to be the dimensions of the non-registered grid minus one or equal to
# the number of segments between the grid points in the non-registered grid.

regioneastpixel, regionnorthpixel = vd.grid_coordinates(region=region, spacing=spacing, adjust='region', pixel_register=True)
print(regioneastpixel.shape, regionnorthpixel.shape)

######################################################################################
# And we can check the coordinates for the grid points with region adjustment.

print(regioneastpixel)
print(regionnorthpixel)

######################################################################################
# If we set the ``adjust`` parameter to ``spacing`` the function will return
# the same dimensions as when ``adjust`` is set to ``region``.

spacingeastpixel, spacingnorthpixel = vd.grid_coordinates(region=region, spacing=500, adjust='spacing', pixel_register=True)
print(spacingeastpixel.shape, spacingnorthpixel.shape)

######################################################################################
# Again we can check the coordinates for grid points with spacing adjustment.

print(spacingeastpixel)
print(spacingnorthpixel)

######################################################################################
# Lastly, we can plot up the pixel-registered grid points to see where they fall
# within the original region bounds. 

plt.figure(figsize=(6,6))
currentAxis = plt.gca()
currentAxis.add_patch(plt.Rectangle((west, south), east, north, fill=None, label='Region Bounds'))
plt.scatter(regioneastpixel, regionnorthpixel, label='Adjust Region Grid Nodes', marker='o', color='blue')
plt.scatter(spacingeastpixel, spacingnorthpixel, label='Adjust Spacing Grid Nodes', marker='o', color='orange')
plt.ylim(-50,1600)
plt.xlim(-50,1600)
plt.xlabel('Easting')
plt.ylabel('Northing')
plt.legend(loc='upper center')
plt.show()
