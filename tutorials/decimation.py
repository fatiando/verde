"""
Data decimation
===============

Often times, raw spatial data can be highly oversampled in a direction. In these cases,
we need to decimate the data before interpolation to avoid aliasing effects. For
example, our sample shipborne bathymetry data has a higher sampling frequency along the
tracks than between tracks:
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import verde as vd

# Load the data as a pandas.DataFrame
data = vd.datasets.fetch_baja_bathymetry()

# Plot it using matplotlib and Cartopy
crs = ccrs.PlateCarree()
plt.figure(figsize=(7, 7))
ax = plt.axes(projection=ccrs.Mercator())
ax.set_title('Locations of bathymetry measurements from Baja California', pad=25)
# Plot the land as a solid color
ax.add_feature(cfeature.LAND, edgecolor='black')
# Plot the bathymetry data locations as black dots
plt.plot(data.longitude, data.latitude, '.k', markersize=1, transform=crs)
ax.gridlines(draw_labels=True)
plt.tight_layout()
plt.show()

########################################################################################
# Class :class:`verde.BlockReduce` can be used to apply a reduction/aggregation operation (mean,
# median, standard deviation, etc) to the data in regular blocks.
# All data inside each block will be replaced by their aggregated value.
# :class:`~verde.BlockReduce` takes an aggregation function as input.
# It can be any function that receives a numpy array as input and returns a single
# scalar value. The :func:`numpy.mean` or :func:`numpy.median` functions are usually
# what we want.
import numpy as np

########################################################################################
# Blocked means and medians are good ways to decimate data for interpolation. Let's use
# a blocked median on our data to decimate it to our desired grid interval of 5
# arc-minutes. The reason for using a median over a mean is because bathymetry data can
# vary abruptly and a mean would smooth the data too much. For data varies more
# smoothly (like gravity and magnetic data), a mean would be a better option.
reducer = vd.BlockReduce(reduction=np.median, spacing=5/60)
print(reducer)

########################################################################################
# To reduce our data using the blocked median, use the :meth:`~verde.BlockReduce.filter`
# method:
coordinates, bathymetry = reducer.filter(coordinates=(data.longitude, data.latitude),
                                         data=data.bathymetry_m)

plt.figure(figsize=(7, 7))
ax = plt.axes(projection=ccrs.Mercator())
ax.set_title('Locations of decimated data', pad=25)
# Plot the land as a solid color
ax.add_feature(cfeature.LAND, edgecolor='black')
# Plot the bathymetry data locations as black dots
plt.plot(*coordinates, '.k', markersize=1, transform=crs)
ax.gridlines(draw_labels=True)
plt.tight_layout()
plt.show()


########################################################################################
# By default, the coordinates of the decimated data are obtained by applying the same
# reduction operation to the coordinates of the original data. Alternatively, we can
# tell :class:`~verde.BlockReduce` to return the coordinates of centers of each block:
coordinates_center, _ = vd.BlockReduce(reduction=np.median, spacing=5/60,
                                       center_coordinates=True).filter(coordinates=(data.longitude,
                                                                                    data.latitude),
                                                                       data=data.bathymetry_m)

plt.figure(figsize=(7, 7))
ax = plt.axes(projection=ccrs.Mercator())
ax.set_title('Locations of decimated data using block centers', pad=25)
# Plot the land as a solid color
ax.add_feature(cfeature.LAND, edgecolor='black')
# Plot the bathymetry data locations as black dots
plt.plot(*coordinates_center, '.k', markersize=1, transform=crs)
ax.gridlines(draw_labels=True)
plt.tight_layout()
plt.show()

########################################################################################
# Now the data is ready for interpolation.

########################################################################################
# Using weights
# -------------
#
# :class:`~verde.BlockReduce` can't output weights for each data point because it
# doesn't know which reduction operation it's using. If you want to do a weighted
# interpolation, like :class:`verde.Spline`, you can use :class:`verde.BlockMean`
# instead. It can calculate different kinds of weights, depending on configuration
# options and what you give it as input:
#
# * If input weights aren't given, then the output weights will be 1 over the variance
#   of the data in each block.
# * If input weights are given and they are not related to the uncertainty of the data,
#   then the output weights will be 1 over the weighted variance of the data in each
#   block. This is the case when data weights are chosen by the user, not based on the
#   measurement uncertainty. For example, when you need to give less importance to a
#   portion of the data and no uncertainties are available.
# * If input weights are given and they are 1 over the data uncertainty squared, then
#   use option ``uncertainty=True`` to tell :class:`~verde.BlockMean` to calculate
#   weights based on the propagated uncertainty of the data. The output weights will be
#   1 over the propagated uncertainty squared. In this case, the **input weights should
#   not be normalized**. This is preferable if you have the uncertainty of the data.
#
# .. note::
#
#    Output weights are always normalized to the ]0, 1] range. See
#    :func:`verde.variance_to_weights`.
#
# Since our bathymetry data doesn't come with uncertainties, we can see what the output
# weights would look like for options 1.
coordinates, _, weights_variance = vd.BlockMean(spacing=5/60).filter((data.longitude,
                                                            data.latitude),
                                                                     data.bathymetry_m)

# The weights vary a lot so it's better to plot them using a logarithmic color scale
from matplotlib.colors import LogNorm

plt.figure(figsize=(8, 7))
ax = plt.axes(projection=ccrs.Mercator())
ax.set_title('Decimated data weights based on the variance', pad=25)
# Plot the land as a solid color
ax.add_feature(cfeature.LAND, edgecolor='black')
# Plot the output weights as colored circles
plt.scatter(*coordinates, c=weights_variance, s=1, cmap='magma', transform=crs,
            norm=LogNorm())
plt.colorbar(aspect=50)
ax.gridlines(draw_labels=True)
plt.tight_layout()
plt.show()
