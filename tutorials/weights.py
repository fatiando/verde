"""
Using weights
=============

One of the advantages of using a Green's functions approach to interpolation is that we
can easily weight the data to give each point more or less influence over the results.
This is a good way to not less data points with large uncertainties bias the
interpolation or the data decimation.

We'll use some sample GPS vertical ground velocity which has some variable uncertainties
associated with each data point.
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
# We need these two classes to set proper ticklabels for Cartopy maps
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import numpy as np
import verde as vd

# Load the data as a pandas.DataFrame
data = vd.datasets.fetch_california_gps()
print(data.head())

# TODO: plot the data and uncertainty

# Plot it using matplotlib and Cartopy
crs = ccrs.PlateCarree()

########################################################################################
# Weights in data decimation
# --------------------------
#
# :class:`~verde.BlockReduce` can't output weights for each data point because it
# doesn't know which reduction operation it's using. If you want to do a weighted
# interpolation, like :class:`verde.Spline`, you can use :class:`verde.BlockMean`
# instead to produce weights of the decimated data. It can calculate different kinds of
# weights, depending on configuration options and what you give it as input:
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
coordinates, _, weights_variance = vd.BlockMean(spacing=15/60).filter((data.longitude,
                                                            data.latitude),
                                                                     data.velocity_up)
_, _, weights_wvariance = vd.BlockMean(spacing=15/60).filter((data.longitude,
                                                            data.latitude),
                                                                     data.velocity_up,
                                                             weights=1/data.std_up**2)
_, _, weights_uncertainty = vd.BlockMean(spacing=15/60,uncertainty=True).filter((data.longitude,
                                                            data.latitude),
                                                                     data.velocity_up,
                                                             weights=1/data.std_up**2)


# The weights vary a lot so it's better to plot them using a logarithmic color scale
from matplotlib.colors import LogNorm

def setup_map(ax, title):
    "Draw coastlines, ticks, land, ocean, and set the title."
    ax.set_title(title)
    # Plot the land as a solid color
    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='gray')
    ax.add_feature(cfeature.OCEAN)
    ax.set_extent(vd.get_region(coordinates), crs=crs)
    # Set the proper ticks for a Cartopy map
    ax.set_xticks(np.arange(-124, -115, 4), crs=crs)
    ax.set_yticks(np.arange(33, 42, 2), crs=crs)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

plt.figure(figsize=(8, 7))
ax = plt.axes(projection=ccrs.Mercator())
# Plot the land as a solid color
ax.add_feature(cfeature.LAND, edgecolor='black')
# Plot the output weights as colored circles
plt.scatter(*coordinates, c=weights_variance, s=50, cmap='magma', transform=crs,
            norm=LogNorm())
plt.colorbar(aspect=50)
setup_map(ax, 'Decimated data weights based on the variance')
plt.tight_layout()

plt.figure(figsize=(8, 7))
ax = plt.axes(projection=ccrs.Mercator())
# Plot the land as a solid color
ax.add_feature(cfeature.LAND, edgecolor='black')
# Plot the output weights as colored circles
plt.scatter(*coordinates, c=weights_wvariance, s=50, cmap='magma', transform=crs,
            norm=LogNorm())
plt.colorbar(aspect=50)
setup_map(ax, 'Decimated data weights based on the weighted variance')
plt.tight_layout()

plt.figure(figsize=(8, 7))
ax = plt.axes(projection=ccrs.Mercator())
# Plot the land as a solid color
ax.add_feature(cfeature.LAND, edgecolor='black')
# Plot the output weights as colored circles
plt.scatter(*coordinates, c=weights_uncertainty, s=50, cmap='magma', transform=crs,
            norm=LogNorm())
plt.colorbar(aspect=50)
setup_map(ax, 'Decimated data weights based on the weighted variance')
plt.tight_layout()
plt.show()
