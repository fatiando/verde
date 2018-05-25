"""
Block mean uncertainty and variance
===================================

:class:`verde.BlockReduce` is not able to output weights because we need to
make assumptions about the reduction operation to know how to propagate
uncertainties or calculated weighted variances.
That's why verde provides specialized reductions like :class:`verde.BlockMean`,
which can calculate weights from input data in two ways:

1. Propagating the uncertainties in the data
2. Calculating a weighted variance of the data

The uncertainties are more adequate if your data is smooth in each block (low
variance) but have very different uncertainties. The weighted variance should
be used when the data vary a lot in each block (high variance) but have very
similar uncertainties.
"""
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
# We need these two classes to set proper ticklabels for Cartopy maps
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import pyproj
import numpy as np
import verde as vd

# We'll test this on the California vertical GPS velocity data because it comes
# with the uncertainties
data = vd.datasets.fetch_california_gps()
coordinates = (data.longitude, data.latitude)

spacing = 30/60
weights = 1/data.std_up**2
_, _, weights_var = (vd.BlockMean(spacing, center_coordinates=True)
                     .filter(coordinates, data.velocity_up, weights))
block_coords, velocity, weights_uncert = (
    vd.BlockMean(spacing, center_coordinates=True, uncertainty=True)
    .filter(coordinates, data.velocity_up, weights))

# Now we can plot the different weights side by side on Mercator maps
fig, axes = plt.subplots(1, 2, figsize=(9, 7),
                         subplot_kw=dict(projection=ccrs.Mercator()))
titles = ['Variance weights', 'Uncertainty weights']
crs = ccrs.PlateCarree()
for ax, title, w in zip(axes, titles, (weights_var, weights_uncert)):
    ax.set_title(title)
    # Plot the weights
    pc = ax.scatter(*coordinates, c=w, s=70, transform=crs,
                    norm=PowerNorm(gamma=1/2))
    plt.colorbar(pc, ax=ax, orientation='horizontal', pad=0.05)
    # Plot the land as a solid color
    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='gray')
    ax.add_feature(cfeature.OCEAN)
    ax.set_extent(vd.get_region(coordinates), crs=crs)
    # Set the proper ticks for a Cartopy map
    ax.set_xticks(np.arange(-124, -115, 4), crs=crs)
    ax.set_yticks(np.arange(33, 42, 2), crs=crs)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
ax.legend(loc='lower left')
plt.tight_layout()
plt.show()
