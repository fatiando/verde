"""
GPS velocities from California
==============================

We provide sample 3-component GPS velocity data from the West coast of the U.S.
The data were cut from EarthScope Plate Boundary Observatory data provided by
UNAVCO. The velocities are in the North American tectonic plate reference
system (NAM08). The velocities and their associated standard deviations are in
meters/year.
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
# We need these two classes to set proper ticklabels for Cartopy maps
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import numpy as np
import verde as vd


# The data are in a pandas.DataFrame
data = vd.datasets.fetch_california_gps()
print(data.head())


def setup_map(ax):
    "Set the proper ticks for a Cartopy map and draw land and water"
    ax.set_xticks(np.arange(-124, -115, 4), crs=crs)
    ax.set_yticks(np.arange(33, 42, 2), crs=crs)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_extent(vd.get_region((data.longitude, data.latitude)), crs=crs)
    # Plot the land and ocean as a solid color
    ax.add_feature(cfeature.LAND, facecolor='gray')
    ax.add_feature(cfeature.OCEAN)


# Make a plot of the data using Cartopy to handle projections and coastlines
crs = ccrs.PlateCarree()
fig, axes = plt.subplots(1, 2, figsize=(8, 4),
                         subplot_kw=dict(projection=ccrs.Mercator()))
# Plot the horizontal velocity vectors
ax = axes[0]
ax.set_title('GPS horizontal velocities')
setup_map(ax)
ax.quiver(data.longitude.values, data.latitude.values,
          data.velocity_east.values, data.velocity_north.values,
          scale=0.3, transform=crs)
# Plot the vertical velocity
ax = axes[1]
ax.set_title('Vertical velocity')
setup_map(ax)
maxabs = np.abs([data.velocity_up.min(), data.velocity_up.max()]).max()
tmp = ax.scatter(data.longitude, data.latitude, c=data.velocity_up,
                 s=10, vmin=-maxabs/3, vmax=maxabs/3, cmap='seismic',
                 transform=crs)
cb = plt.colorbar(tmp, ax=ax)
cb.set_label('meters/year')
plt.tight_layout(w_pad=0)
plt.show()
