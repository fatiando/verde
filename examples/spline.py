"""
Dealing with outliers using splines
===================================

Biharmonic spline interpolation is based on estimating vertical forces acting
on an elastic sheet that yield deformations in the sheet equal to the observed
data.
The results are equivalent to using ``verde.ScipyGridder(method='cubic')`` but
the interpolation is usually slower.
The advantage of using :class:`verde.Spline` is that you can assign weights to
the data to deal with large uncertainties or outliers.
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
# We need these two classes to set proper ticklabels for Cartopy maps
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import pyproj
import numpy as np
import verde as vd

# We'll test this on the Baja California shipborne bathymetry data
data = vd.datasets.fetch_baja_bathymetry()
region = vd.get_region(data.longitude, data.latitude)

# Before gridding, we need to decimate the data to avoid aliasing.
spacing = 10/60
lon, lat, bathymetry = vd.block_reduce(data.longitude, data.latitude,
                                       data.bathymetry_m, reduction=np.median,
                                       spacing=spacing)

# Now we can add some outliers to test our spline
outliers = np.random.RandomState(2).randint(0, bathymetry.size, size=20)
bathymetry[outliers] += 5000
print("Index of outliers:", outliers)

# Project the data using pyproj so that we can use it as input for the gridder.
# We'll set the latitude of true scale to the mean latitude of the data.
projection = pyproj.Proj(proj='merc', lat_ts=data.latitude.mean())
coordinates = projection(lon, lat)

# Create an array of weights and set the weights for the outliers to a very low
# value
weights = np.ones_like(bathymetry)
weights[outliers] = 1e-5

# Now we can fit two splines to our data, one with weights and one without.
# The default spline will fit the data exactly.
spline = vd.Spline().fit(coordinates, bathymetry)
# The weights only work if we use an approximate least-squares solution.
# This means that the spline won't be exact on the data points.
# The easiest way of doing this is to apply some damping regularization to
# smooth the solution.
spline_weights = vd.Spline(damping=1e-8).fit(coordinates, bathymetry, weights)

# We'll make two geographic grids, one for each spline, to compare the results
grids = [sp.grid(region=region, spacing=spacing, projection=projection,
                 dims=['latitude', 'longitude'], data_names=['bathymetry'])
         for sp in [spline, spline_weights]]

# Now we can plot the two grids side by side on Mercator maps
fig, axes = plt.subplots(1, 2, figsize=(9, 6),
                         subplot_kw=dict(projection=ccrs.Mercator()))
titles = ['No Weights', 'Weights']
crs = ccrs.PlateCarree()
for ax, title, grid in zip(axes, titles, grids):
    ax.set_title(title)
    pc = ax.pcolormesh(grid.longitude, grid.latitude, grid.bathymetry,
                       transform=crs, vmax=0, vmin=data.bathymetry_m.min())
    cb = plt.colorbar(pc, ax=ax, orientation='horizontal', pad=0.05)
    cb.set_label('bathymetry [m]')
    # Plot the land as a solid color
    ax.add_feature(cfeature.LAND, edgecolor='black', zorder=2)
    # Plot the locations of the decimated data
    ax.plot(lon, lat, '.k', markersize=0.5, transform=crs)
    # Plot the locations of the outliers
    ax.plot(lon[outliers], lat[outliers], 'xk', transform=crs,
            label='Outliers')
    ax.set_extent(region, crs=crs)
    # Set the proper ticks for a Cartopy map
    ax.set_xticks(np.arange(-114, -105, 2), crs=crs)
    ax.set_yticks(np.arange(21, 30, 2), crs=crs)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
ax.legend(loc='upper right')
plt.tight_layout()
plt.show()
