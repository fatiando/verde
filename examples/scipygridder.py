"""
Gridding with Scipy
===================

Scipy offers a range of interpolation methods in :mod:`scipy.interpolate` and 3
specifically for 2D data (linear, nearest neighbors, and bicubic). Verde offers
an interface for these 3 scipy interpolators in :class:`verde.ScipyGridder`.
"""
import matplotlib.pyplot as plt
import cmocean
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import verde as vd

# We'll test this on the Baja California shipborne bathymetry data
data = vd.datasets.fetch_baja_bathymetry()

# Before gridding, we need to decimate the data to avoid aliasing because of
# the oversampling along the ship tracks. We'll use a blocked median with 10
# arc-minute blocks.
spacing = 10/60
lon, lat, bathymetry = vd.block_reduce(
    data.longitude, data.latitude, data.bathymetry_m,
    reduction=np.median, spacing=spacing)

# Now we can grid the decimated data to a 10' grid.
grd = vd.ScipyGridder().fit(lon, lat, bathymetry)
grid = grd.grid(spacing=spacing)

proj = ccrs.Mercator()
crs = ccrs.PlateCarree()

plt.figure(figsize=(7, 6))
ax = plt.axes(projection=proj)
ax.set_title("10' Gridded Bathymetry", pad=25)
# Plot the bathymetry as colored circles.
# Cartopy requires setting the projection of the original data through the
# transform argument. Use PlateCarree for geographic data.
grid.scalars.plot.pcolormesh(ax=ax, transform=crs, cmap='viridis',
                             vmin=grid.scalars.min(), vmax=0)
plt.plot(lon, lat, '.k', transform=crs)
# Plot the land as a solid color
ax.add_feature(cfeature.LAND, edgecolor='black', zorder=1000)
ax.gridlines(draw_labels=True)
ax.set_extent(grd.region_, crs=crs)
plt.tight_layout()
plt.show()
