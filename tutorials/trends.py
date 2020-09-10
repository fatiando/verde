"""
Trend Estimation
================

Trend estimation and removal is a common operation, particularly when dealing
with geophysical data. Moreover, some of the interpolation methods, like
:class:`verde.Spline`, can struggle with long-wavelength trends in the data.
The :class:`verde.Trend` class fits a 2D polynomial trend of arbitrary degree
to the data and can be used to remove it.
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import verde as vd

########################################################################################
# Our sample air temperature data from Texas has a clear trend from land to the ocean:

data = vd.datasets.fetch_texas_wind()
coordinates = (data.longitude, data.latitude)

plt.figure(figsize=(8, 6))
ax = plt.axes(projection=ccrs.Mercator())
plt.scatter(
    data.longitude,
    data.latitude,
    c=data.air_temperature_c,
    s=100,
    cmap="plasma",
    transform=ccrs.PlateCarree(),
)
plt.colorbar().set_label("Air temperature (C)")
vd.datasets.setup_texas_wind_map(ax)
plt.show()

########################################################################################
# We can estimate the polynomial coefficients for this trend:

trend = vd.Trend(degree=1).fit(coordinates, data.air_temperature_c)
print(trend.coef_)

########################################################################################
# More importantly, we can predict the trend values and remove them from our data:

trend_values = trend.predict(coordinates)
residuals = data.air_temperature_c - trend_values

fig, axes = plt.subplots(
    1, 2, figsize=(10, 6), subplot_kw=dict(projection=ccrs.Mercator())
)

ax = axes[0]
ax.set_title("Trend")
tmp = ax.scatter(
    data.longitude,
    data.latitude,
    c=trend_values,
    s=60,
    cmap="plasma",
    transform=ccrs.PlateCarree(),
)
plt.colorbar(tmp, ax=ax, orientation="horizontal", pad=0.06)
vd.datasets.setup_texas_wind_map(ax)

ax = axes[1]
ax.set_title("Residuals")
maxabs = vd.maxabs(residuals)
tmp = ax.scatter(
    data.longitude,
    data.latitude,
    c=residuals,
    s=60,
    cmap="bwr",
    vmin=-maxabs,
    vmax=maxabs,
    transform=ccrs.PlateCarree(),
)
plt.colorbar(tmp, ax=ax, orientation="horizontal", pad=0.08)
vd.datasets.setup_texas_wind_map(ax)
plt.show()

########################################################################################
# The fitting, prediction, and residual calculation can all be done in a single step
# using the :meth:`~verde.Trend.filter` method:

# filter always outputs coordinates and weights as well, which we don't need and will
# ignore here.
__, res_filter, __ = vd.Trend(degree=1).filter(coordinates, data.air_temperature_c)

print(np.allclose(res_filter, residuals))

########################################################################################
# Additionally, :class:`verde.Trend` implements the :ref:`gridder interface <overview>`
# and has the :meth:`~verde.Trend.grid` and :meth:`~verde.Trend.profile` methods.
