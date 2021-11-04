"""
Derivatives of vector fields
============================
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import pyproj
import verde as vd


# Fetch the wind speed data from Texas.
data = vd.datasets.fetch_texas_wind()
# Separate out some of the data into utility variables
coordinates = (data.longitude.values, data.latitude.values)
region = vd.get_region(coordinates)
# Use a Mercator projection because Spline is a Cartesian gridder
projection = pyproj.Proj(proj="merc", lat_ts=data.latitude.mean())

# Split the data into a training and testing set. We'll fit the gridder on the
# training set and use the testing set to evaluate how well the gridder is
# performing.
train, test = vd.train_test_split(
    coordinates=projection(*coordinates),
    data=(data.wind_speed_east_knots, data.wind_speed_north_knots),
    test_size=0.1,
    random_state=1,
)

estimator = vd.Vector([vd.Spline(damping=1e-6, mindist=500e3) for i in range(2)])

# Fit on the training data
estimator.fit(*train)
# And score on the testing data. The best possible score is 1, meaning a perfect
# prediction of the test data.
score = estimator.score(*test)
print("Validation score (R²): {:.2f}".format(score))

# Interpolate the wind speed onto a regular geographic grid and mask the data
# that are outside of the convex hull of the data points.
dims = ["latitude", "longitude"]
grid = estimator.grid(region, spacing=20 / 60, projection=projection, dims=dims)
grid = vd.convexhull_mask(coordinates, grid=grid, projection=projection)

spacing = 10 / 60
step = 0.5 * spacing * 100e3

east_derivs = vd.Derivative(estimator, step=step, direction=(1, 0)).grid(
    region, spacing=spacing, projection=projection, dims=dims
)
north_derivs = vd.Derivative(estimator, step=step, direction=(0, 1)).grid(
    region, spacing=spacing, projection=projection, dims=dims
)

divergence = east_derivs.east_component + north_derivs.north_component

# Make maps of the original and gridded wind speed
plt.figure(figsize=(6, 6))
ax = plt.axes(projection=ccrs.Mercator())

divergence.plot(ax=ax, transform=ccrs.PlateCarree())

tmp = ax.quiver(
    data.longitude.values,
    data.latitude.values,
    data.wind_speed_east_knots.values,
    data.wind_speed_north_knots.values,
    width=0.003,
    scale=100,
    color="black",
    transform=ccrs.PlateCarree(),
)

# Use an utility function to add tick labels and land and ocean features to the map.
vd.datasets.setup_texas_wind_map(ax)
plt.tight_layout()
plt.show()
