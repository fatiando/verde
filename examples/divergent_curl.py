"""
Divergence and curl of vector fields
====================================
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import pyproj
import verde as vd


# Fetch the wind speed data from Texas.
data = vd.datasets.fetch_texas_wind()
print(data.head())

# Separate out some of the data into utility variables
coordinates = (data.longitude.values, data.latitude.values)
region = vd.get_region(coordinates)
# Use a Mercator projection because Spline is a Cartesian gridder
projection = pyproj.Proj(proj="merc", lat_ts=data.latitude.mean())

# Split the data into a training and testing set. We'll fit the gridder on the training
# set and use the testing set to evaluate how well the gridder is performing.
train, test = vd.train_test_split(
    projection(*coordinates),
    (data.wind_speed_east_knots, data.wind_speed_north_knots),
    random_state=2,
)

# Chain together a blocked mean to avoid aliasing, a polynomial trend (Spline usually
# requires de-trended data), and finally a Spline for each component. Notice that
# BlockReduce can work on multicomponent data without the use of Vector.
chain = vd.Chain(
    [
        ("trend", vd.Vector([vd.Trend(degree=1) for i in range(2)])),
        (
            "spline",
            vd.Vector([vd.Spline(damping=1e-10, mindist=500e3) for i in range(2)]),
        ),
    ]
)
print(chain)

# Fit on the training data
chain.fit(*train)
# And score on the testing data. The best possible score is 1, meaning a perfect
# prediction of the test data.
score = chain.score(*test)
print("Cross-validation R^2 score: {:.2f}".format(score))

# Interpolate the wind speed onto a regular geographic grid and mask the data
# that are outside of the convex hull of the data points.
dims = ["latitude", "longitude"]
grid_full = chain.grid(region, spacing=20 / 60, projection=projection, dims=dims)
grid = vd.convexhull_mask(coordinates, grid=grid_full, projection=projection)


spacing = 10 / 60
step = 0.5 * spacing * 100e3

east_derivs = vd.Gradient(chain, step=step, direction=(1, 0)).grid(
    region, spacing=spacing, projection=projection, dims=dims
)
north_derivs = vd.Gradient(chain, step=step, direction=(0, 1)).grid(
    region, spacing=spacing, projection=projection, dims=dims
)

divergence = east_derivs.east_component + north_derivs.north_component
curl = north_derivs.east_component - east_derivs.north_component


# Make maps of the original and gridded wind speed
plt.figure(figsize=(6, 6))
ax = plt.axes(projection=ccrs.Mercator())

curl.plot(ax=ax, transform=ccrs.PlateCarree())

tmp = ax.quiver(
    grid.longitude.values,
    grid.latitude.values,
    grid.east_component.values,
    grid.north_component.values,
    width=0.0015,
    scale=100,
    color="tab:blue",
    transform=ccrs.PlateCarree(),
)

# Use an utility function to add tick labels and land and ocean features to the map.
vd.datasets.setup_texas_wind_map(ax)
plt.tight_layout()
plt.show()
