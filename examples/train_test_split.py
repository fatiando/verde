"""
Splitting data into train and test sets
=======================================


"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import verde as vd

# Let's split the Baja California shipborne bathymetry data
data = vd.datasets.fetch_baja_bathymetry()
coordinates = (data.longitude, data.latitude)
values = data.bathymetry_m

# Assign 30% of the data to the testing set.
test_size = 0.2

# Split the data randomly into training and testing. Set the random state
# (seed) so that we get the same result if running this code again.
train, test = vd.train_test_split(
    coordinates, values, test_size=test_size, random_state=123
)
# train and test are tuples = (coordinates, data, weights).
print("Train and test size for random splits:", train[0][0].size, test[0][0].size)

# A better strategy for spatial data is to first assign the data to blocks and
# then split the blocks randomly. The size of the blocks is controlled by the
# 'spacing' argument.
train_block, test_block = vd.train_test_split(
    coordinates,
    values,
    blocked=True,
    spacing=0.5,
    test_size=test_size,
    random_state=213,
)
# Verde will attempt to balance the data between the splits so that the desired
# amount is assigned to the test set. It won't be exact since blocks contain
# different amounts of data points.
print(
    "Train and test size for block splits: ",
    train_block[0][0].size,
    test_block[0][0].size,
)

# Cartopy requires setting the coordinate reference system (CRS) of the
# original data through the transform argument. Their docs say to use
# PlateCarree to represent geographic data.
crs = ccrs.PlateCarree()

# Make Mercator maps of the two different ways of splitting
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(10, 6), subplot_kw=dict(projection=ccrs.Mercator())
)

# Use an utility function to setup the tick labels and the land feature
vd.datasets.setup_baja_bathymetry_map(ax1)
vd.datasets.setup_baja_bathymetry_map(ax2)

ax1.set_title("Random splitting")
ax1.plot(*train[0], ".b", markersize=1, transform=crs, label="Train")
ax1.plot(*test[0], ".r", markersize=1, transform=crs, label="Test", alpha=0.5)

ax2.set_title("Blocked random splitting")
ax2.plot(*train_block[0], ".b", markersize=1, transform=crs, label="Train")
ax2.plot(*test_block[0], ".r", markersize=1, transform=crs, label="Test")
ax2.legend(loc="upper right")

plt.subplots_adjust(wspace=0.15, top=1, bottom=0, left=0.05, right=0.95)
plt.show()
