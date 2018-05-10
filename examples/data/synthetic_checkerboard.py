"""
Checkerboard function
=====================

The :class:`verde.datasets.CheckerBoard` class generates synthetic data in a
checkerboard pattern. Use it like any gridder class.
"""
import matplotlib.pyplot as plt
import verde as vd

# Instantiate the data generator class and fit it to set the data region.
synth = vd.datasets.CheckerBoard().fit()

# Default values are provided for the region and the wavelengths of the
# function (these are determined from the region).
print("region:", synth.region_)
print("wavelengths (east, north):", synth.w_east, synth.w_north)

# The CheckerBoard class behaves like any gridder class
print("Checkerboard value at (2000, -2500):",
      synth.predict(easting=2000, northing=-2500))

# Generating a grid results in a xarray.Dataset
grid = synth.grid()
print("\nData grid:\n", grid)

# while a random scatter generates a pandas.DataFrame
table = synth.scatter(size=100)
print("\nTable of scattered data:\n", table.head())

fig = plt.figure(figsize=(5.5, 4))
ax = plt.subplot(111)
ax.set_title('CheckerBoard')
ax.set_aspect('equal')
grid.scalars.plot.pcolormesh(ax=ax)
plt.tight_layout(pad=0)
plt.show()
