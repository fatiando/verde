# Copyright (c) 2017 The Verde Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Checkerboard function
=====================

The :class:`verde.datasets.CheckerBoard` class generates synthetic data in a
checkerboard pattern. It has the same data generation methods that most
gridders have: predict, grid, scatter, and profile.
"""
import matplotlib.pyplot as plt
import verde as vd

# Instantiate the data generator class and fit it to set the data region.
synth = vd.datasets.CheckerBoard()

# Default values are provided for the wavelengths of the function determined
# from the region.
print("wavelengths (east, north):", synth.w_east_, synth.w_north_)

# The CheckerBoard class behaves like any gridder class
print(
    "Checkerboard value at (easting=2000, northing=-2500):",
    synth.predict((2000, -2500)),
)

# Generating a grid results in a xarray.Dataset
grid = synth.grid(shape=(150, 100))
print("\nData grid:\n", grid)

# while a random scatter generates a pandas.DataFrame
table = synth.scatter(size=100)
print("\nTable of scattered data:\n", table.head())

fig = plt.figure(figsize=(5.5, 4))
ax = plt.subplot(111)
ax.set_title("CheckerBoard")
ax.set_aspect("equal")
grid.scalars.plot.pcolormesh(ax=ax)
plt.tight_layout(pad=0)
plt.show()
