Making your first grid
======================

TODO:

* Use an actual projection here
* Simplify the intro tutorial. It should grid using geographic coordinates without projections and no blocking. Then refer to the tutorial.
* This one produces cartesian grids.
* Next tutorial will show how to make geographic grids

Intro:

* Load the new carribean or hawaiian bathymetry data data
* Fit a spline or Cubic to the raw geographic data
* Make a grid
* Make a profile

Tutorial structure:

1. Make a grid in Cartesian coordinates using block reduce, projection, and Spline with defaults. Use new bathymetry data.
2. Make a geographic grid by passing the projection to the gridder. Explain a bit of what is going on. Also use the bathymetry data.
3. Use weights when we have uncertainties. How to use them in BlockMean. How to use them in Spline. Use GPS vertical data. All geographic.
3. Make a Chain with blockmean and spline.
3. Using cross-validation to find the model performance. Use train-test-split. Use block kfold. Use GPS data. Use a chain. All geographic.
4. Model selection by grid search. Find the best spline damping. Use GPS data. All geographic with a Chain.
5. Interpolating vectors with Vector and vectorspline. Use all of the above already.


How-tos:

* Interpolate large datasets. Use KNearest.
* Project a grid.
* Slice points to a given window.
* Mask points according to distance.
* Mask points out of the convexhull.
* Decimate large datasets. Run block reduce on the volcano lidar.
* Estimate a polinomial trend.
* Bin statistics. Calculate standard deviation within blocks on volcano lidar. Measure of roughness.
* Split spatial data by blocks. Run block_split.
* Split points along rollowing windows.

Explanations:

* The different types of grid and line coordinates.
* How spline interpolation works. Build the Jacobian and show how to solve the system.
* Conventions

.. jupyter-execute::

   import verde as vd
   import ensaio
   import matplotlib.pyplot as plt
   import numpy as np
   import pandas as pd

   path_to_data = ensaio.fetch_bushveld_gravity(version=1)
   data = pd.read_csv(path_to_data)
   data = data.assign(easting=data.longitude * 111e3, northing=data.latitude * 111e3)
   data


.. jupyter-execute::

   spacing = 5e3
   print(np.median(vd.median_distance((data.easting, data.northing)))
   elevation = data.height_sea_level_m
   coordinates = data.easting, data.northing

.. jupyter-execute::

   plt.scatter(coordinates[0], coordinates[1], c=elevation, s=1)
   plt.colorbar()
   plt.show()

.. jupyter-execute::

   gridder = vd.Spline()

   gridder.fit(coordinates, elevation)

   grid = gridder.grid(spacing=spacing, data_names="elevation")

   grid.elevation.plot()
