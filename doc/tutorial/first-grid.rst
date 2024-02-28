Making your first grid
======================

.. jupyter-execute::

   import verde as vd
   import ensaio
   import matplotlib.pyplot as plt
   import numpy as np
   import pandas as pd
   import pyproj

   path_to_data = ensaio.fetch_caribbean_bathymetry(version=2)
   data = pd.read_csv(path_to_data)
   data


.. jupyter-execute::

   projection = pyproj.Proj(proj="merc", lat_ts=data.latitude.mean())
   easting, northing = projection(easting, northing)

.. jupyter-execute::

   spacing = 10e3
   blockmedian = vd.BlockReduce(np.median, spacing=spacing)
   coordinates, bathymetry = blockmedian.filter((easting, northing), data.bathymetry_m)
   print(bathymetry.size)

.. jupyter-execute::

   plt.scatter(coordinates[0], coordinates[1], c=bathymetry, s=1)
   plt.colorbar()
   plt.show()

.. jupyter-execute::

   # gridder = vd.Spline()
   # gridder.fit(coordinates, elevation)
   # grid = gridder.grid(spacing=spacing, data_names="elevation")
   # grid.elevation.plot()
