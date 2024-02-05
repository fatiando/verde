.. _conventions:

Important conventions
=====================

Here are a few of the conventions we use across Verde:

* Coordinates can be **Cartesian or Geographic**. We generally make **no
  assumptions** about which one you're using.
* **Order of coordinates:** West-East and South-North. All functions and
  classes expect coordinates in this order. This applies to the actual
  coordinate values, bounding regions, grid spacing, etc. Exceptions to this
  rule are the ``dims`` and ``shape`` arguments.
* **We don't use names like "x" and "y"** to avoid ambiguity. Cartesian
  coordinates are "easting" and "northing" and Geographic coordinates are
  "longitude" and "latitude".
* The term **"region" means the bounding box** of the data. It is ordered west,
  east, south, north.

.. _gridder_interface:

The gridder interface
---------------------

All gridding and trend estimation classes in Verde share the same interface
(they all inherit from :class:`verde.base.BaseGridder`). Since most gridders
in Verde are linear models, we based our gridder interface on the
`scikit-learn <http://scikit-learn.org/>`__ estimator interface: they all
implement a :meth:`~verde.base.BaseGridder.fit` method that estimates the
model parameters based on data and a :meth:`~verde.base.BaseGridder.predict`
method that calculates new data based on the estimated parameters.

Unlike scikit-learn, our data model is not a feature matrix and a target
vector (e.g., ``est.fit(X, y)``) but a tuple of coordinate arrays and a data
vector (e.g., ``grd.fit((easting, northing), data)``). This makes more sense
for spatial data and is common to all classes and functions in Verde.
