.. _api:

API Reference
=============

.. automodule:: verde

.. currentmodule:: verde

Interpolators
-------------

.. autosummary::
   :toctree: generated/

    Spline
    VectorSpline2D
    ScipyGridder

Data Processing
---------------

.. autosummary::
   :toctree: generated/

    BlockReduce
    BlockMean
    Trend

Composite Estimators
--------------------

.. autosummary::
   :toctree: generated/

    Chain
    Vector

Model Selection
---------------

.. autosummary::
   :toctree: generated/

    train_test_split
    cross_val_score

Coordinate Manipulation
-----------------------

.. autosummary::
   :toctree: generated/

    grid_coordinates
    scatter_points
    profile_coordinates
    get_region
    pad_region
    project_region
    inside
    block_split

Utilities
---------

.. autosummary::
   :toctree: generated/

    test
    maxabs
    distance_mask
    variance_to_weights
    grid_to_table
    median_distance

Input/Output
------------

.. autosummary::
   :toctree: generated/

    load_surfer


.. automodule:: verde.datasets

.. currentmodule:: verde

Datasets
--------

.. autosummary::
   :toctree: generated/

    datasets.CheckerBoard
    datasets.fetch_baja_bathymetry
    datasets.setup_baja_bathymetry_map
    datasets.fetch_california_gps
    datasets.setup_california_gps_map
    datasets.fetch_texas_wind
    datasets.setup_texas_wind_map
    datasets.fetch_rio_magnetic
    datasets.setup_rio_magnetic_map

Base Classes and Functions
--------------------------

.. autosummary::
   :toctree: generated/

    base.BaseGridder
    base.n_1d_arrays
    base.check_fit_input
    base.least_squares
