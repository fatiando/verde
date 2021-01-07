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
    SplineCV
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
    BlockShuffleSplit
    BlockKFold

Coordinate Manipulation
-----------------------

.. autosummary::
   :toctree: generated/

    grid_coordinates
    scatter_points
    profile_coordinates
    get_region
    pad_region
    inside
    block_split
    rolling_window
    expanding_window

Projection
----------

.. autosummary::
   :toctree: generated/

    project_region
    project_grid

Masking
-------

.. autosummary::
   :toctree: generated/

    distance_mask
    convexhull_mask

Utilities
---------

.. autosummary::
   :toctree: generated/

    test
    maxabs
    variance_to_weights
    grid_to_table
    make_xarray_grid
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

    datasets.locate
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
    base.BaseBlockCrossValidator
    base.n_1d_arrays
    base.check_fit_input
    base.least_squares
