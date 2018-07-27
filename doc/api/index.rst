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
    Vector2D
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
    Components

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
    distance_mask

Utilities
---------

.. autosummary::
   :toctree: generated/

    test
    maxabs
    variance_to_weights

Base Classes
------------

.. autosummary::
   :toctree: generated/

    BaseGridder
