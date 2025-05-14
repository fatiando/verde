# Copyright (c) 2017 The Verde Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
# Import functions/classes to make the public API
from . import synthetic
from ._version import __version__
from .blockreduce import BlockMean, BlockReduce
from .coordinates import (
    expanding_window,
    get_region,
    grid_coordinates,
    inside,
    longitude_continuity,
    pad_region,
    profile_coordinates,
    rolling_window,
    scatter_points,
)
from .distances import median_distance
from .io import load_surfer
from .mask import convexhull_mask, distance_mask
from .model_selection import (
    BlockKFold,
    BlockShuffleSplit,
    cross_val_score,
    train_test_split,
)
from .neighbors import KNeighbors
from .projections import project_grid, project_region
from .scipygridder import Cubic, Linear
from .spline import Spline, SplineCV
from .trend import Trend
from .utils import grid_to_table, make_xarray_grid, maxabs, variance_to_weights
from .vector import Vector, VectorSpline2D

# Append a leading "v" to the generated version by setuptools_scm
__version__ = f"v{__version__}"
