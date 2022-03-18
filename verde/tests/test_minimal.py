# Copyright (c) 2017 The Verde Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
A minimal integration test to make sure the most critical parts of Verde work
as expected.
"""
import numpy.testing as npt

from ..blockreduce import BlockMean
from ..chain import Chain
from ..coordinates import get_region
from ..datasets import fetch_california_gps
from ..mask import distance_mask
from ..model_selection import train_test_split
from ..spline import Spline
from ..trend import Trend
from ..vector import Vector


def projection(longitude, latitude):
    """
    Simple projection function for testing.
    """
    return longitude * 111e3, latitude * 111e3


def test_minimal_integration_2d_gps():
    "Grid the 2D GPS data to make sure things don't break in obvious ways."
    data = fetch_california_gps()
    proj_coords = projection(data.longitude.values, data.latitude.values)
    spacing = 12 / 60
    train, test = train_test_split(
        coordinates=proj_coords,
        data=(data.velocity_east, data.velocity_north),
        weights=(1 / data.std_east**2, 1 / data.std_north**2),
        random_state=1,
    )
    chain = Chain(
        [
            ("mean", BlockMean(spacing=spacing * 111e3, uncertainty=True)),
            ("trend", Vector([Trend(1), Trend(1)])),
            ("spline", Vector([Spline(damping=1e-10), Spline(damping=1e-10)])),
        ]
    )
    chain.fit(*train)
    score = chain.score(*test)
    npt.assert_allclose(0.99, score, atol=0.01)
    # This part just makes sure there are no exceptions when calling this code.
    region = get_region((data.longitude, data.latitude))
    grid = chain.grid(
        region=region,
        spacing=spacing,
        projection=projection,
        dims=["latitude", "longitude"],
    )
    grid = distance_mask(
        (data.longitude, data.latitude),
        maxdist=spacing * 2 * 111e3,
        grid=grid,
        projection=projection,
    )
