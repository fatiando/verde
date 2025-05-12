# Copyright (c) 2017 The Verde Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test the synthetic data generation functions and classes.
"""
import numpy.testing as npt

from verde.synthetic import CheckerBoard


def test_checkerboard_scatter_projection():
    "Test generating scattered points when passing in a projection"

    # Lets say the projection is doubling the coordinates
    def proj(lon, lat, inverse=False):
        "Project from the new coordinates to the original"
        if inverse:
            return (lon / 2, lat / 2)
        return (lon * 2, lat * 2)

    region = (0, 10, -10, -5)
    region_proj = (0, 5, -5, -2.5)
    checker = CheckerBoard(region=region)
    checker_proj = CheckerBoard(region=region_proj)
    scatter = checker.scatter(region, 1000, random_state=0, projection=proj)
    scatter_proj = checker_proj.scatter(region, 1000, random_state=0)
    npt.assert_allclose(scatter.scalars, scatter_proj.scalars)
    npt.assert_allclose(scatter.easting, scatter_proj.easting)
    npt.assert_allclose(scatter.northing, scatter_proj.northing)
