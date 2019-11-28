"""
A minimal integration test to make sure the most critical parts of Verde work
as expected.
"""
import numpy.testing as npt
import pyproj

from ..datasets import fetch_california_gps
from ..spline import Spline
from ..vector import Vector
from ..trend import Trend
from ..chain import Chain
from ..model_selection import train_test_split
from ..blockreduce import BlockMean
from ..coordinates import get_region
from ..mask import distance_mask


def test_minimal_integration_2d_gps():
    "Grid the 2D GPS data to make sure things don't break in obvious ways."
    data = fetch_california_gps()
    projection = pyproj.Proj(proj="merc", lat_ts=data.latitude.mean(), ellps="WGS84")
    proj_coords = projection(data.longitude.values, data.latitude.values)
    spacing = 12 / 60
    train, test = train_test_split(
        coordinates=proj_coords,
        data=(data.velocity_east, data.velocity_north),
        weights=(1 / data.std_east ** 2, 1 / data.std_north ** 2),
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
