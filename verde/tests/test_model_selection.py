"""
Test the model selection code (cross-validation, etc).
"""
from sklearn.model_selection import ShuffleSplit
import numpy.testing as npt
from dask.distributed import Client

from .. import Trend, grid_coordinates
from ..model_selection import cross_val_score


def test_cross_val_score_client():
    "Test the deprecated dask Client interface"
    coords = grid_coordinates((0, 10, -10, -5), spacing=0.1)
    data = 10 - coords[0] + 0.5 * coords[1]
    model = Trend(degree=1)
    nsplits = 5
    cross_validator = ShuffleSplit(n_splits=nsplits, random_state=0)
    client = Client(processes=False)
    futures = cross_val_score(model, coords, data, cv=cross_validator, client=client)
    scores = [future.result() for future in futures]
    client.close()
    assert len(scores) == nsplits
    npt.assert_allclose(scores, 1)
