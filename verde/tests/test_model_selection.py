"""
Test the model selection code (cross-validation, etc).
"""
import pytest
from sklearn.model_selection import ShuffleSplit
import numpy as np
import numpy.testing as npt
from dask.distributed import Client

from .. import Trend, grid_coordinates
from ..model_selection import cross_val_score, BlockShuffleSplit, BlockKFold


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


def test_blockshufflesplit_fails_balancing():
    "Should raise an exception if balancing < 1."
    with pytest.raises(ValueError):
        BlockShuffleSplit(spacing=1, balancing=0)


@pytest.mark.parametrize("test_size", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9])
def test_blockshufflesplit_balancing(test_size):
    "Make sure that the sets have the right number of points"
    coords = np.random.RandomState(seed=0).multivariate_normal(
        mean=[5, -7.5], cov=[[4, 0], [0, 9]], size=1000,
    )
    npoints = coords.shape[0]
    train_size = 1 - test_size
    cv = BlockShuffleSplit(spacing=1, random_state=0, test_size=test_size, balancing=20)
    for train, test in cv.split(coords):
        npt.assert_allclose(train.size / npoints, train_size, atol=0.01)
        npt.assert_allclose(test.size / npoints, test_size, atol=0.01)


def test_blockkfold_fails_n_splits():
    "Should raise an exception if n_splits < 2."
    BlockKFold(spacing=1, n_splits=2)
    with pytest.raises(ValueError):
        BlockKFold(spacing=1, n_splits=1)
