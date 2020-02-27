"""
Test the utility functions.
"""
from unittest import mock

import numpy as np
import numpy.testing as npt
import xarray as xr
from scipy.spatial import cKDTree  # pylint: disable=no-name-in-module
import pytest

from ..coordinates import grid_coordinates
from ..utils import parse_engine, dummy_jit, kdtree, grid_to_table
from .. import utils


def test_parse_engine():
    "Check that it works for common input"
    assert parse_engine("numba") == "numba"
    assert parse_engine("numpy") == "numpy"
    with mock.patch.object(utils, "numba", None):
        assert parse_engine("auto") == "numpy"
    with mock.patch.object(utils, "numba", mock.MagicMock()):
        assert parse_engine("auto") == "numba"


def test_parse_engine_fails():
    "Check that the exception is raised for invalid engines"
    with pytest.raises(ValueError):
        parse_engine("some invalid engine")


def test_dummy_jit():
    "Make sure the dummy function raises an exception"

    @dummy_jit(target="cpt")
    def function():
        "Some random function"
        return 0

    with pytest.raises(RuntimeError):
        function()


def test_kdtree():
    "Test that the kdtree returned works for query"
    coords = grid_coordinates((-10, 0, 0, 20), spacing=1)
    for use_pykdtree in [True, False]:
        tree = kdtree(coords, use_pykdtree=use_pykdtree)
        dist, labels = tree.query(np.array([[-10, 0.1]]))
        assert labels.size == 1
        assert labels[0] == 0
        npt.assert_allclose(dist, 0.1)
        if not use_pykdtree:
            assert isinstance(tree, cKDTree)


def test_grid_to_table_order():
    "Check that coordinates are in the right order when converting to tables"
    lon, lat = grid_coordinates(region=(1, 10, -10, -1), shape=(3, 4))
    data = lon ** 2
    # If the DataArray is created with coords in an order that doesn't match
    # the dims (which is valid), we were getting it wrong because we were
    # relying on the order of the coords instead of dims. This test would have
    # caught that bug.
    grid = xr.DataArray(
        data=data,
        coords={"longitude": lon[0, :], "latitude": lat[:, 0]},
        dims=("latitude", "longitude"),
    ).to_dataset(name="field")
    table = grid_to_table(grid)
    true_lat = [-10, -10, -10, -10, -5.5, -5.5, -5.5, -5.5, -1, -1, -1, -1]
    true_lon = [1, 4, 7, 10, 1, 4, 7, 10, 1, 4, 7, 10]
    true_field = [1, 16, 49, 100, 1, 16, 49, 100, 1, 16, 49, 100]
    npt.assert_allclose(true_lat, table.latitude)
    npt.assert_allclose(true_lon, table.longitude)
    npt.assert_allclose(true_field, table.field)
