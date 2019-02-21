"""
Test the utility functions.
"""
import sys
from unittest import mock

import numpy as np
import numpy.testing as npt
from scipy.spatial import cKDTree  # pylint: disable=no-name-in-module
import pytest

from ..coordinates import grid_coordinates
from ..utils import parse_engine, dummy_jit, kdtree


def test_parse_engine():
    "Check that it works for common input"
    assert parse_engine("numba") == "numba"
    assert parse_engine("numpy") == "numpy"
    with mock.patch.dict(sys.modules, {"numba": None}):
        assert parse_engine("auto") == "numpy"
    with mock.patch.dict(sys.modules, {"numba": mock.MagicMock()}):
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


def test_surfer_header():
    "Loads and parses the fake surfer header"
    fake_header = ["DSAA", "100 200", "1.0 1000.0", "2.0 2000.0", "3.0 300.0"]

    def load_surfer_header(fname):
        with StringIO(fname) as input_file:
            # DSAA is a Surfer ASCII GRD ID
            grd_id = input_file.readline().strip()
            # Read the number of columns (ny) and rows (nx)
            ydims, xdims = [int(s) for s in input_file.readline().split()]
            # Our x points North, so the first thing we read is y, not x.
            south, north = [float(s) for s in input_file.readline().split()]
            west, east = [float(s) for s in input_file.readline().split()]
            dmin, dmax = [float(s) for s in input_file.readline().split()]

        return [grd_id, ydims, xdims, south, north, west, east, dmin, dmax]

    test_header = "\n".join(fake_header)
    grid_id_name = fake_header[0]
    ydimension = int(fake_header[1].split(" ")[0])
    xdimension = int(fake_header[1].split(" ")[1])
    southmin = float(fake_header[2].split(" ")[0])
    northmax = float(fake_header[2].split(" ")[1])
    westmin = float(fake_header[3].split(" ")[0])
    eastmax = float(fake_header[3].split(" ")[1])
    zmin = float(fake_header[4].split(" ")[0])
    zmax = float(fake_header[4].split(" ")[1])

    surfer_grd = load_surfer_header(test_header)

    assert surfer_grd == [
        grid_id_name,
        ydimension,
        xdimension,
        southmin,
        northmax,
        westmin,
        eastmax,
        zmin,
        zmax,
    ]
