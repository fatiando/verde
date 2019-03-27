"""
Test the utility functions.
"""
import sys
from unittest import mock
from io import StringIO

import numpy as np
import numpy.testing as npt
from scipy.spatial import cKDTree  # pylint: disable=no-name-in-module
import pytest

from ..coordinates import grid_coordinates
from ..utils import (
    parse_engine,
    dummy_jit,
    kdtree,
    _read_surfer_header,
    _create_surfer_field,
    _check_surfer_integrity,
)


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
    "Tests the surfer header parser"
    test_header = StringIO(
        "DSAA\n 2 6\n 1.0 1000.0\n 2.0 2000.0\n 11 61\n"
        "11 21 31 41 51 61\n 12 22 32 42 52 1.70141e38"
    )
    (
        grid_id,
        ydims,
        xdims,
        south,
        north,
        west,
        east,
        dmin,
        dmax,
    ) = _read_surfer_header(test_header)
    grd_id = "DSAA"
    ydimensions, xdimensions = 2, 6
    southmin, northmax = 1.0, 1000.0
    westmin, eastmax = 2.0, 2000.0
    zmin, zmax = 11, 61
    assert grid_id == grd_id
    assert ydims == ydimensions
    assert xdims == xdimensions
    assert south == southmin
    assert north == northmax
    assert west == westmin
    assert east == eastmax
    assert dmin == zmin
    assert dmax == zmax


def test_surfer_field():
    "Tests the surfer field creation function"
    test_header = StringIO(
        "DSAA\n 2 6\n 1.0 1000.0\n 2.0 2000.0\n 11 61\n"
        "11 21 31 41 51 61\n 12 22 32 42 52 1.70141e38"
    )
    (
        grid_id,
        ydims,
        xdims,
        south,
        north,
        west,
        east,
        dmin,
        dmax,
    ) = _read_surfer_header(test_header)
    field, dims, coords = _create_surfer_field(
        test_header, south, north, west, east, ydims, xdims
    )

    fld = np.asarray([[11, 21, 31, 41, 51, 61],
     [12, 22, 32, 42, 52, 1.70141e38]])
    dimensions = ["northing", "easting"]
    northing = np.linspace(south, north, ydims)
    easting = np.linspace(west, east, xdims)

    assert dims == dimensions
    npt.assert_array_equal(field, fld)
    npt.assert_array_equal(coords["northing"], northing)
    npt.assert_array_equal(coords["easting"], easting)


def test_surfer_integrity():
    "Tests the surfer integrity check function"
    test_header = StringIO(
        "DSAA\n 2 6\n 1.0 1000.0\n 2.0 2000.0\n 11 61\n"
        "11 21 31 41 51 61\n 12 22 32 42 52 1.70141e38"
    )
    (
        grid_id,
        ydims,
        xdims,
        south,
        north,
        west,
        east,
        dmin,
        dmax,
    ) = _read_surfer_header(test_header)
    field, dims, coords = _create_surfer_field(
        test_header, south, north, west, east, ydims, xdims
    )
    _check_surfer_integrity(field, ydims, xdims, dmin, dmax)
