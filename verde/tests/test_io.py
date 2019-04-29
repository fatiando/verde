# pylint: disable=redefined-outer-name
"""
Test the I/O functions.
"""
import os
from tempfile import NamedTemporaryFile
from io import StringIO

import pytest
import numpy as np
import numpy.testing as npt

from ..io import load_surfer, _read_surfer_header, _check_surfer_integrity


@pytest.fixture
def sample_grid():
    "A simple grid for testing"
    grid = StringIO(
        """DSAA
        2 6
        1.0 1000.0
        2.0 2000.0
        11 61
        11 21 31 41 51 61
        12 22 32 42 52 27.5
        """
    )
    return grid


@pytest.fixture
def sample_grid_nans():
    "A simple grid with NaNs for testing"
    grid = StringIO(
        """DSAA
        2 6
        1.0 1000.0
        2.0 2000.0
        11 61
        11 21 1.70141e38 41 1.70141e38 61
        12 22 32 42 52 1.70141e38
        """
    )
    return grid


def test_load_surfer_header(sample_grid):
    "Tests the surfer header parser"
    grid_id, shape, region, data_range = _read_surfer_header(sample_grid)
    assert grid_id == "DSAA"
    assert shape == (2, 6)
    npt.assert_allclose(region, (2, 2000, 1, 1000))
    npt.assert_allclose(data_range, (11, 61))


def test_load_surfer(sample_grid):
    "Tests the main function on a StringIO file-like object"
    grid = load_surfer(sample_grid)
    assert grid.dims == ("northing", "easting")
    assert grid.attrs["gridID"] == "DSAA"
    assert grid.shape == (2, 6)
    npt.assert_allclose(grid.northing.min(), 1)
    npt.assert_allclose(grid.northing.max(), 1000)
    npt.assert_allclose(grid.easting.min(), 2)
    npt.assert_allclose(grid.easting.max(), 2000)
    npt.assert_allclose(grid.min(), 11)
    npt.assert_allclose(grid.max(), 61)
    field = np.array([[11, 21, 31, 41, 51, 61], [12, 22, 32, 42, 52, 27.5]])
    npt.assert_allclose(grid.values, field)


def test_load_surfer_nans(sample_grid_nans):
    "Tests the main function on a StringIO file-like object"
    grid = load_surfer(sample_grid_nans)
    assert grid.dims == ("northing", "easting")
    assert grid.attrs["gridID"] == "DSAA"
    assert grid.shape == (2, 6)
    npt.assert_allclose(grid.northing.min(), 1)
    npt.assert_allclose(grid.northing.max(), 1000)
    npt.assert_allclose(grid.easting.min(), 2)
    npt.assert_allclose(grid.easting.max(), 2000)
    npt.assert_allclose(grid.min(), 11)
    npt.assert_allclose(grid.max(), 61)
    field = np.array([[11, 21, np.nan, 41, np.nan, 61], [12, 22, 32, 42, 52, np.nan]])
    npt.assert_allclose(grid.values, field)


def test_load_surfer_file(sample_grid):
    "Tests the main function on a file on disk"
    tmpfile = NamedTemporaryFile(delete=False)
    tmpfile.close()
    try:
        with open(tmpfile.name, "w") as infile:
            infile.write(sample_grid.getvalue())
        grid = load_surfer(tmpfile.name)
        assert grid.dims == ("northing", "easting")
        assert grid.attrs["gridID"] == "DSAA"
        assert grid.attrs["file"] == tmpfile.name
        assert grid.shape == (2, 6)
        npt.assert_allclose(grid.northing.min(), 1)
        npt.assert_allclose(grid.northing.max(), 1000)
        npt.assert_allclose(grid.easting.min(), 2)
        npt.assert_allclose(grid.easting.max(), 2000)
        npt.assert_allclose(grid.min(), 11)
        npt.assert_allclose(grid.max(), 61)
        field = np.array([[11, 21, 31, 41, 51, 61], [12, 22, 32, 42, 52, 27.5]])
        npt.assert_allclose(grid.values, field)
    finally:
        os.remove(tmpfile.name)


def test_load_surfer_integrity():
    "Tests the surfer integrity check function"
    field = np.array([[1, 2], [3, 4], [5, 6]])
    _check_surfer_integrity(field, shape=(3, 2), data_range=(1, 6))
    with pytest.raises(IOError):
        _check_surfer_integrity(field, shape=(2, 3), data_range=(1, 6))
    with pytest.raises(IOError):
        _check_surfer_integrity(field, shape=(3, 2), data_range=(-2, 3))
    # Make a corrupted grid file
    wrong_shape = StringIO(
        """DSAA
        2 5
        1.0 1000.0
        2.0 2000.0
        11 61
        11 21 31 41 51 61
        12 22 32 42 52 1.70141e38
        """
    )
    with pytest.raises(IOError):
        load_surfer(wrong_shape)
