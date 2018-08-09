"""
Test the utility functions.
"""
import sys
from unittest import mock

import pytest

from ..utils import parse_engine, dummy_jit


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
