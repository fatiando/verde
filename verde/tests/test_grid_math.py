"""
Test the functions in verde.grid_math
"""
import pytest

from ..grid_math import distance_mask


def test_distance_mask():
    "Check that the mask works for basic input"
    region = (0, 5, -10, -5)
    spacing = 1
    mask = distance_mask((2.5, -7.5), maxdist=2, region=region,
                         spacing=spacing)
    true = [[False, False, False, False, False, False],
            [False, False, True, True, False, False],
            [False, True, True, True, True, False],
            [False, True, True, True, True, False],
            [False, False, True, True, False, False],
            [False, False, False, False, False, False]]
    assert mask.tolist() == true


def test_distance_mask_fails():
    "Check that the function fails if no coordinates or region are given"
    with pytest.raises(ValueError):
        distance_mask((2.5, -7.5), maxdist=2)
