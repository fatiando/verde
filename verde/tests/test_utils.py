"""
Test the misc utilities in verde
"""
import pytest

from ..utils import check_region, profile_coordinates


def test_check_region():
    "Make sure an exception is raised for bad regions"
    with pytest.raises(ValueError):
        check_region([])
    with pytest.raises(ValueError):
        check_region([1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        check_region([1, 2, 3])
    with pytest.raises(ValueError):
        check_region([1, 2, 3, 1])
    with pytest.raises(ValueError):
        check_region([2, 1, 3, 4])
    with pytest.raises(ValueError):
        check_region([-1, -2, -4, -3])
    with pytest.raises(ValueError):
        check_region([-2, -1, -2, -3])


def test_profile_coordiantes_fails():
    "Should raise an exception for invalid input"
    with pytest.raises(ValueError):
        profile_coordinates((0, 1), (1, 2), size=0)
    with pytest.raises(ValueError):
        profile_coordinates((0, 1), (1, 2), size=-10)
    with pytest.raises(ValueError):
        profile_coordinates((0, 1), (1, 2), size=10, coordinate_system="meh")
    with pytest.raises(NotImplementedError):
        profile_coordinates((0, 1), (1, 2), size=10,
                            coordinate_system="geographic")
