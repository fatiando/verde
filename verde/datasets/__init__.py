"""
Loading test data and generating synthetic datasets.
"""
from .synthetic import CheckerBoard
from .sample_data import (
    fetch_baja_bathymetry,
    setup_baja_bathymetry_map,
    fetch_rio_magnetic,
    setup_rio_magnetic_map,
    fetch_california_gps,
    setup_california_gps_map,
)
