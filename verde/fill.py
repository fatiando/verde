# Copyright (c) 2017 The Verde Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
General utilities.
"""
import copy
import warnings

import xarray as xr

from .neighbors import KNeighbors
from .utils import grid_to_table


def fill_missing(
    grid,
    interpolator=None,
):
    """
    Fill missing values in a grid with a choice of interpolation method.
    This will  fill missing values for all variables in the supplied grid.

    Interpolation methods include nearest neighbor, linear, trend, cubic,
    or splines.

    Parameters
    ----------
    grid : :class:`xarray.DataArray` | :class:`xarray.Dataset`
        A 2D grid with one or more data variable, some of which may have
        missing values (NaNs).
    interpolator : class | None
        The verde interpolator class instance to use for filling missing
        values. Can be one of the following  :class:`verde.KNeighbors`,
        :class:`verde.Linear`, :class:`verde.Cubic`, :class:`verde.Spline`,
        :class:`verde.SplineCV`, :class:`verde.Trend`, by default is
        class:`verde.KNeighbors` using the nearest 5 neighbors.

    Returns
    -------
    filled_grid : :class:`xarray.DataArray` | :class:`xarray.Dataset`
        A 2D grid with the NaN values filled for each variable.
    """
    grid = grid.copy()

    if interpolator is None:
        interpolator = KNeighbors(k=5)

    # if input was a datarray turn into dataset
    if isinstance(grid, xr.DataArray):
        ds = grid.to_dataset()
    else:
        ds = grid

    # get grid coordinate names
    coord_names = list(ds.coords)

    # iterate over variables
    for var_name, var_da in ds.items():

        # turn grid into dataframe
        df = grid_to_table(var_da)

        # if no nans, continue without change original grid
        if not df[var_name].isna().any():
            continue

        # drop rows with data column is NaN
        df_no_nans = df[df[var_name].notna()]

        # get coordinate columns (first two columns)
        coords_no_nans = (df_no_nans.iloc[:, 1], df_no_nans.iloc[:, 0])

        interp = copy.deepcopy(interpolator)

        interp.fit(coords_no_nans, df_no_nans[var_name])

        # predict only at NaNs and add to dataframe
        df_nans = df[df[var_name].isna()]
        df.loc[df_nans.index, var_name] = interp.predict(
            (df_nans.iloc[:, 1], df_nans.iloc[:, 0])
        )

        # convert to dataarray
        filled_da = df.set_index([coord_names[0], coord_names[1]]).to_xarray()[var_name]

        # warn if still nans due to no extrapolation allowed for
        # `Cubic` and `Linear` interpolators
        if filled_da.isnull().any():
            msg = (
                "NaNs are still present in this grid! This may be due "
                f"to the choice of interpolator {type(interp)}, "
                "some of which don't allow extrapolation. To fill the "
                "remaining values run `fill_missing()` again with an "
                "interpolator which allows extrpolation. We recommend "
                "`vd.KNeighbors` if you have a large grid (>~10,000 points) "
                "or `vd.Spline` or `vd.SplineCV` if you  have a a smaller grid "
                "or require smoother results."
            )
            warnings.warn(msg, UserWarning, stacklevel=2)

        # if input was a datarray, fill nans with new values and return that
        # if it was a dataset, update each variable
        if isinstance(grid, xr.DataArray):
            grid = grid.where(grid.notnull(), filled_da)
        else:
            grid[var_name] = grid[var_name].where(grid[var_name].notnull(), filled_da)

    return grid
