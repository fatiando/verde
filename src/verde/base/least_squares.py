# Copyright (c) 2017 The Verde Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Functions for least-squares fitting with optional regularization.
"""
from warnings import warn

import numpy as np
import scipy as sp


def least_squares(jacobian, data, weights, damping=None, copy_jacobian=False):
    """
    Solve a weighted least-squares problem with optional damping regularization

    Scales the Jacobian matrix so that each column has unit variance. This
    helps keep the regularization parameter in a sensible range. The scaling is
    undone before returning the estimated parameters so that scaling isn't
    required for predictions. Doesn't normalize the column means because that
    operation can't be undone.

    .. warning::

        Setting `copy_jacobian` to True will copy the Jacobian matrix, doubling
        the memory required. Use it only if the Jacobian matrix is needed
        afterwards.

    Parameters
    ----------
    jacobian : 2d-array
        The Jacobian/sensitivity/feature matrix.
    data : 1d-array
        The data array. Must be a single 1D array. If fitting multiple data
        components, stack the arrays and the Jacobian matrices.
    weights : None or 1d-array
        The data weights. Like the data, this must also be a 1D array. Stack
        the weights in the same order as the data. Use ``weights=None`` to fit
        without weights.
    damping : None or float
        The positive damping (Tikhonov 0th order) regularization parameter. If
        ``damping=None``, will use a regular least-squares fit.
    copy_jacobian: bool
        If False, the Jacobian matrix will be scaled inplace. If True, the
        Jacobian matrix will be copied before scaling. Default False.

    Returns
    -------
    parameters : 1d-array
        The estimated 1D array of parameters that fit the data.

    """
    if jacobian.shape[0] < jacobian.shape[1]:
        warn(
            "Under-determined problem detected (ndata, nparams)={}.".format(
                jacobian.shape
            )
        )
    std = np.std(jacobian, axis=0)
    std[std < 10 * np.finfo(std.dtype).eps] = 1
    data = data.ravel()
    nparams = jacobian.shape[1]
    if copy_jacobian:
        jacobian = jacobian / std
    else:
        jacobian /= std
    if weights is not None:
        weights = np.sqrt(weights)
        data = data * weights
        if copy_jacobian:
            jacobian = (jacobian.T * weights).T
        else:
            transpose = jacobian.T
            transpose *= weights
    hessian = jacobian.T @ jacobian
    gradient = jacobian.T @ data
    if damping is None:
        params = sp.linalg.lstsq(hessian, gradient, overwrite_a=True, overwrite_b=True)[
            0
        ]
    else:
        hessian.flat[:: nparams + 1] += damping
        params = sp.linalg.solve(
            hessian,
            gradient,
            overwrite_a=True,
            overwrite_b=True,
            assume_a="positive definite",
        )
    params = params / std
    return params
