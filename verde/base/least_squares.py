"""
Functions for least-squares fitting with optional regularization.
"""
from warnings import warn

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge


def least_squares(jacobian, data, weights, damping=None):
    """
    Solve a weighted least-squares problem with optional damping regularization

    Scales the Jacobian matrix so that each column has unit variance. This
    helps keep the regularization parameter in a sensible range. The scaling is
    undone before returning the estimated parameters so that scaling isn't
    required for predictions. Doesn't normalize the column means because that
    operation can't be undone.

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
    scaler = StandardScaler(copy=False, with_mean=False, with_std=True)
    jacobian = scaler.fit_transform(jacobian)
    if damping is None:
        regr = LinearRegression(fit_intercept=False, normalize=False)
    else:
        regr = Ridge(alpha=damping, fit_intercept=False, normalize=False)
    regr.fit(jacobian, data.ravel(), sample_weight=weights)
    params = regr.coef_ / scaler.scale_
    return params
