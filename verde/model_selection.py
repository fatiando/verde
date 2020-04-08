# pylint: disable=stop-iteration-return
"""
Functions for automating model selection through cross-validation.

Supports using a dask.distributed.Client object for parallelism. The
DummyClient is used as a serial version of the parallel client.
"""
import warnings

import numpy as np
from sklearn.model_selection import KFold, ShuffleSplit, BaseCrossValidator
from sklearn.base import clone

from .base import check_fit_input, n_1d_arrays
from .coordinates import block_split
from .utils import dispatch


# Otherwise, DeprecationWarning won't be shown, kind of defeating the purpose.
warnings.simplefilter("default")


class BlockShuffleSplit(BaseCrossValidator):
    """
    Random permutation of spatial blocks cross-validator.


    Parameters
    ----------

    See also
    --------
    train_test_split : Split a dataset into a training and a testing set.
    cross_val_score : Score an estimator/gridder using cross-validation.

    Examples
    --------

    >>> from verde import grid_coordinates
    >>> import numpy as np
    >>> # Make a regular grid of data points
    >>> coords = grid_coordinates(region=(0, 3, -10, -7), spacing=1)
    >>> # Need to convert the coordinates into a feature matrix
    >>> X = np.transpose([i.ravel() for i in coords])
    >>> shuffle = BlockShuffleSplit(spacing=1.5, n_splits=3, random_state=0)
    >>> # These are the 1D indices of the points belonging to each set
    >>> for train, test in shuffle.split(X):
    ...     print("Train: {} Test: {}".format(train, test))
    Train: [ 0  1  2  3  4  5  6  7 10 11 14 15] Test: [ 8  9 12 13]
    Train: [ 0  1  4  5  8  9 10 11 12 13 14 15] Test: [2 3 6 7]
    Train: [ 2  3  6  7  8  9 10 11 12 13 14 15] Test: [0 1 4 5]
    >>> # A better way to visualize this is to create a 2D array and put
    >>> # "train" or "test" in the corresponding locations.
    >>> shape = coords[0].shape
    >>> mask = np.full(shape=shape, fill_value="     ")
    >>> for iteration, (train, test) in enumerate(shuffle.split(X)):
    ...     # The index needs to be converted to 2D so we can index our matrix.
    ...     mask[np.unravel_index(train, shape)] = "train"
    ...     mask[np.unravel_index(test, shape)] = " test"
    ...     print("Iteration {}:".format(iteration))
    ...     print(mask)
    Iteration 0:
    [['train' 'train' 'train' 'train']
     ['train' 'train' 'train' 'train']
     [' test' ' test' 'train' 'train']
     [' test' ' test' 'train' 'train']]
    Iteration 1:
    [['train' 'train' ' test' ' test']
     ['train' 'train' ' test' ' test']
     ['train' 'train' 'train' 'train']
     ['train' 'train' 'train' 'train']]
    Iteration 2:
    [[' test' ' test' 'train' 'train']
     [' test' ' test' 'train' 'train']
     ['train' 'train' 'train' 'train']
     ['train' 'train' 'train' 'train']]


    """

    def __init__(
        self,
        spacing=None,
        shape=None,
        n_splits=10,
        test_size=0.1,
        train_size=None,
        random_state=None,
        balancing_iterations=50,
    ):
        if spacing is None and shape is None:
            raise ValueError("Either 'spacing' or 'shape' must be provided.")
        self.spacing = spacing
        self.shape = shape
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
        self.balancing_iterations = balancing_iterations

    def _iter_test_indices(self, X=None, y=None, groups=None):
        """
        Generates integer indices corresponding to test sets.

        Runs several iterations until a split is found that yields blocks with
        the right amount of data points in it.

        Parameters
        ----------
        X : array-like, shape (n_samples, 2)
            Columns should be the easting and northing coordinates of data
            points, respectively.
        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems. Always
            ignored.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set. Always ignored.

        Yields
        ------
        test : ndarray
            The testing set indices for that split.

        """
        labels = block_split(
            coordinates=(X[:, 0], X[:, 1]),
            spacing=self.spacing,
            shape=self.shape,
            region=None,
            adjust="spacing",
        )[1]
        block_ids = np.unique(labels)
        # Generate many more splits so that we can pick and choose the ones
        # that have the right balance of training and testing data.
        shuffle = ShuffleSplit(
            n_splits=self.n_splits * self.balancing_iterations,
            test_size=self.test_size,
            train_size=self.train_size,
            random_state=self.random_state,
        ).split(block_ids)
        for _ in range(self.n_splits):
            test_sets, balance = [], []
            for _ in range(self.balancing_iterations):
                # This is a false positive in pylint which is why the warning
                # is disabled at the top of this file:
                # https://github.com/PyCQA/pylint/issues/1830
                train_blocks, test_blocks = next(shuffle)
                train_points = np.where(np.isin(labels, block_ids[train_blocks]))[0]
                test_points = np.where(np.isin(labels, block_ids[test_blocks]))[0]
                # The proportion of data points assigned to each group should
                # be close the proportion of blocks assigned to each group.
                balance.append(
                    abs(
                        train_points.size / test_points.size
                        - train_blocks.size / test_blocks.size
                    )
                )
                test_sets.append(test_points)
            best = np.argmin(balance)
            yield test_sets[best]

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, 2)
            Columns should be the easting and northing coordinates of data
            points, respectively.
        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems. Always
            ignored.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set. Always ignored.

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.

        """
        if X.shape[1] != 2:
            raise ValueError(
                "X must have exactly 2 columns ({} given).".format(X.shape[1])
            )
        for train, test in super().split(X, y, groups):
            yield train, test

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Returns the number of splitting iterations in the cross-validator

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
        y : object
            Always ignored, exists for compatibility.
        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits


def train_test_split(coordinates, data, weights=None, **kwargs):
    r"""
    Split a dataset into a training and a testing set for cross-validation.

    Similar to :func:`sklearn.model_selection.train_test_split` but is tuned to
    work on multi-component spatial data with optional weights.

    Extra keyword arguments will be passed to
    :class:`sklearn.model_selection.ShuffleSplit`, except for ``n_splits``
    which is always 1.

    Parameters
    ----------
    coordinates : tuple of arrays
        Arrays with the coordinates of each data point. Should be in the
        following order: (easting, northing, vertical, ...).
    data : array or tuple of arrays
        the data values of each data point. If the data has more than one
        component, *data* must be a tuple of arrays (one for each component).
    weights : none or array or tuple of arrays
        if not none, then the weights assigned to each data point. If more than
        one data component is provided, you must provide a weights array for
        each data component (if not none).

    Returns
    -------
    train, test : tuples
        Each is a tuple = (coordinates, data, weights) generated by separating
        the input values randomly.

    See also
    --------
    cross_val_score : Score an estimator/gridder using cross-validation.
    BlockShuffleSplit : Random permutation of spatial blocks cross-validator.

    Examples
    --------

    >>> import numpy as np
    >>> # Split 2-component data with weights
    >>> data = (np.array([1, 3, 5, 7]), np.array([0, 2, 4, 6]))
    >>> coordinates = (np.arange(4), np.arange(-4, 0))
    >>> weights = (np.array([1, 1, 2, 1]), np.array([1, 2, 1, 1]))
    >>> train, test = train_test_split(coordinates, data, weights,
    ...                                random_state=0)
    >>> print("Coordinates:", train[0], test[0], sep='\n  ')
    Coordinates:
      (array([3, 1, 0]), array([-1, -3, -4]))
      (array([2]), array([-2]))
    >>> print("Data:", train[1], test[1], sep='\n  ')
    Data:
      (array([7, 3, 1]), array([6, 2, 0]))
      (array([5]), array([4]))
    >>> print("Weights:", train[2], test[2], sep='\n  ')
    Weights:
      (array([1, 1, 1]), array([1, 2, 1]))
      (array([2]), array([1]))
    >>> # Split single component data without weights
    >>> train, test = train_test_split(coordinates, data[0], None,
    ...                                random_state=0)
    >>> print("Coordinates:", train[0], test[0], sep='\n  ')
    Coordinates:
      (array([3, 1, 0]), array([-1, -3, -4]))
      (array([2]), array([-2]))
    >>> print("Data:", train[1], test[1], sep='\n  ')
    Data:
      (array([7, 3, 1]),)
      (array([5]),)
    >>> print("Weights:", train[2], test[2], sep='\n  ')
    Weights:
      (None,)
      (None,)

    """
    args = check_fit_input(coordinates, data, weights, unpack=False)
    ndata = args[1][0].size
    indices = np.arange(ndata)
    split = next(ShuffleSplit(n_splits=1, **kwargs).split(indices))
    train, test = (tuple(select(i, index) for i in args) for index in split)
    return train, test


def cross_val_score(
    estimator, coordinates, data, weights=None, cv=None, client=None, delayed=False
):
    """
    Score an estimator/gridder using cross-validation.

    Similar to :func:`sklearn.model_selection.cross_val_score` but modified to
    accept spatial multi-component data with weights.

    By default, will use :class:`sklearn.model_selection.KFold` with
    ``n_splits=5`` and ``random_state=0`` to split the dataset. Any other
    cross-validation class can be passed in through the *cv* argument.

    Can optionally run in parallel using :mod:`dask`. To do this, use
    ``delayed=True`` to dispatch computations with :func:`dask.delayed` instead
    of running them. The returned scores will be "lazy" objects instead of the
    actual scores. To trigger the computation (which Dask will run in parallel)
    call the `.compute()` method of each score or :func:`dask.compute` with the
    entire list of scores.

    .. warning::

        The ``client`` parameter is deprecated and will be removed in Verde
        v2.0.0. Use ``delayed`` instead.

    Parameters
    ----------
    estimator : verde gridder
        Any verde gridder class that has the ``fit`` and ``score`` methods.
    coordinates : tuple of arrays
        Arrays with the coordinates of each data point. Should be in the
        following order: (easting, northing, vertical, ...).
    data : array or tuple of arrays
        the data values of each data point. If the data has more than one
        component, *data* must be a tuple of arrays (one for each component).
    weights : none or array or tuple of arrays
        if not none, then the weights assigned to each data point. If more than
        one data component is provided, you must provide a weights array for
        each data component (if not none).
    cv : None or cross-validation generator
        Any scikit-learn cross-validation generator. Defaults to
        :class:`sklearn.model_selection.KFold`.
    client : None or dask.distributed.Client
        **DEPRECATED:** This option is deprecated and will be removed in Verde
        v2.0.0. If None, then computations are run serially. Otherwise, should
        be a dask ``Client`` object. It will be used to dispatch computations
        to the dask cluster.
    delayed : bool
        If True, will use :func:`dask.delayed` to dispatch computations without
        actually executing them. The returned scores will be a list of delayed
        objects. Call `.compute()` on each score or :func:`dask.compute` on the
        entire list to trigger the actual computations.

    Returns
    -------
    scores : array
        Array of scores for each split of the cross-validation generator. If
        *delayed*, will be a list of Dask delayed objects (see the *delayed*
        option). If *client* is not None, then the scores will be futures.

    See also
    --------
    train_test_split : Split a dataset into a training and a testing set.
    BlockShuffleSplit : Random permutation of spatial blocks cross-validator.

    Examples
    --------

    As an example, we can score :class:`verde.Trend` on data that actually
    follows a linear trend.

    >>> from verde import grid_coordinates, Trend
    >>> coords = grid_coordinates((0, 10, -10, -5), spacing=0.1)
    >>> data = 10 - coords[0] + 0.5*coords[1]
    >>> model = Trend(degree=1)

    In this case, the model should perfectly predict the data and R² scores
    should be equal to 1.

    >>> scores = cross_val_score(model, coords, data)
    >>> print(', '.join(['{:.2f}'.format(score) for score in scores]))
    1.00, 1.00, 1.00, 1.00, 1.00

    There are 5 scores because the default cross-validator is
    :class:`sklearn.model_selection.KFold` with ``n_splits=5``.

    We can use different cross-validators by assigning them to the ``cv``
    argument:

    >>> from sklearn.model_selection import ShuffleSplit
    >>> # Set the random state to get reproducible results
    >>> cross_validator = ShuffleSplit(n_splits=3, random_state=0)
    >>> scores = cross_val_score(model, coords, data, cv=cross_validator)
    >>> print(', '.join(['{:.2f}'.format(score) for score in scores]))
    1.00, 1.00, 1.00

    Often, spatial data are autocorrelated (points that are close together are
    more likely to have similar values), which can cause cross-validation with
    random splits to overestimate the prediction accuracy [Roberts_etal2017]_.
    To account for the autocorrelation, we can split the data in blocks rather
    than randomly with :class:`verde.BlockShuffleSplit`:

    >>> from verde import BlockShuffleSplit
    >>> # spacing controls the size of the spatial blocks
    >>> cross_validator = BlockShuffleSplit(
    ...     spacing=2, n_splits=3, random_state=0
    ... )
    >>> scores = cross_val_score(model, coords, data, cv=cross_validator)
    >>> print(', '.join(['{:.2f}'.format(score) for score in scores]))
    1.00, 1.00, 1.00

    We didn't see a difference here since our model and data are perfect. See
    :ref:`model_evaluation` for an example with real data.

    If using many splits, we can speed up computations by running them in
    parallel with Dask:

    >>> cross_validator = ShuffleSplit(n_splits=10, random_state=0)
    >>> scores_delayed = cross_val_score(
    ...     model, coords, data, cv=cross_validator, delayed=True
    ... )
    >>> # The scores are delayed objects.
    >>> # To actually run the computations, call dask.compute
    >>> import dask
    >>> scores = dask.compute(*scores_delayed)
    >>> print(', '.join(['{:.2f}'.format(score) for score in scores]))
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00

    Note that you must have enough RAM to fit multiple models simultaneously.
    So this is best used when fitting several smaller models.

    """
    if client is not None:
        warnings.warn(
            "The 'client' parameter of 'verde.cross_val_score' is deprecated "
            "and will be removed in Verde 2.0.0. "
            "Use the 'delayed' parameter instead.",
            DeprecationWarning,
        )
    coordinates, data, weights = check_fit_input(
        coordinates, data, weights, unpack=False
    )
    if cv is None:
        cv = KFold(shuffle=True, random_state=0, n_splits=5)
    feature_matrix = np.transpose(n_1d_arrays(coordinates, 2))
    fit_args = (coordinates, data, weights)
    scores = []
    for train_index, test_index in cv.split(feature_matrix):
        train = tuple(select(i, train_index) for i in fit_args)
        test = tuple(select(i, test_index) for i in fit_args)
        # Clone the estimator to avoid fitting the same object simultaneously
        # when delayed=True.
        score = dispatch(fit_score, client=client, delayed=delayed)(
            clone(estimator), train, test
        )
        scores.append(score)
    if not delayed and client is None:
        scores = np.asarray(scores)
    return scores


def fit_score(estimator, train_data, test_data):
    """
    Fit an estimator on the training data and then score it on the testing data
    """
    return estimator.fit(*train_data).score(*test_data)


def select(arrays, index):
    """
    Index each array in a tuple of arrays.

    If the arrays tuple contains a ``None``, the entire tuple will be returned
    as is.

    Parameters
    ----------
    arrays : tuple of arrays
    index : array
        An array of indices to select from arrays.

    Returns
    -------
    indexed_arrays : tuple of arrays

    Examples
    --------

    >>> import numpy as np
    >>> select((np.arange(5), np.arange(-3, 2, 1)), [1, 3])
    (array([1, 3]), array([-2,  0]))
    >>> select((None, None, None, None), [1, 2])
    (None, None, None, None)

    """
    if arrays is None or any(i is None for i in arrays):
        return arrays
    return tuple(i.ravel()[index] for i in arrays)
