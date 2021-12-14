# Copyright (c) 2017 The Verde Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Functions for automating model selection through cross-validation.
"""
import warnings

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.utils import check_random_state

from .base import BaseBlockCrossValidator, check_fit_input, n_1d_arrays
from .base.utils import score_estimator
from .coordinates import block_split
from .utils import dispatch, partition_by_sum


class BlockShuffleSplit(BaseBlockCrossValidator):
    """
    Random permutation of spatial blocks cross-validator.

    Yields indices to split data into training and test sets. Data are first
    grouped into rectangular blocks of size given by the *spacing* argument.
    Alternatively, blocks can be defined by the number of blocks in each
    dimension using the *shape* argument instead of *spacing*. The blocks are
    then split into testing and training sets randomly.

    The proportion of blocks assigned to each set is controlled by *test_size*
    and/or *train_size*. However, the total amount of actual data points in
    each set could be different from these values since blocks can have
    a different number of data points inside them. To guarantee that the
    proportion of actual data is as close as possible to the proportion of
    blocks, this cross-validator generates an extra number of splits and
    selects the one with proportion of data points in each set closer to the
    desired amount [Valavi_etal2019]_. The number of balancing splits per
    iteration is controlled by the *balancing* argument.

    This cross-validator is preferred over
    :class:`sklearn.model_selection.ShuffleSplit` for spatial data to avoid
    overestimating cross-validation scores. This can happen because of the
    inherent autocorrelation that is usually associated with this type of data
    (points that are close together are more likely to have similar values).
    See [Roberts_etal2017]_ for an overview of this topic.

    .. note::

        Like :class:`sklearn.model_selection.ShuffleSplit`, this
        cross-validator cannot guarantee that all folds will be different,
        although this is still very likely for sizeable datasets.

    Parameters
    ----------
    spacing : float, tuple = (s_north, s_east), or None
        The block size in the South-North and West-East directions,
        respectively. A single value means that the spacing is equal in both
        directions. If None, then *shape* **must be provided**.
    shape : tuple = (n_north, n_east) or None
        The number of blocks in the South-North and West-East directions,
        respectively. If None, then *spacing* **must be provided**.
    n_splits : int, default 10
        Number of re-shuffling & splitting iterations.
    test_size : float, int, None, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If ``train_size`` is also None, it will
        be set to 0.1.
    train_size : float, int, or None, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    balancing : int
        The number of splits generated per iteration to try to balance the
        amount of data in each set so that *test_size* and *train_size* are
        respected. If 1, then no extra splits are generated (essentially
        disabling the balancing). Must be >= 1.

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
    Train: [ 2  3  6  7  8  9 10 11 12 13 14 15] Test: [0 1 4 5]
    Train: [ 0  1  4  5  8  9 10 11 12 13 14 15] Test: [2 3 6 7]
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
    [[' test' ' test' 'train' 'train']
     [' test' ' test' 'train' 'train']
     ['train' 'train' 'train' 'train']
     ['train' 'train' 'train' 'train']]
    Iteration 2:
    [['train' 'train' ' test' ' test']
     ['train' 'train' ' test' ' test']
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
        balancing=10,
    ):
        super().__init__(spacing=spacing, shape=shape, n_splits=n_splits)
        if balancing < 1:
            raise ValueError(
                "The *balancing* argument must be >= 1. To disable balancing, use 1."
            )
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
        self.balancing = balancing

    def _iter_test_indices(self, X=None, y=None, groups=None):  # noqa: N803,U100
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
            n_splits=self.n_splits * self.balancing,
            test_size=self.test_size,
            train_size=self.train_size,
            random_state=self.random_state,
        ).split(block_ids)
        for _ in range(self.n_splits):
            test_sets, balance = [], []
            for _ in range(self.balancing):
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


class BlockKFold(BaseBlockCrossValidator):
    """
    K-Folds over spatial blocks cross-validator.

    Yields indices to split data into training and test sets. Data are first
    grouped into rectangular blocks of size given by the *spacing* argument.
    Alternatively, blocks can be defined by the number of blocks in each
    dimension using the *shape* argument instead of *spacing*. The blocks are
    then split into testing and training sets iteratively along k folds of the
    data (k is given by *n_splits*).

    By default, the blocks are split into folds in a way that makes each fold
    have approximately the same number of data points. Sometimes this might not
    be possible, which can happen if the number of splits is close to the
    number of blocks. In these cases, each fold will have the same number of
    blocks, regardless of how many data points are in each block. This
    behaviour can also be disabled by setting ``balance=False``.

    Shuffling the blocks prior to splitting is strongly encouraged. Not
    shuffling will essentially lead to the creation of *n_splits* large blocks
    since blocks are spatially adjacent when not shuffled. The default
    behaviour is not to shuffle for compatibility with similar cross-validators
    in scikit-learn.

    This cross-validator is preferred over
    :class:`sklearn.model_selection.KFold` for spatial data to avoid
    overestimating cross-validation scores. This can happen because of the
    inherent autocorrelation that is usually associated with this type of data
    (points that are close together are more likely to have similar values).
    See [Roberts_etal2017]_ for an overview of this topic.

    Parameters
    ----------
    spacing : float, tuple = (s_north, s_east), or None
        The block size in the South-North and West-East directions,
        respectively. A single value means that the spacing is equal in both
        directions. If None, then *shape* **must be provided**.
    shape : tuple = (n_north, n_east) or None
        The number of blocks in the South-North and West-East directions,
        respectively. If None, then *spacing* **must be provided**.
    n_splits : int
        Number of folds. Must be at least 2.
    shuffle : bool
        Whether to shuffle the data before splitting into batches.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    balance : bool
        Whether or not to split blocks into fold with approximately equal
        number of data points. If False, each fold will have the same number of
        blocks (which can have different number of data points in them).

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
    >>> kfold = BlockKFold(spacing=1.5, n_splits=4)
    >>> # These are the 1D indices of the points belonging to each set
    >>> for train, test in kfold.split(X):
    ...     print("Train: {} Test: {}".format(train, test))
    Train: [ 2  3  6  7  8  9 10 11 12 13 14 15] Test: [0 1 4 5]
    Train: [ 0  1  4  5  8  9 10 11 12 13 14 15] Test: [2 3 6 7]
    Train: [ 0  1  2  3  4  5  6  7 10 11 14 15] Test: [ 8  9 12 13]
    Train: [ 0  1  2  3  4  5  6  7  8  9 12 13] Test: [10 11 14 15]
    >>> # A better way to visualize this is to create a 2D array and put
    >>> # "train" or "test" in the corresponding locations.
    >>> shape = coords[0].shape
    >>> mask = np.full(shape=shape, fill_value="     ")
    >>> for iteration, (train, test) in enumerate(kfold.split(X)):
    ...     # The index needs to be converted to 2D so we can index our matrix.
    ...     mask[np.unravel_index(train, shape)] = "train"
    ...     mask[np.unravel_index(test, shape)] = " test"
    ...     print("Iteration {}:".format(iteration))
    ...     print(mask)
    Iteration 0:
    [[' test' ' test' 'train' 'train']
     [' test' ' test' 'train' 'train']
     ['train' 'train' 'train' 'train']
     ['train' 'train' 'train' 'train']]
    Iteration 1:
    [['train' 'train' ' test' ' test']
     ['train' 'train' ' test' ' test']
     ['train' 'train' 'train' 'train']
     ['train' 'train' 'train' 'train']]
    Iteration 2:
    [['train' 'train' 'train' 'train']
     ['train' 'train' 'train' 'train']
     [' test' ' test' 'train' 'train']
     [' test' ' test' 'train' 'train']]
    Iteration 3:
    [['train' 'train' 'train' 'train']
     ['train' 'train' 'train' 'train']
     ['train' 'train' ' test' ' test']
     ['train' 'train' ' test' ' test']]

    For spatial data, it's often good to shuffle the blocks before assigning
    them to folds:

    >>> # Set the random_state to make sure we always get the same result.
    >>> kfold = BlockKFold(
    ...     spacing=1.5, n_splits=4, shuffle=True, random_state=123,
    ... )
    >>> for train, test in kfold.split(X):
    ...     print("Train: {} Test: {}".format(train, test))
    Train: [ 0  1  2  3  4  5  6  7  8  9 12 13] Test: [10 11 14 15]
    Train: [ 2  3  6  7  8  9 10 11 12 13 14 15] Test: [0 1 4 5]
    Train: [ 0  1  4  5  8  9 10 11 12 13 14 15] Test: [2 3 6 7]
    Train: [ 0  1  2  3  4  5  6  7 10 11 14 15] Test: [ 8  9 12 13]

    These should be the same splits as we got before but in a different order.
    This only happens because in this example we have the number of splits
    equal to the number of blocks (4). With real data the effects would be more
    dramatic.

    """

    def __init__(
        self,
        spacing=None,
        shape=None,
        n_splits=5,
        shuffle=False,
        random_state=None,
        balance=True,
    ):
        super().__init__(spacing=spacing, shape=shape, n_splits=n_splits)
        if n_splits < 2:
            raise ValueError(
                "Number of splits must be >=2 for BlockKFold. Given {}.".format(
                    n_splits
                )
            )
        self.shuffle = shuffle
        self.random_state = random_state
        self.balance = balance

    def _iter_test_indices(self, X=None, y=None, groups=None):  # noqa: N803,U100
        """
        Generates integer indices corresponding to test sets.

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
        if self.n_splits > block_ids.size:
            raise ValueError(
                "Number of k-fold splits ({}) cannot be greater than the number of "
                "blocks ({}). Either decrease n_splits or increase the number of "
                "blocks.".format(self.n_splits, block_ids.size)
            )
        if self.shuffle:
            check_random_state(self.random_state).shuffle(block_ids)
        if self.balance:
            block_sizes = [np.isin(labels, i).sum() for i in block_ids]
            try:
                split_points = partition_by_sum(block_sizes, parts=self.n_splits)
                folds = np.split(np.arange(block_ids.size), split_points)
            except ValueError:
                warnings.warn(
                    "Could not balance folds to have approximately the same "
                    "number of data points. Dividing into folds with equal "
                    "number of blocks instead. Decreasing n_splits or increasing "
                    "the number of blocks may help.",
                    UserWarning,
                )
                folds = [i for _, i in KFold(n_splits=self.n_splits).split(block_ids)]
        else:
            folds = [i for _, i in KFold(n_splits=self.n_splits).split(block_ids)]
        for test_blocks in folds:
            test_points = np.where(np.isin(labels, block_ids[test_blocks]))[0]
            yield test_points


def train_test_split(
    coordinates, data, weights=None, spacing=None, shape=None, **kwargs
):
    r"""
    Split a dataset into a training and a testing set for cross-validation.

    Similar to :func:`sklearn.model_selection.train_test_split` but is tuned to
    work on single- or multi-component spatial data with optional weights.

    If arguments *shape* or *spacing* are provided, will group the data by
    spatial blocks before random splitting (using
    :class:`verde.BlockShuffleSplit` instead of
    :class:`sklearn.model_selection.ShuffleSplit`). The argument *spacing*
    specifies the size of the spatial blocks. Alternatively, use *shape* to
    specify the number of blocks in each dimension.

    Extra keyword arguments will be passed to the cross-validation class. The
    exception is ``n_splits`` which is always 1.

    Grouping by spatial blocks is preferred over plain random splits for
    spatial data to avoid overestimating validation scores. This can happen
    because of the inherent autocorrelation that is usually associated with
    this type of data (points that are close together are more likely to have
    similar values). See [Roberts_etal2017]_ for an overview of this topic. To
    use spatial blocking, you **must provide** a *spacing* or *shape* argument
    (see below).

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
    spacing : float, tuple = (s_north, s_east), or None
        The spatial block size in the South-North and West-East directions,
        respectively. A single value means that the spacing is equal in both
        directions. If None, then *shape* must be provided in order to use
        spatial blocking.
    shape : tuple = (n_north, n_east) or None
        The number of blocks in the South-North and West-East directions,
        respectively. If None, then *spacing* must be provided in order to use
        spatial blocking.

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

    To randomly split the data between training and testing sets:

    >>> import numpy as np
    >>> # Make some data
    >>> data = np.array([10, 30, 50, 70])
    >>> coordinates = (np.arange(4), np.arange(-4, 0))
    >>> train, test = train_test_split(coordinates, data, random_state=0)
    >>> # The training set:
    >>> print("coords:", train[0])
    coords: (array([3, 1, 0]), array([-1, -3, -4]))
    >>> print("data:", train[1])
    data: (array([70, 30, 10]),)
    >>> # The testing set:
    >>> print("coords:", test[0])
    coords: (array([2]), array([-2]))
    >>> print("data:", test[1])
    data: (array([50]),)

    If weights are given, they will also be split among the sets:

    >>> weights = np.array([4, 3, 2, 5])
    >>> train, test = train_test_split(
    ...     coordinates, data, weights, random_state=0,
    ... )
    >>> # The training set:
    >>> print("coords:", train[0])
    coords: (array([3, 1, 0]), array([-1, -3, -4]))
    >>> print("data:", train[1])
    data: (array([70, 30, 10]),)
    >>> print("weights:", train[2])
    weights: (array([5, 3, 4]),)
    >>> # The testing set:
    >>> print("coords:", test[0])
    coords: (array([2]), array([-2]))
    >>> print("data:", test[1])
    data: (array([50]),)
    >>> print("weights:", test[2])
    weights: (array([2]),)

    Data with multiple components can also be split:

    >>> data = (np.array([10, 30, 50, 70]), np.array([-70, -50, -30, -10]))
    >>> train, test = train_test_split(coordinates, data, random_state=0)
    >>> # The training set:
    >>> print("coords:", train[0])
    coords: (array([3, 1, 0]), array([-1, -3, -4]))
    >>> print("data:", train[1])
    data: (array([70, 30, 10]), array([-10, -50, -70]))
    >>> # The testing set:
    >>> print("coords:", test[0])
    coords: (array([2]), array([-2]))
    >>> print("data:", test[1])
    data: (array([50]), array([-30]))

    To split data grouped in spatial blocks:

    >>> from verde import grid_coordinates
    >>> # Make a regular grid of data points
    >>> coordinates = grid_coordinates(region=(0, 3, 4, 7), spacing=1)
    >>> data = np.arange(16).reshape((4, 4))
    >>> # We must specify the size of the blocks via the spacing argument.
    >>> # Blocks of 1.5 will split the domain into 4 blocks.
    >>> train, test = train_test_split(
    ...     coordinates, data, random_state=0, spacing=1.5,
    ... )
    >>> # The training set:
    >>> print("coords:", train[0][0], train[0][1], sep="\n")
    coords:
    [0. 1. 2. 3. 0. 1. 2. 3. 2. 3. 2. 3.]
    [4. 4. 4. 4. 5. 5. 5. 5. 6. 6. 7. 7.]
    >>> print("data:", train[1])
    data: (array([ 0,  1,  2,  3,  4,  5,  6,  7, 10, 11, 14, 15]),)
    >>> # The testing set:
    >>> print("coords:", test[0][0], test[0][1])
    coords: [0. 1. 0. 1.] [6. 6. 7. 7.]
    >>> print("data:", test[1])
    data: (array([ 8,  9, 12, 13]),)

    """
    args = check_fit_input(coordinates, data, weights, unpack=False)
    if spacing is None and shape is None:
        indices = np.arange(args[1][0].size)
        shuffle = ShuffleSplit(n_splits=1, **kwargs).split(indices)
    else:
        feature_matrix = np.transpose(n_1d_arrays(coordinates, 2))
        shuffle = BlockShuffleSplit(
            n_splits=1, spacing=spacing, shape=shape, **kwargs
        ).split(feature_matrix)
    split = next(shuffle)
    train, test = (tuple(select(i, index) for i in args) for index in split)
    return train, test


def cross_val_score(
    estimator,
    coordinates,
    data,
    weights=None,
    cv=None,
    client=None,
    delayed=False,
    scoring=None,
):
    """
    Score an estimator/gridder using cross-validation.

    Similar to :func:`sklearn.model_selection.cross_val_score` but modified to
    accept spatial multi-component data with weights.

    By default, will use :class:`sklearn.model_selection.KFold` with
    ``n_splits=5`` and ``random_state=0`` to split the dataset. Any other
    cross-validation class from scikit-learn or Verde can be passed in through
    the *cv* argument.

    The score is calculated using the estimator/gridder's ``.score`` method by
    default. Alternatively, use the *scoring* parameter to specify a different
    scoring function (e.g., mean square error, mean absolute error, etc).

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
        Any scikit-learn or Verde cross-validation generator. Defaults to
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
    scoring : None, str, or callable
        A scoring function (or name of a function) known to scikit-learn. See
        the description of *scoring* in
        :func:`sklearn.model_selection.cross_val_score` for details. If None,
        will fall back to the estimator's ``.score`` method.

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

    To calculate the score with a different metric, use the *scoring* argument:

    >>> scores = cross_val_score(
    ...     model, coords, data, scoring="neg_mean_squared_error",
    ... )
    >>> print(', '.join(['{:.2f}'.format(-score) for score in scores]))
    0.00, 0.00, 0.00, 0.00, 0.00

    In this case, we calculated the (negative) mean squared error (MSE) which
    measures the distance between test data and predictions. This way, 0 is the
    best possible value meaning that the data and prediction are the same. The
    "neg" part indicates that this is the negative mean square error. This is
    required because scikit-learn assumes that higher scores are always treated
    as better (which is the opposite for MSE). For display, we take the
    negative of the score to get the actual MSE.

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
            FutureWarning,
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
        # Clone the estimator to avoid fitting the same object simultaneously
        # when delayed=True.
        score = dispatch(fit_score, client=client, delayed=delayed)(
            clone(estimator),
            tuple(select(i, train_index) for i in fit_args),
            tuple(select(i, test_index) for i in fit_args),
            scoring,
        )
        scores.append(score)
    if not delayed and client is None:
        scores = np.asarray(scores)
    return scores


def fit_score(estimator, train_data, test_data, scoring):
    """
    Fit an estimator on the training data and then score it on the testing data
    """
    estimator.fit(*train_data)
    if scoring is None:
        score = estimator.score(*test_data)
    else:
        score = score_estimator(scoring, estimator, *test_data)
    return score


def select(arrays, index):
    """
    Index each array in a tuple of arrays.

    If the arrays tuple contains a ``None``, the entire tuple will be returned
    as is.

    Parameters
    ----------
    arrays : tuple of arrays
        The arrays to index
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
