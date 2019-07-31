"""
Implements sklearn interface to NBSVM classifier.
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model.base import LinearClassifierMixin
from sklearn.utils.validation import check_X_y
from sklearn.utils.multiclass import unique_labels
from sklearn.svm import LinearSVC

def log_normalize_count_vector(arr):
    """
    Takes count vector and normalizes by L1 norm, then takes log.
    """
    return np.log(arr / np.linalg.norm(arr, 1))


class NBSVM(BaseEstimator, LinearClassifierMixin):
    """
    A NBSVM classifier following the sklearn API.

    Parameters
    ----------
    alpha : float, default=1.
        Smoothing parameter for count vectors.
    beta : float, default=0.25
        Interpolation parameter between NB and SVM.
    C : float, default=1.
        Penalty parameter of the L2 error term for SVM.

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """
    def __init__(self, alpha=1., beta=0.25, C=1.):
        self.alpha = alpha
        self.beta = beta
        self.C = C

    def fit(self, X, y):
        """
        Fit the NBSVM to a dataset.

        Parameters
        ----------
        X : scipy.sparse.csr_matrix, shape (n_samples, n_features)
            The training input samples. Must be a sparse matrix containing
            no negative entries.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(
            X,
            y,
            accept_sparse='csr'
         )

        # Validate all X are non-negative
        if (X.data < 0.).any():
            raise ValueError('All X entries should be non-negative.')

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        # Iterate over single class predictions
        coefficients, intercepts = zip(*[
            self._fit_single_class(X, y == class_)
            for class_ in self.classes_
        ])

        self.coef_ = np.concatenate(coefficients)
        self.intercept_ = np.concatenate(intercepts)

        # Return the classifier
        return self

    def _fit_single_class(self, X, y):
        """
        Fit for a single class.
        """

        # smoothed (by alpha) count vectors
        p = (self.alpha + X[y == 1].sum(axis=0))
        q = (self.alpha + X[y == 0].sum(axis=0))

        r = log_normalize_count_vector(p) - log_normalize_count_vector(q)

        # scale X by log count ratio
        X_ = X.multiply(r)

        # fit svm classifier
        svm = LinearSVC(
            C=self.C
        ).fit(X_, y)

        mean_weight = np.abs(svm.coef_).mean()

        coef = (1 - self.beta) * mean_weight + self.beta * svm.coef_

        # reweight for prediction
        # (so that input X effectively gets elementwise product)
        coef *= r

        return coef, svm.intercept_
