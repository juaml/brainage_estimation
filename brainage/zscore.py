import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from scipy.stats import zscore


class ZScore(BaseEstimator, TransformerMixin):

    def __init__(self, axis=0):
        self.axis = axis

    def fit(self, X, y=None):
        X = check_array(X)
        self.mean_ = np.mean(X, axis=self.axis)
        self.std_ = np.std(X, axis=self.axis)
        return self

    def transform(self, X):
        X = check_array(X)
        mean = (
            self.mean_.reshape(-1, 1)
            if self.axis
            else self.mean_
        )

        std = (
            self.std_.reshape(-1, 1)
            if self.axis
            else self.std_
        )
        # print(f"{X.shape = }")
        # print(f"{mean.shape = }")
        # print(f"{std.shape = }")

        return (X - mean) / std


class ZScoreSubwise(BaseEstimator, TransformerMixin):

    def __init__(self, axis=0):
        self.axis = axis

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = check_array(X)
        return zscore(X, axis=self.axis)

