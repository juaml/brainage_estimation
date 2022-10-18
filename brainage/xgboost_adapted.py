from xgboost import XGBRegressor
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
import numpy as np

class XGBoostAdapted(BaseEstimator):

    def __init__(self, early_stopping_rounds=10, eval_metric=None, eval_set_percent=0.2, random_seed=None, n_jobs=1, max_depth=6, n_estimators=50, nthread=1, reg_alpha=0):
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.eval_set_percent = eval_set_percent
        self.random_seed = random_seed
        self.n_jobs = n_jobs
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.nthread = nthread
        self.reg_alpha = reg_alpha

            
    def fit(self, X, y):
        self._xgbregressor = XGBRegressor(n_jobs=self.n_jobs, max_depth=self.max_depth, n_estimators=self.n_estimators, nthread=self.nthread, reg_alpha=self.reg_alpha)

        X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=self.eval_set_percent, random_state=self.random_seed)

        eval_set = [(X_test, y_test)]

        self._xgbregressor.fit(X_train, y_train, early_stopping_rounds=self.early_stopping_rounds, eval_metric=self.eval_metric, eval_set=eval_set)
        
        return self

    def score(self, X, y, sample_weight=None):
        return self._xgbregressor.score(X.values, y.values, sample_weight)

    def predict(self, X):
        return self._xgbregressor.predict(X.values)




        
        





