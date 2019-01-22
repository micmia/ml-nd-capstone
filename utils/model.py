from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from . import preprocessing, cv
import xgboost as xgb
import numpy as np


class Model:
    def __init__(self, params=None, **kwargs):
        self.params = params
        self.kwargs = kwargs
        self.use_sklearn = kwargs['use_sklearn'] if 'use_sklearn' in kwargs else False

        if params:
            if self.use_sklearn:
                self.bst = xgb.XGBRegressor(**self.params)

    def train(self, X, y):
        if self.use_sklearn:
            self.bst.fit(X, y)
        else:
            X_train, X_test, y_train, y_test = X[41088:], X[:41088], y[41088:], y[:41088]
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dvalid = xgb.DMatrix(X_test, label=y_test)
            watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
            self.bst = xgb.train(self.params, dtrain, self.kwargs['num_boost_round'], evals=watchlist,
                                 feval=cv.rmspe_xgb, early_stopping_rounds=self.kwargs['early_stopping_rounds'],
                                 verbose_eval=True)

    def predict(self, X, weight=0.995):
        if self.use_sklearn:
            y_pred = self.bst.predict(weight * X)
        else:
            y_pred = np.expm1(weight * self.bst.predict(xgb.DMatrix(X)))

        return y_pred

    def save_model(self, filename):
        joblib.dump(self.bst, filename)

    def load_model(self, filename):
        self.bst = joblib.load(filename)
