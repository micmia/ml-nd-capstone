from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from . import preprocessing, cv
import xgboost as xgb


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
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, shuffle=False)
            dtrain = xgb.DMatrix(X_train, label=preprocessing.transform_y(y_train))
            dvalid = xgb.DMatrix(X_test, label=preprocessing.transform_y(y_test))
            watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
            self.bst = xgb.train(self.params, dtrain, self.kwargs['num_boost_round'], evals=watchlist,
                                 feval=cv.rmspe_xgb, early_stopping_rounds=self.kwargs['early_stopping_rounds'],
                                 verbose_eval=True)

    def predict(self, X):
        if self.use_sklearn:
            y_pred = self.bst.predict(X)
        else:
            y_pred = preprocessing.restore_y(self.bst.predict(xgb.DMatrix(X)))

        return y_pred

    def save_model(self, filename):
        joblib.dump(self.bst, filename)

    def load_model(self, filename):
        self.bst = joblib.load(filename)
