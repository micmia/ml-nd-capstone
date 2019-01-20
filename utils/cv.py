import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer


def rmspe(y_true, y_pred):
    err = np.sqrt(np.mean((1 - y_pred / y_true) ** 2))

    return err


def rmspe_xgb(y_pred, y_true):
    y_true = y_true.get_label()
    err = rmspe(y_true, y_pred)

    return 'rmspe', err


class GridCV:
    def __init__(self, X, y, param_grid, n_splits=5, verbose=2):
        self.X = X
        self.y = y
        self.n_splits = n_splits
        self.verbose = verbose
        self.param_grid = param_grid
        self.xgb_estimator = xgb.XGBRegressor(
            learning_rate=0.3,
            n_estimators=300,
            silent=True,
            objective='reg:linear',
            gamma=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=5
        )

    def fit(self):
        grid = GridSearchCV(self.xgb_estimator, self.param_grid, n_jobs=5, cv=self.n_splits,
                            scoring=make_scorer(rmspe, greater_is_better=False),
                            verbose=self.verbose)
        grid.fit(self.X, self.y)

        return grid
