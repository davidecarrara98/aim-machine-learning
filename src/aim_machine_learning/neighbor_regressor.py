import numpy as np
from aim_machine_learning.base_regressor import Regressor
from sklearn.neighbors import KNeighborsRegressor
from scipy.spatial.distance import cdist


class NeighborRegressor(Regressor):
    def __init__(self, k=1, **params):
        super().__init__(**params)
        self.X, self.y = None, None
        self.k = k

    def fit(self, X, y):
        self.X, self.y = X, y

        if len(self.X.shape) == 1:
            self.X = self.X[:, np.newaxis]
        return

    def predict(self, X):

        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        dists = cdist(X, self.X)
        indices = np.argsort(dists, axis=1)[:, :self.k]

        return np.mean(self.y[indices], axis=1)

class MySklearnNeighborRegressor(KNeighborsRegressor, Regressor):
    def __init__(self, n_neighbors):
        super().__init__(n_neighbors=n_neighbors)


