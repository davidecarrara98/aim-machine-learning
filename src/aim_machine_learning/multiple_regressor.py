from aim_machine_learning.base_regressor import Regressor
import numpy as np

class MultipleRegressor(Regressor):
    def __init__(self, a, b, **params):
        super().__init__(**params)
        self.a, self.b = a, b

    def fit(self, X, y):
        pass

    def predict(self, X):
        y = self.b + np.dot(X, self.a)
        y = y.squeeze()
        return y

    def __add__(self, other):
        return MultipleRegressor(a=[[self.a], [other.a]], b=self.b + other.b)
