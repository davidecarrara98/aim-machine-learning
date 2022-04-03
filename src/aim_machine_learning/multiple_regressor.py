from aim_machine_learning.base_regressor import Regressor

class MultipleRegressor(Regressor):
    def __init__(self, a, b, **params):
        super().__init__(**params)
        self.a, self.b = a, b

    def fit(self, X, y):
        pass

    def predict(self, X):
        y = self.b + self.a * X
        y = y.squeeze()
        print(y.shape)
        return y