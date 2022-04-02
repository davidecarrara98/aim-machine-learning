import numpy as np


class CustomError(Exception):
    pass


class Evaluator:
    def __init__(self, supported_metrics):
        self.supported_metrics = supported_metrics
        self.metric = None

    def set_metric(self, new_metric):
        if new_metric not in self.supported_metrics:
            raise NameError("Metric not supported")
        else:
            self.metric = new_metric
        return self

    def __call__(self, y_true, y_pred, *args, **kwargs):
        if self.metric is None:
            raise CustomError('You did not specify a metric')
        if self.metric == 'mse':
            mean = np.power(y_true - y_pred, 2).mean()
            std  = np.std(y_true - y_pred)
            return {'mean': mean, 'std':std}
        if self.metric == 'mae':
            mean = np.abs(y_true - y_pred).mean()
            std  = np.std(y_true - y_pred)
            return {'mean': mean, 'std':std}
        if self.metric == 'corr':
            corr = np.corrcoef(y_true, y_pred)[0, 1]
            return {'corr': corr}
