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
            return np.linalg.norm(y_true - y_pred, 2) / y_true.shape[0]
        if self.metric == 'mae':
            return np.linalg.norm(y_true - y_pred, 1) / y_true.shape[0]
        if self.metric == 'corr':
            return 1 - np.corrcoef(y_true, y_pred)[0, 1]
