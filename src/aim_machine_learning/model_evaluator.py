import numpy as np


class ModelEvaluator:
    def __init__(self, model_class, X, y, params):
        self.model = model_class(params['k'])
        self.X, self.y = X, y

    def train_test_split_eval(self, eval_obj, test_proportion=0.2):
        """

        :param eval_obj: evaluator
        :param test_proportion: test_proportion
        :return: error computed by eval obj
        """

        # train test split
        test_l = int(self.X.shape[0] * test_proportion)
        X_test, X_train = self.X[:test_l], self.X[test_l:]
        y_test, y_train = self.y[:test_l], self.y[test_l:]

        # fit on training data and evaluate on test data
        self.model.fit(X_train, y_train)
        error = self.model.evaluate(X_test, y_test, eval_obj)

        return error

    def kfold_cv_eval(self, eval_obj, K=5):
        """

        :param eval_obj: evaluator object
        :param K: number of spli for k-fold crossvalidation
        :return: error score
        """
        split_l = int(self.X.shape[0] / K)
        errs = np.zeros(shape=K)
        stds = np.zeros(shape=K)

        for i in range(K):
            # Split in train and test
            test_mask = np.zeros(self.X.shape[0], bool)
            test_mask[i * split_l: (i+1) * split_l] = True
            X_test, y_test = self.X[test_mask], self.y[test_mask]
            X_train, y_train = self.X[np.logical_not(test_mask)], self.y[np.logical_not(test_mask)]

            self.model.fit(X_train, y_train)
            errs[i] = self.model.evaluate(X_test, y_test, eval_obj)['mean']
            stds[i] = self.model.evaluate(X_test, y_test, eval_obj)['std']

        return np.mean(errs)