import numpy as np
import matplotlib.pyplot as plt

from aim_machine_learning.model_evaluator import ModelEvaluator

class ParametersTuner:
    def __init__(self, model_class, X, y, supported_eval_types, output_path=None):
        self.model_class = model_class
        self.X, self.y = X, y
        self.supported_eval_types = supported_eval_types
        self.output_path = output_path

    def product(self, param_list):
        """
        Generator producing at each iteration one combination of parameters
        :param param_list: list of list - values of parameters to be evaluated
        :return: tuple of chosen parameters
        """
        if not param_list:
            yield ()
        else:
            for a in param_list[0]:
                for prod in self.product(param_list[1:]):
                    yield (a,) + prod

    def tune_parameters(self, params, eval_type, eval_obj, fig_name=None, **kwargs):
        """

        :param params: dict of lists of params to tune
        :param eval_type: evaluation type among those supported
        :param eval_obj: object of type evaluator
        :param fig_name: name of figure to be saved
        :param kwargs: parameters need for model evaluator
        :return: dict with best parameter for the original model
        """
        # check the evaluation type
        if eval_type not in self.supported_eval_types:
            raise NameError('Evaluation type not supported')

        keys_list = list(params.keys())

        params_list = list(self.product(list(params.values())))
        errs = np.empty(shape=len(params_list))

        for i,k in enumerate(list(params_list)):

            params_dict = {}
            for ind, el in enumerate(k):
                params_dict[keys_list[ind]] = el

            mod_eval = ModelEvaluator(self.model_class, self.X, self.y, params_dict)

            if eval_type == 'ttsplit':
                err_dict = mod_eval.train_test_split_eval(eval_obj=eval_obj, test_proportion=kwargs['test_proportion'])
                errs[i] = err_dict['mean'] + err_dict['std']

            if eval_type == 'kfold':
                err_dict = mod_eval.kfold_cv_eval(eval_obj=eval_obj, K=kwargs['K'])
                errs[i] = err_dict['mean'] + err_dict['std']

        if fig_name is not None:
            self.plot_error(params_list, errs, fig_name)

        final_dict = {}
        for ind, el in enumerate(params_list[np.argmin(errs)]):
            final_dict[keys_list[ind]] = el

        return final_dict

    def plot_error(self, param_list, errs, fig_name=None):
        """

        :param param_list: list of parameter evaluated
        :param errs: list/array of error
        :param fig_name: name of the saved file
        :return:
        """

        plt.figure()
        plt.plot(param_list, errs)
        plt.title('Parameter Tuning')
        plt.xlabel('param')
        plt.ylabel('error')
        plt.savefig(f'{self.output_path}/{fig_name}')

        return