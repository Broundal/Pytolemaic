import logging

import numpy
from lime.lime_tabular import LimeTabularExplainer
from matplotlib import pyplot as plt
from sklearn.linear_model import ElasticNet

from pytolemaic.utils.dmd import DMD
from pytolemaic.utils.general import GeneralUtils


class ElasticNetWrapper(ElasticNet):

    def fit(self, X, y, check_input=False, sample_weight=None):
        return super(ElasticNetWrapper, self).fit(X, y, check_input)


class LimeReport():
    def __init__(self, kernel_width=3, n_features_to_show=None, tol=1e-2, max_samples=256000):

        self.kernel_width = kernel_width
        self.n_features_to_show = n_features_to_show
        self.predict_function = None
        self.tol = tol
        self.max_samples = max_samples

    def fit(self, dmd_train: DMD, model):
        is_classification = GeneralUtils.is_classification(model)

        self.explainer = LimeTabularExplainer(
            training_data=dmd_train.values,
            mode="classification" if is_classification else "regression",
            training_labels=None,  # ???
            feature_names=dmd_train.feature_names,
            categorical_features=dmd_train.categorical_features,  ###
            categorical_names=dmd_train.categorical_encoding_by_icols,  ###
            kernel_width=self.kernel_width,
            kernel=None,  # default is np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))
            verbose=False,
            class_names=dmd_train.labels,
            feature_selection='auto',  # ??? options are 'forward_selection', 'lasso_path', 'none' or 'auto'.
            discretize_continuous=True,
            discretizer='decile',  # ??? options are 'quartile', 'decile', 'entropy'
            sample_around_instance=True,  # default is False
            random_state=0,
            training_data_stats=None)

        self.model = model
        self.predict_function = self.model.predict_proba if is_classification else self.model.predict
        self.labels = numpy.arange(len(test.labels)) if is_classification else None
        self.n_features_to_show = self.n_features_to_show or dmd_train.n_features

    def explain(self, sample):
        try:
            exp = self._lime_explainer(sample)

            return sorted(exp.as_list(), key=lambda a: a[1], reverse=True)[:self.n_features_to_show]
        except:
            return None

    def plot(self, sample):
        try:
            exp = self._lime_explainer(sample)

            label = self.model.predict(sample.reshape(1, -1))

            if GeneralUtils.is_classification(model):
                exp.as_pyplot_figure(label=int(label))
            else:
                exp.as_pyplot_figure(label=None)
                plt.title("Local explanation for predicted value of %.3g" % label)

            plt.tight_layout()
            plt.draw()
        except ValueError as e:
            logging.exception("Failed to plot Lime for instance\n{}".format(sample))

    def _lime_explainer(self, sample):

        nan_mask = ~numpy.isfinite(sample)
        if any(nan_mask):
            logging.warning("Lime cannot handle missing values. Fillna(0) was used to coerce the issue.")
            sample[nan_mask] = 0

        try:

            as_dict = lambda exp: {k: numpy.round(v, 5) for k, v in exp.as_list()}

            dict_delta = lambda exp_dict1, exp_dict2, features_to_show: {k: abs(exp_dict1[k] - exp_dict2[k])
                                                                         for k in exp_dict1 if k in features_to_show}
            best_features = lambda exp_dict: sorted(exp_dict.keys(), key=lambda key: abs(exp_dict[key]), reverse=True)[
                                             :self.n_features_to_show]

            model_regressor = ElasticNetWrapper(random_state=0, l1_ratio=0.9, alpha=1e-3, warm_start=True, copy_X=False,
                                                selection='random', tol=1e-4)

            num_samples = 16000
            exp = lm.explainer.explain_instance(sample, self.predict_function,
                                                labels=self.labels,
                                                num_features=self.n_features_to_show,
                                                num_samples=num_samples,
                                                model_regressor=model_regressor)

            lower_exp = as_dict(exp)

            converged = False
            while not converged and num_samples < self.max_samples:
                num_samples *= 2
                exp = lm.explainer.explain_instance(sample, self.predict_function,
                                                    labels=self.labels,
                                                    num_features=self.n_features_to_show,
                                                    num_samples=num_samples,
                                                    model_regressor=model_regressor)

                higher_exp = as_dict(exp)

                features_to_show = best_features(higher_exp)
                diff = dict_delta(lower_exp, higher_exp, features_to_show)

                max_value = numpy.max(numpy.abs(list(higher_exp.values())))
                delta = numpy.array(list(diff.values())) / max_value

                if max(delta) < self.tol:
                    converged = True
                else:
                    converged = False
                    lower_exp = higher_exp

            if not converged:
                logging.warning("Lime explainer did not converge with {} samples".format(num_samples))

            return exp

        except ValueError as e:
            logging.exception("Failed to explain Lime for instance\n{}".format(sample))
            raise


if __name__ == '__main__':
    from examples.datasets.california_housing import CaliforniaHousing

    # dataset = UCIAdult()
    dataset = CaliforniaHousing()

    train, test = dataset.as_dmd()
    print(train.categorical_encoding_by_feature_name)

    model = dataset.get_model()

    lm = LimeReport()
    lm.fit(train, model)

    # asking for explanation for LIME model

    for i in range(3):
        # # yp = model.predict_proba(dmd_train.values[i, :]) if is_classification else model.predict(dmd_train.values[i, :])
        # exp = lm.explainer.explain_instance(test.values[i, :], model.predict_proba, labels=numpy.arange(len(test.labels)), num_features=train.n_features)
        # # exp.as_pyplot_figure(label=int(model.predict(test.values[i, :].reshape(1,-1))))
        # exp.as_pyplot_figure()
        # print(test._x.iloc[i, :])
        # print(model.predict(test.values[i, :].reshape(1,-1)))
        lm.plot(test.values[i, :])
        # explain = lm.explain(test.values[0,:])
        # e0 = explain[0]
        # e1 = explain[1]
        # print(e0)
        a = 0

    plt.show()
