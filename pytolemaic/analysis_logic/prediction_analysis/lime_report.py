import logging

import numpy
try:
    from lime.lime_tabular import LimeTabularExplainer
except Exception as e:
    logging.exception("Failed to import lime module:")
    logging.error("Please try to run 'pip install lime'")
    raise

from matplotlib import pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.utils.multiclass import unique_labels

from pytolemaic.utils.dmd import DMD
from pytolemaic.utils.general import GeneralUtils


class ElasticNetWrapper(ElasticNet):

    def fit(self, X, y, check_input=False, sample_weight=None):
        return super(ElasticNetWrapper, self).fit(X, y, check_input)


class LimeExplainer():
    def __init__(self, kernel_width=3, n_features_to_plot=None, tol=1e-2, max_samples=256000, fillna=0):
        """

        :param kernel_width: Lime parameter
        :param n_features_to_plot: # of features to show/plot
        :param tol: desired convergence tol on explanation
        :param max_samples: limit on # of samples used to create explanation
        """

        self.kernel_width = kernel_width
        self.n_features_to_plot = n_features_to_plot
        self.predict_function = None
        self.tol = tol
        self.max_samples = max_samples
        self.fillna = fillna

    def fit(self, dmd_train: DMD, model):
        is_classification = GeneralUtils.is_classification(model)

        x = dmd_train.values.astype(float, copy=True)
        nan_mask = ~numpy.isfinite(x)
        if numpy.any(nan_mask):
            logging.warning(
                "Lime cannot handle missing values. Fillna={} was used to coerce the issue.".format(self.fillna))
            x[nan_mask] = self.fillna

        self.explainer = LimeTabularExplainer(
            training_data=x,
            mode="classification" if is_classification else "regression",
            training_labels=None,  # ???
            feature_names=dmd_train.feature_names,
            categorical_features=dmd_train.categorical_features if dmd_train.categorical_features is not None else [],
            ###
            categorical_names=dmd_train.categorical_encoding_by_icols,  ###
            kernel_width=self.kernel_width,
            kernel=None,
            # default is np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))
            verbose=False,
            class_names=dmd_train.labels,
            feature_selection='auto',
            # ??? options are 'forward_selection', 'lasso_path', 'none' or 'auto'.
            discretize_continuous=True,
            discretizer='decile',
            # -- Lime discretizers do not support nans (options are 'quartile', 'decile', 'entropy')
            sample_around_instance=True,  # default is False
            random_state=0,
            training_data_stats=None)

        self.model = model
        self.predict_function = self.model.predict_proba if is_classification else self.model.predict

        if is_classification:
            labels = dmd_train.labels
            if labels is None:
                labels = set(unique_labels(dmd_train.target))
            self.labels = numpy.arange(len(labels))
        else:
            self.labels = None

        self.n_features = dmd_train.n_features
        self.n_features_to_plot = self.n_features_to_plot or dmd_train.n_features
        self.n_features_to_plot = min(self.n_features_to_plot, self.n_features)

    def explain(self, sample: numpy.ndarray) -> [dict, None]:
        try:
            exp = self._converged_lime_explaination(sample)

            label = self.model.predict(sample.reshape(1, -1))
            return dict(exp.as_list(label=int(label)))
        except:
            logging.exception("Failed to produce lime explanation for sample {}".format(sample))
            return None

    def plot(self, sample: numpy.ndarray):
        try:
            exp = self._converged_lime_explaination(sample)

            label = self.model.predict(sample.reshape(1, -1))

            if GeneralUtils.is_classification(self.model):
                exp.as_pyplot_figure(label=int(label))
            else:
                exp.as_pyplot_figure(label=None)
                plt.title("Local explanation for predicted value of %.3g" % label)

            plt.tight_layout()
            plt.draw()
        except ValueError as e:
            logging.exception("Failed to plot Lime for instance\n{}".format(sample))

    def _lime_explaination(self, sample, num_samples=16000):
        model_regressor = ElasticNetWrapper(random_state=0, l1_ratio=0.9, alpha=1e-3, warm_start=True, copy_X=False,
                                            selection='random', tol=1e-4)

        exp = self.explainer.explain_instance(sample.ravel(),
                                              self.predict_function,
                                              labels=self.labels,
                                              num_features=self.n_features,
                                              num_samples=num_samples,
                                              model_regressor=model_regressor)

        return exp

    def _convergence_acheived(self, lower_exp, higher_exp):
        features_to_show = sorted(higher_exp.keys(), key=lambda key: abs(higher_exp[key]), reverse=True)[:self.n_features_to_plot]
        # inefficient diff, but self.n_features_to_plot is expected to be <20
        diff = {k: abs(lower_exp[k] - higher_exp[k]) for k in lower_exp if k in features_to_show}

        max_value = numpy.max(numpy.abs(list(higher_exp.values())))
        delta = numpy.array(list(diff.values())) / max_value
        if max(delta) < self.tol:
            converged = True
        else:
            converged = False
        return converged

    def _converged_lime_explaination(self, sample):
        def as_dict(exp):
            return {k: numpy.round(v, 5) for k, v in exp.as_list()}

        sample = numpy.array(sample, dtype=float, copy=True)
        nan_mask = ~numpy.isfinite(sample)
        if numpy.any(nan_mask):
            logging.warning("Lime cannot handle missing values. Fillna(0) was used to coerce the issue.")
            sample = numpy.copy(sample)
            sample[nan_mask] = self.fillna

        try:

            num_samples = min(16000, self.max_samples // 2)

            exp = self._lime_explaination(sample=sample, num_samples=num_samples)
            higher_exp = as_dict(exp)

            converged = False
            while not converged and num_samples < self.max_samples:
                num_samples *= 2
                lower_exp = higher_exp
                exp = self._lime_explaination(sample=sample, num_samples=num_samples)
                higher_exp = as_dict(exp)

                converged = self._convergence_acheived(lower_exp=lower_exp, higher_exp=higher_exp)

            if not converged:
                logging.warning("Lime explainer did not converge with {} samples".format(num_samples))

            return exp

        except ValueError as e:
            logging.exception("Failed to explain Lime for instance\n{}".format(sample))
            raise



