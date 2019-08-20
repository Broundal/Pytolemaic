import logging

import numpy
import numpy as np

from pytolemaic.utils.dmd import DMD
from pytolemaic.utils.general import GeneralUtils
from pytolemaic.utils.metrics import Metrics


class SensitivityAnalysis():
    def __init__(self):
        self.metrics = Metrics.supported_metrics()
        self.model_support_dmd = None

    @classmethod
    def shuffle_x(cls, x, index, seed=0):
        rs = np.random.RandomState(seed)
        new_order = rs.permutation(x.shape[0])
        x[:, index] = x[:, index][new_order]
        return x

    @classmethod
    def get_shuffled_x(cls, dmdx: DMD, index=None, method='perturb', seed=0,
                       model_support_dmd=False):
        if index is None:
            return dmdx.values

        x_copy = numpy.copy(dmdx.values)
        if method == 'perturb':
            x_copy = cls.shuffle_x(x_copy, index=index, seed=index + seed)
        if method == 'missing':
            x_copy[:, index] = np.nan

        if model_support_dmd:
            return DMD(x=x_copy,
                       samples_meta=dmdx._samples_meta,
                       columns_meta=dmdx._columns_meta,
                       splitter=dmdx.splitter)
        else:
            return x_copy

    def sensitivity_analysis(self, model, dmd_test: DMD, metric,
                             method='perturb', raw_scores=False,
                             y_pred=None):

        self.model_support_dmd = GeneralUtils.dmd_supported(model, dmd_test)
        x = dmd_test if self.model_support_dmd else dmd_test.values

        y_pred = y_pred or model.predict(x)
        ytest = dmd_test.target

        score_function = self.metrics[metric].function
        if metric in ['auc', 'logloss'] and ytest is not None:
            base_score = score_function(ytest, y_pred)
            y_pred = ytest
        else:
            base_score = 0
            y_pred = y_pred

        predict_function = model.predict_proba if self.metrics[
            metric].is_proba \
            else model.predict

        pred_func = lambda i: predict_function(
            self.get_shuffled_x(dmd_test, i, method=method,
                                model_support_dmd=self.model_support_dmd))

        scores = {name: score_function(y_pred, pred_func(i)) - base_score
                  for i, name in enumerate(dmd_test.feature_names)}

        if raw_scores:
            return scores

        # higher score / lower loss means the shuffled feature did less impact
        if self.metrics[metric].is_loss:
            impact = scores
        else:
            impact = {name: 1 - score for name, score in scores.items()}

        total_impact = sum([score for score in impact.values()])
        impact = {name: float(numpy.round(score / total_impact, 8)) for
                  name, score in impact.items()}

        return impact

    def _sensitivity_meta(self, sensitivity):
        if not sensitivity:
            return {}

        sensitivity = np.array(list(sensitivity.values()))
        n_features = len(sensitivity)
        n_zero = np.sum(sensitivity < min(1e-4, 1 / n_features))
        n_low = np.sum(sensitivity < max(sensitivity) * 0.05)
        return {
            'n_features': n_features,
            'n_zero': n_zero,
            'n_non_zero': n_features - n_zero,
            'n_low': n_low
        }

    def _sensitivity_scores(self, perturbed_sensitivity, missing_sensitivity,
                            perturbed_sensitivity_meta):
        report = {}
        report['leakge_score'] = self._leakage(**perturbed_sensitivity_meta)
        report['overfit_score'] = self._overfit(**perturbed_sensitivity_meta)
        report['imputation_score'] = self._imputation_score(
            shuffled=self.perturbed_sensitivity,
            missing=self.missing_sensitivity)

        return report

    def _leakage(self, n_features, n_zero, **kwargs):
        """
        measure the chance for data leakage
        :param perturbed_sensitivity:
        :return:
        """

        if n_features < 2:
            return 0

        n_non_zero = n_features - n_zero

        return np.power(n_zero / n_features, n_non_zero - 1)

    def _overfit(self, n_features, n_low, n_zero, **kwargs):
        """
        higher value when there are many features with low contribution
        :param perturbed_sensitivity:
        :return:
        """

        return max(n_low / n_features, np.sqrt(n_zero / n_features))

    def _imputation_score(self, shuffled, missing):
        """
        missing sensitivity should (more or less) match shuffled in impact.
        If it does not match, it may mean there is an issue with imputation
        :param shuffled:
        :param missing:
        :return:
        """
        if not missing:
            return 0

        deltas = numpy.abs([shuffled[i] - missing[i] for i in shuffled])
        deltas = deltas[deltas >= max(deltas) * 1e-3]
        score = np.mean(deltas) / max(deltas)
        return score

    def calculate_sensitivity(self, model, dmd_test: DMD, metric: str):
        self.perturbed_sensitivity = self.sensitivity_analysis(
            model=model,
            dmd_test=dmd_test,
            metric=metric,
            method='perturb',
            raw_scores=False)

        try:
            self.missing_sensitivity = self.sensitivity_analysis(
                model=model,
                dmd_test=dmd_test,
                metric=metric,
                method='missing',
                raw_scores=False)
        except:
            logging.exception(
                "Failed to calculate sensitivity with 'missing' method. Does you model support imputation?")
            self.missing_sensitivity = {}

    def sensitivity_report(self):
        report = {}

        report['perturbed_sensitivity'] = self.perturbed_sensitivity

        report['missing_sensitivity'] = self.missing_sensitivity

        report['perturbed_sensitivity_meta'] = self._sensitivity_meta(
            self.perturbed_sensitivity)
        report['missing_sensitivity_meta'] = self._sensitivity_meta(
            self.missing_sensitivity)

        report['perturbed_sensitivity_scores'] = self._sensitivity_scores(
            perturbed_sensitivity=self.perturbed_sensitivity,
            missing_sensitivity=self.missing_sensitivity,
            perturbed_sensitivity_meta=report['perturbed_sensitivity_meta'])

        return report
