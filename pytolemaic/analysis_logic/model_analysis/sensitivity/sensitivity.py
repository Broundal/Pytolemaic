import logging

import numpy
import numpy as np

from pytolemaic.utils.dmd import DMD
from pytolemaic.utils.general import GeneralUtils
from pytolemaic.utils.metrics import Metrics
from pytolemaic.utils.report import Report
from pytolemaic.utils.report_keys import ReportSensitivity


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

        scores = {}
        for i, name in enumerate(dmd_test.feature_names):
            shuffled_x = self.get_shuffled_x(dmd_test, i, method=method,
                                             model_support_dmd=self.model_support_dmd)
            shuffled_pred = predict_function(shuffled_x)
            scores[name] = score_function(y_pred, shuffled_pred) - base_score

        if raw_scores:
            # description = "The raw scores of how each feature affects the model's predictions."
            return scores

        # higher score / lower loss means the shuffled feature did less impact
        if self.metrics[metric].is_loss:
            impact = scores
        else:
            impact = {name: 1 - score for name, score in scores.items()}

        total_impact = sum([score for score in impact.values()])
        impact = {name: float(score / total_impact) for
                  name, score in impact.items()}
        impact = GeneralUtils.round_values(impact)

        # description="The impact of each feature on model's predictions. "
        #             "Higher value mean larger impact (0 means no impact at all). "
        #             "Values are normalized to 1.")
        return impact

    def _sensitivity_meta(self, sensitivity):
        if not sensitivity:
            return {}

        sensitivity = np.array(list(sensitivity.values()))
        n_features = len(sensitivity)
        n_zero = np.sum(sensitivity < min(1e-4, 1 / n_features))
        n_low = np.sum(sensitivity < max(sensitivity) * 0.05)
        return {
            ReportSensitivity.N_FEATURES: n_features,
            ReportSensitivity.N_ZERO: n_zero,
            ReportSensitivity.N_NON_ZERO: n_features - n_zero,
            ReportSensitivity.N_LOW: n_low
        }

    def _sensitivity_scores(self, perturbed_sensitivity, missing_sensitivity,
                            perturbed_sensitivity_meta):
        # lower is better
        n_features = perturbed_sensitivity_meta[ReportSensitivity.N_FEATURES]
        n_zero = perturbed_sensitivity_meta[ReportSensitivity.N_ZERO]
        n_low = perturbed_sensitivity_meta[ReportSensitivity.N_LOW]

        report = {}
        report[ReportSensitivity.LEAKAGE] = self._leakage(n_features=n_features,
                                                          n_zero=n_zero)
        report[ReportSensitivity.OVERFIIT] = self._overfit(n_features=n_features,
                                                           n_low=n_low,
                                                           n_zero=n_zero)
        report[ReportSensitivity.IMPUTATION] = self._imputation_score(
            shuffled=perturbed_sensitivity,
            missing=missing_sensitivity)

        report = GeneralUtils.round_values(report)
        return report

    def _leakage(self, n_features, n_zero, **kwargs):
        """
        measure the chance for data leakage - strong data leakage cause only few features to contribute to the model.
        :param perturbed_sensitivity:
        :return:
        """

        if n_features < 2:
            return 0

        n_non_zero = n_features - n_zero

        return np.power(n_zero / n_features, n_non_zero - 1)

    def _overfit(self, n_features, n_low, n_zero, **kwargs):
        """
        Many features with low sensitivity indicate the model relies on non-informative feature. This may cause overfit.
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
        if max(abs(deltas)) == 0:
            return 0

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
            logging.error(
                "Failed to calculate sensitivity with 'missing' method... Does your model handle missing values?")

            self.missing_sensitivity = {}

    def sensitivity_report(self):
        report = {}

        report[ReportSensitivity.SHUFFLE] = {}
        report[ReportSensitivity.SHUFFLE][ReportSensitivity.SENSITIVITY] = self.perturbed_sensitivity
        perturb_meta = self._sensitivity_meta(self.perturbed_sensitivity)
        report[ReportSensitivity.SHUFFLE][ReportSensitivity.META] = perturb_meta

        report[ReportSensitivity.MISSING]  = {}
        report[ReportSensitivity.MISSING][ReportSensitivity.SENSITIVITY]= self.missing_sensitivity
        report[ReportSensitivity.MISSING][ReportSensitivity.META]= self._sensitivity_meta(
            self.missing_sensitivity)

        report[ReportSensitivity.VULNERABILITY] = self._sensitivity_scores(
            perturbed_sensitivity=self.perturbed_sensitivity,
            missing_sensitivity=self.missing_sensitivity,
            perturbed_sensitivity_meta=perturb_meta)

        return Report(report)
