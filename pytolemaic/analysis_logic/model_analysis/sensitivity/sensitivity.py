import logging

import numpy
import numpy as np

from pytolemaic.analysis_logic.model_analysis.sensitivity.sensitivity_reports import SensitivityOfFeaturesReport, \
    SensitivityStatsReport, SensitivityVulnerabilityReport, SensitivityFullReport, SensitivityTypes
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
    def get_shuffled_x(cls, dmdx: DMD, index=None, method=SensitivityTypes.shuffled, seed=0,
                       model_support_dmd=False):
        if index is None:
            return dmdx.values

        x_copy = numpy.copy(dmdx.values)
        if method == SensitivityTypes.shuffled:
            x_copy = cls.shuffle_x(x_copy, index=index, seed=index + seed)
        if method == SensitivityTypes.missing:
            x_copy[:, index] = np.nan

        if model_support_dmd:
            return DMD(x=x_copy,
                       samples_meta=dmdx._samples_meta,
                       columns_meta=dmdx._columns_meta,
                       splitter=dmdx.splitter)
        else:
            return x_copy

    def sensitivity_analysis(self, model, dmd_test: DMD, metric,
                             method=SensitivityTypes.shuffled, raw_scores=False,
                             y_pred=None):

        self.model_support_dmd = GeneralUtils.dmd_supported(model, dmd_test)
        x = dmd_test if self.model_support_dmd else dmd_test.values

        y_pred = y_pred or model.predict(x)
        ytest = dmd_test.target

        score_function = self.metrics[metric].function
        if metric in ['auc', 'logloss'] and ytest is not None:
            base_score = score_function(ytest, model.predict_proba(x))
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

            if base_score > 0:
                scores[name] = 1 - abs(base_score - score_function(y_pred,
                                                                   shuffled_pred))  # higher difference - more impact so add 1- in front
            else:
                scores[name] = score_function(y_pred, shuffled_pred)  # higher score - less impact

        if raw_scores:
            # description = "The raw scores of how each feature affects the model's predictions."
            return SensitivityOfFeaturesReport(method=method, sensitivities=scores)

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
        return SensitivityOfFeaturesReport(method=method, sensitivities=impact)

    def _sensitivity_stats(self, sensitivity: SensitivityOfFeaturesReport):
        if not sensitivity:
            return {}

        sensitivity = np.array(list(sensitivity.sensitivities.values()))
        n_features = len(sensitivity)
        n_zero = int(np.sum(sensitivity < min(1e-4, 1 / n_features)))
        n_low = int(np.sum(sensitivity < max(sensitivity) * 0.05))
        return SensitivityStatsReport(n_features=n_features,
                                      n_low=n_low,
                                      n_zero=n_zero)

    def _vulnerability_report(self, shuffled_sensitivity: SensitivityOfFeaturesReport,
                              missing_sensitivity: SensitivityOfFeaturesReport,
                              shuffled_sensitivity_stats: SensitivityStatsReport):
        # lower is better
        stats = shuffled_sensitivity_stats

        leakage = self._leakage(n_features=stats.n_features,
                                n_zero=stats.n_zero)
        too_many_features = self._too_many_features(n_features=stats.n_features,
                                                    n_low=stats.n_low,
                                                    n_zero=stats.n_zero)
        imputation = self._imputation_score(
            shuffled=shuffled_sensitivity,
            missing=missing_sensitivity)

        return SensitivityVulnerabilityReport(imputation=imputation,
                                              too_many_features=too_many_features,
                                              leakage=leakage)

    def _leakage(self, n_features, n_zero, **kwargs):
        """
        measure the chance for data leakage - strong data leakage cause only few features to contribute to the model.
        :param shuffled_sensitivity:
        :return:
        """

        if n_features < 2:
            return 0

        n_non_zero = n_features - n_zero

        return np.power(n_zero / n_features, n_non_zero - 1)

    def _too_many_features(self, n_features, n_low, n_zero, **kwargs):
        """
        Many features with low sensitivity indicate the model relies on non-informative feature. This may cause overfit.
        higher value when there are many features with low contribution
        :param shuffled_sensitivity:
        :return:
        """

        return max(n_low / n_features, np.sqrt(n_zero / n_features))

    def _imputation_score(self, shuffled: SensitivityOfFeaturesReport, missing: SensitivityOfFeaturesReport):
        """
        missing sensitivity should (more or less) match shuffled in impact.
        If it does not match, it may mean there is an issue with imputation
        :param shuffled:
        :param missing:
        :return:
        """
        if not missing:
            return 0

        deltas = numpy.abs([shuffled.sensitivities[i] - missing.sensitivities[i] for i in shuffled.sensitivities])
        deltas = deltas[deltas >= max(deltas) * 1e-3]
        if max(abs(deltas)) == 0:
            return 0

        score = np.mean(deltas) / max(deltas)
        return score

    def calculate_sensitivity(self, model, dmd_test: DMD, metric: str):
        self.shuffled_sensitivity = self.sensitivity_analysis(
            model=model,
            dmd_test=dmd_test,
            metric=metric,
            method=SensitivityTypes.shuffled,
            raw_scores=False)

        try:
            self.missing_sensitivity = self.sensitivity_analysis(
                model=model,
                dmd_test=dmd_test,
                metric=metric,
                method=SensitivityTypes.missing,
                raw_scores=False)
        except:
            logging.error(
                "Failed to calculate sensitivity with {} method... Does your model handle missing values?".format(
                    SensitivityTypes.missing))

            self.missing_sensitivity = None

    def sensitivity_report(self) -> SensitivityFullReport:

        shuffle_stats_report = self._sensitivity_stats(self.shuffled_sensitivity)
        missing_stats_report = self._sensitivity_stats(self.missing_sensitivity)
        vulnerability_report = self._vulnerability_report(
            shuffled_sensitivity=self.shuffled_sensitivity,
            missing_sensitivity=self.missing_sensitivity,
            shuffled_sensitivity_stats=missing_stats_report)

        return SensitivityFullReport(
            shuffle_report=self.shuffled_sensitivity,
            shuffle_stats_report=shuffle_stats_report,
            missing_report=self.missing_sensitivity,
            missing_stats_report=missing_stats_report,
            vulnerability_report=vulnerability_report
        )
