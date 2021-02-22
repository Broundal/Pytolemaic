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
        self.max_samples_to_use = 20000
        self.low_sensitivity_threshold = 0.05
        self.very_low_sensitivity_threshold = 1e-4

    @classmethod
    def shuffle_x(cls, x, index, dmd_train=None, seed=0):
        rs = np.random.RandomState(seed)
        if dmd_train is None:
            new_order = rs.permutation(x.shape[0])
            x[:, index] = x[:, index][new_order]
        else:
            new_order = rs.permutation(dmd_train.n_samples)[:x.shape[0]]
            x[:, index] = dmd_train.values[:, index][new_order]
        return x

    @classmethod
    def get_shuffled_x(cls, dmdx: DMD, index=None, dmd_train=None, method=SensitivityTypes.shuffled, seed=0,
                       model_support_dmd=False):
        if index is None:
            return dmdx.values

        x_copy = numpy.copy(dmdx.values)
        if method == SensitivityTypes.shuffled:
            x_copy = cls.shuffle_x(x_copy, dmd_train=dmd_train, index=index, seed=index + seed)
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
                             dmd_train=None,
                             method=SensitivityTypes.shuffled, raw_scores=False,
                             y_pred=None) -> SensitivityOfFeaturesReport:

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
            if dmd_test.n_samples > self.max_samples_to_use:
                rs = numpy.random.RandomState(i)
                subset = rs.permutation(dmd_test.n_samples)[:self.max_samples_to_use]
                dmd_test_ = dmd_test.split_by_indices(subset)
                y_pred_ = y_pred[subset]
            else:
                dmd_test_ = dmd_test
                y_pred_ = y_pred

            shuffled_x = self.get_shuffled_x(dmd_test_, i, dmd_train=dmd_train, method=method,
                                             model_support_dmd=self.model_support_dmd)
            shuffled_pred = predict_function(shuffled_x)

            if base_score > 0:
                scores[name] = 1 - abs(base_score - score_function(y_pred_,
                                                                   shuffled_pred))  # higher difference - more impact so add 1- in front
            else:
                scores[name] = score_function(y_pred_, shuffled_pred)  # higher score - less impact

        if raw_scores:
            # description = "The raw scores of how each feature affects the model's predictions."
            return SensitivityOfFeaturesReport(method=method, sensitivities=scores,
                                               stats_report=self._sensitivity_stats_report(scores))

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
        return SensitivityOfFeaturesReport(method=method, sensitivities=impact,
                                           stats_report=self._sensitivity_stats_report(sensitivities=impact))

    def _sensitivity_stats_report(self, sensitivities: dict) -> [SensitivityStatsReport, None]:
        if not sensitivities:
            return None

        sensitivity = np.abs(np.array(list(sensitivities.values())))
        n_features = len(sensitivity)

        low_sensitivity = max(sensitivity) * self.low_sensitivity_threshold
        very_low_sensitivity = max(sensitivity) * self.very_low_sensitivity_threshold
        zero_sensitivity = 0

        n_low = int(np.sum(sensitivity < low_sensitivity))
        n_very_low = int(np.sum(sensitivity < very_low_sensitivity))
        n_zero = int(np.sum(sensitivity <= zero_sensitivity))

        return SensitivityStatsReport(n_features=n_features,
                                      n_low=n_low,
                                      n_very_low=n_very_low,
                                      n_zero=n_zero)

    def _vulnerability_report(self, shuffled_sensitivity: SensitivityOfFeaturesReport,
                              missing_sensitivity: SensitivityOfFeaturesReport) -> SensitivityVulnerabilityReport:
        # lower is better
        stats = shuffled_sensitivity.stats_report

        leakage = self._leakage(n_features=stats.n_features,
                                n_very_low=stats.n_very_low)
        too_many_features = self._too_many_features(n_features=stats.n_features,
                                                    n_low=stats.n_low,
                                                    n_very_low=stats.n_very_low,
                                                    n_zero=stats.n_zero)
        imputation = self._imputation_score(
            shuffled=shuffled_sensitivity,
            missing=missing_sensitivity)

        return SensitivityVulnerabilityReport(imputation=imputation,
                                              too_many_features=too_many_features,
                                              leakage=leakage)

    def _leakage(self, n_features, n_very_low, **kwargs):
        """
        measure the chance for data leakage - strong data leakage cause only few features to contribute to the model.
        :param shuffled_sensitivity:
        :return:
        """

        if n_features < 2:
            return 0

        n_non_zero = n_features - n_very_low

        return np.power(n_very_low / n_features, n_non_zero - 1)

    def _too_many_features(self, n_features, n_low, n_very_low, n_zero, **kwargs):
        """
        Many features with low sensitivity indicate the model relies on non-informative feature. This may cause overfit.
        higher value when there are many features with low contribution
        :param shuffled_sensitivity:
        :return:
        """

        return max((n_low-n_zero) / n_features, np.sqrt((n_very_low-n_zero) / n_features))

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

    def calculate_sensitivity(self, model, dmd_test: DMD, metric: str, dmd_train=None, **kwargs):
        dmd_test = dmd_test or dmd_train
        self.shuffled_sensitivity = self.sensitivity_analysis(
            model=model,
            dmd_train=dmd_train,
            dmd_test=dmd_test,
            metric=metric,
            method=SensitivityTypes.shuffled,
            raw_scores=False)

        try:
            self.missing_sensitivity = self.sensitivity_analysis(
                model=model,
                dmd_test=dmd_test,
                dmd_train=dmd_train,
                metric=metric,
                method=SensitivityTypes.missing,
                raw_scores=False)
        except:
            logging.error(
                "Failed to calculate sensitivity with {} method. This is expected if your model cannot handle missing values.".format(
                    SensitivityTypes.missing))

            self.missing_sensitivity = None

    def sensitivity_report(self, **kwargs) -> SensitivityFullReport:

        vulnerability_report = self._vulnerability_report(
            shuffled_sensitivity=self.shuffled_sensitivity,
            missing_sensitivity=self.missing_sensitivity)

        return SensitivityFullReport(
            shuffle_report=self.shuffled_sensitivity,
            missing_report=self.missing_sensitivity,
            vulnerability_report=vulnerability_report
        )
