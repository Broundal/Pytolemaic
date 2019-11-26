from pytolemaic.utils.metrics import Metrics

from pytolemaic.analysis_logic.model_analysis.scoring.scoring_report import ScoringFullReport
from pytolemaic.analysis_logic.model_analysis.sensitivity.sensitivity_reports import SensitivityVulnerabilityReport


class TestSetQualityReport():
    def __init__(self, scoring_report:ScoringFullReport, metric):
        self._scoring_report = scoring_report
        self._metric = metric

        self._test_set_quality = self._calculate_test_set_quality()


    def _calculate_test_set_quality(self):
        test_set_quality = 1.0
        test_set_quality = test_set_quality - self.ci_ratio - (1 - self.separation_quality)
        test_set_quality = max(test_set_quality, 0)
        return test_set_quality

    @property
    def ci_ratio(self):
        return self._scoring_report.metric_scores[self._metric].ci_ratio

    @property
    def metric(self):
        return self._metric

    @property
    def separation_quality(self):
        return self._scoring_report.separation_quality

    @property
    def scoring_report(self)->ScoringFullReport:
        return self._scoring_report

    @property
    def test_set_quality(self):
        return self._test_set_quality


class TrainSetQualityReport():
    def __init__(self, vulnerability_report: SensitivityVulnerabilityReport ):
        self._vulnerability_report = vulnerability_report

        self._test_set_quality = self._calculate_train_set_quality()

    def _calculate_train_set_quality(self):
        train_set_quality = 1.0
        train_set_quality = train_set_quality \
                            - self._vulnerability_report.leakage \
                            - self._vulnerability_report.too_many_features \
                            - self._vulnerability_report.imputation
        train_set_quality = max(train_set_quality, 0)

        return train_set_quality

    @property
    def vulnerability_report(self)->SensitivityVulnerabilityReport:
        return self._vulnerability_report

    @property
    def train_set_quality(self):
        return self._train_set_quality


class QualityReport():
    def __init__(self, train_quality_report:TrainSetQualityReport, test_quality_report:TestSetQualityReport):
        self._train_quality_report = train_quality_report
        self._test_quality_report = test_quality_report

    @property
    def train_quality_report(self):
        return self._train_quality_report

    @property
    def test_quality_report(self):
        return self._test_quality_report


