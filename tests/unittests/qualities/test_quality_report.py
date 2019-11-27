import unittest

from pytolemaic.analysis_logic.model_analysis.scoring.scoring_report import ScoringFullReport, ConfusionMatrixReport, \
    ScatterReport, ScoringMetricReport
from pytolemaic.analysis_logic.model_analysis.sensitivity.sensitivity_reports import SensitivityVulnerabilityReport
from pytolemaic.utils.metrics import Metrics

from pytolemaic.dataset_quality_report import TestSetQualityReport, TrainSetQualityReport, QualityReport


class TestSensitivityReport(unittest.TestCase):

    def equal_to_dict_keys(self, rep):
        d1 = rep.to_dict()
        d2 = rep.to_dict_meaning()
        return d1.keys() == d2.keys()

    def get_scoring_report(self):
        rep1 = ConfusionMatrixReport(y_true=[1, 2, 3], y_pred=[1, 2, 3])
        rep2 = ScatterReport(y_true=[1, 2, 3], y_pred=[1, 2, 3])
        rep3 = ScoringMetricReport(metric='mae', value=0.5, ci_low=0.25, ci_high=0.75)
        return ScoringFullReport(metric_reports=[rep3], separation_quality=0.2, confusion_matrix=rep1, scatter=rep2)

    def get_vulnerability_report(self):
        return SensitivityVulnerabilityReport(imputation=0.5, leakage=0.5, too_many_features=0.5)


    def test_to_dict_meaning(self):
        rep1 = TestSetQualityReport(self.get_scoring_report(), metric=Metrics.mae.name)
        self.assertTrue(self.equal_to_dict_keys(rep1))

        rep2 = TrainSetQualityReport(self.get_vulnerability_report())
        self.assertTrue(self.equal_to_dict_keys(rep2))

        rep3 = QualityReport(test_quality_report=rep1, train_quality_report=rep2)
        self.assertTrue(self.equal_to_dict_keys(rep3))
