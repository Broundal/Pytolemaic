import unittest

from pytolemaic.analysis_logic.model_analysis.scoring.scoring_report import ConfusionMatrixReport, ScatterReport, \
    ScoringMetricReport, ScoringFullReport


class TestScoringReport(unittest.TestCase):

    def equal_to_dict_keys(self, rep):
        d1 = rep.to_dict()
        d2 = rep.to_dict_meaning()
        return d1.keys() == d2.keys()

    def test_to_dict_meaning(self):
        rep1 = ConfusionMatrixReport(y_true=[1,2,3], y_pred=[1,2,3])
        self.assertTrue(self.equal_to_dict_keys(rep1))

        rep2 = ScatterReport(y_true=[1,2,3], y_pred=[1,2,3])
        self.assertTrue(self.equal_to_dict_keys(rep2))

        rep3 = ScoringMetricReport(metric='mae', value=0.5, ci_low=0.25, ci_high=0.75)
        self.assertTrue(self.equal_to_dict_keys(rep3))

        rep4 = ScoringFullReport(target_metric='mae', metric_reports=[rep3], separation_quality=0.2,
                                 confusion_matrix=rep1, scatter=rep2)
        self.assertTrue(self.equal_to_dict_keys(rep4))

