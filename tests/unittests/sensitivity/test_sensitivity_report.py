import unittest

from pytolemaic.analysis_logic.model_analysis.sensitivity.sensitivity_reports import SensitivityStatsReport, \
    SensitivityVulnerabilityReport, SensitivityOfFeaturesReport, SensitivityFullReport, SensitivityTypes


class TestSensitivityReport(unittest.TestCase):

    def equal_to_dict_keys(self, rep):
        d1 = rep.to_dict()
        d2 = rep.to_dict_meaning()
        return d1.keys() == d2.keys()

    def test_to_dict_meaning(self):
        rep1 = SensitivityStatsReport(n_features=2, n_low=0, n_very_low=0, n_zero=0)
        self.assertTrue(self.equal_to_dict_keys(rep1))

        rep2 = SensitivityVulnerabilityReport(imputation=0.5, leakage=0.5, too_many_features=0.5)
        self.assertTrue(self.equal_to_dict_keys(rep2))

        rep3 = SensitivityOfFeaturesReport(method=SensitivityTypes.shuffled, sensitivities=dict(a=0.2, b=0.8),
                                           stats_report=rep1)
        self.assertTrue(self.equal_to_dict_keys(rep3))

        rep4 = SensitivityFullReport(shuffle_report=rep3,
                                     missing_report=rep3,
                                     vulnerability_report=rep2)
        self.assertTrue(self.equal_to_dict_keys(rep4))
