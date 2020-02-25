import unittest
from pprint import pprint

from pytolemaic.analysis_logic.model_analysis.sensitivity.sensitivity import \
    SensitivityAnalysis
from pytolemaic.analysis_logic.model_analysis.sensitivity.sensitivity_reports import SensitivityOfFeaturesReport, \
    SensitivityTypes


class TestSensitivity(unittest.TestCase):

    def test_sensitivity_meta(self):

        sensitivity = SensitivityAnalysis()
        sensitivities = {'a' + str(k): k for k in range(10)}

        mock = SensitivityOfFeaturesReport(method='mock', sensitivities=sensitivities,
                                           stats_report=sensitivity._sensitivity_stats_report(sensitivities))
        stats = mock.stats_report
        pprint(stats.to_dict())

        self.assertEqual(stats.n_features, 10)
        self.assertEqual(stats.n_zero, 1)
        self.assertEqual(stats.n_low, 1)

    def test_leakage(self):
        sensitivity = SensitivityAnalysis()
        self.assertEqual(sensitivity._leakage(n_features=10, n_very_low=0), 0)
        self.assertEqual(sensitivity._leakage(n_features=10, n_very_low=9), 1)
        self.assertGreaterEqual(sensitivity._leakage(n_features=10, n_very_low=8),
                                0.8)

        print([sensitivity._leakage(n_features=10, n_very_low=k) for k in
               range(10)])

    def test_overfit(self):
        sensitivity = SensitivityAnalysis()

        print([sensitivity._too_many_features(n_features=15, n_low=k+5, n_very_low=5+k//2, n_zero=5) for k in
               range(10)])

        self.assertEqual(
            sensitivity._too_many_features(n_features=10, n_low=0, n_very_low=0, n_zero=0), 0)
        self.assertEqual(
            sensitivity._too_many_features(n_features=10, n_low=5, n_very_low=0, n_zero=0), 0.5)

        self.assertGreater(
            sensitivity._too_many_features(n_features=10, n_low=9, n_very_low=9, n_zero=0), 0.9)


        self.assertGreaterEqual(
            sensitivity._too_many_features(n_features=10, n_low=9, n_very_low=9, n_zero=9), 0.0)

    def test_imputation(self):
        sensitivity = SensitivityAnalysis()

        shuffled = SensitivityOfFeaturesReport(method=SensitivityTypes.shuffled,
                                               sensitivities={'a': 0.3, 'b': 0.5, 'c': 0.2},
                                               stats_report=sensitivity._sensitivity_stats_report(
                                                   sensitivities={'a': 0.3, 'b': 0.5, 'c': 0.2}))
        missing = SensitivityOfFeaturesReport(method=SensitivityTypes.missing,
                                              sensitivities={'a': 0.3, 'b': 0.5, 'c': 0.2},
                                              stats_report=sensitivity._sensitivity_stats_report(
                                                  sensitivities={'a': 0.3, 'b': 0.5, 'c': 0.2}))

        self.assertEqual(sensitivity._imputation_score(
            shuffled=shuffled, missing=missing), 0)

        shuffled = SensitivityOfFeaturesReport(method=SensitivityTypes.shuffled, sensitivities={'a': 1, 'b': 0, 'c': 0},
                                               stats_report=sensitivity._sensitivity_stats_report(
                                                   sensitivities={'a': 1, 'b': 0, 'c': 0}))
        missing = SensitivityOfFeaturesReport(method=SensitivityTypes.missing, sensitivities={'a': 0, 'b': 1, 'c': 0},
                                              stats_report=sensitivity._sensitivity_stats_report(
                                                  sensitivities={'a': 0, 'b': 1, 'c': 0}))

        self.assertEqual(sensitivity._imputation_score(
            shuffled=shuffled, missing=missing), 1)
