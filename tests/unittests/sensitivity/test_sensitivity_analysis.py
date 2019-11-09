import unittest

from pytolemaic.analysis_logic.model_analysis.sensitivity.sensitivity import \
    SensitivityAnalysis
from pytolemaic.utils.reports import ReportSensitivity


class TestSensitivity(unittest.TestCase):

    def test_sensitivity_meta(self):
        mock = {'a' + str(k): k for k in range(10)}

        sensitivity = SensitivityAnalysis()
        meta = sensitivity._sensitivity_meta(mock)
        print(meta)

        self.assertEqual(meta[ReportSensitivity.N_FEATURES], 10)
        self.assertEqual(meta[ReportSensitivity.N_ZERO], 1)
        self.assertEqual(meta[ReportSensitivity.N_NON_ZERO],
                         meta[ReportSensitivity.N_FEATURES] - meta[ReportSensitivity.N_ZERO])
        self.assertEqual(meta[ReportSensitivity.N_LOW], 1)

    def test_leakage(self):
        sensitivity = SensitivityAnalysis()
        self.assertEqual(sensitivity._leakage(n_features=10, n_zero=0), 0)
        self.assertEqual(sensitivity._leakage(n_features=10, n_zero=9), 1)
        self.assertGreaterEqual(sensitivity._leakage(n_features=10, n_zero=8),
                                0.8)

        print([sensitivity._leakage(n_features=10, n_zero=k) for k in
               range(10)])

    def test_overfit(self):
        sensitivity = SensitivityAnalysis()

        print([sensitivity._leakage(n_features=10, n_low=k, n_zero=5) for k in
               range(10)])

        self.assertEqual(
            sensitivity._overfit(n_features=10, n_low=0, n_zero=0), 0)
        self.assertEqual(
            sensitivity._overfit(n_features=10, n_low=5, n_zero=0), 0.5)

        self.assertGreaterEqual(
            sensitivity._overfit(n_features=10, n_low=9, n_zero=9), 0.9)

    def test_imputation(self):
        sensitivity = SensitivityAnalysis()

        shuffled = {'a': 0.3, 'b': 0.5, 'c': 0.2}
        missing = {'a': 0.3, 'b': 0.5, 'c': 0.2}

        self.assertEqual(sensitivity._imputation_score(
            shuffled=shuffled, missing=missing), 0)

        shuffled = {'a': 1, 'b': 0, 'c': 0}
        missing = {'a': 0, 'b': 1, 'c': 0}

        self.assertEqual(sensitivity._imputation_score(
            shuffled=shuffled, missing=missing), 1)
