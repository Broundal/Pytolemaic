import unittest

import numpy
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from pytolemaic.analysis_logic.model_analysis.sensitivity.sensitivity import \
    SensitivityAnalysis
from pytolemaic.utils.dmd import DMD
from pytolemaic.utils.general import GeneralUtils
from pytolemaic.utils.metrics import Metrics
from pytolemaic.utils.reports import ReportSensitivity


class TestSensitivity(unittest.TestCase):

    def get_data(self, is_classification, seed=0):
        rs = numpy.random.RandomState(seed)
        x = rs.rand(10000, 10)
        x[:, 1] = 0
        # 1st is double importance, 2nd has no importance
        y = numpy.sum(x, axis=1) + 2 * x[:, 0]
        if is_classification:
            y = numpy.round(y, 0).astype(int)
        return DMD(x=x, y=y,
                   columns_meta={DMD.FEATURE_NAMES: ['f_' + str(k) for k in
                                                     range(x.shape[1])]})

    def get_model(self, is_classification):
        if is_classification:
            estimator = DecisionTreeClassifier
        else:
            estimator = DecisionTreeRegressor

        model = model = GeneralUtils.simple_imputation_pipeline(
            estimator(random_state=0))

        return model

    def test_sensitivity_raw_perturb_classification(self,
                                                    is_classification=True):
        sensitivity = SensitivityAnalysis()
        model = self.get_model(is_classification)

        train = self.get_data(is_classification)
        model.fit(train.values, train.target.ravel())

        test = self.get_data(is_classification, seed=1)

        raw_scores = sensitivity.sensitivity_analysis(model=model,
                                                      metric=Metrics.recall.name,
                                                      dmd_test=test,
                                                      method='perturb',
                                                      raw_scores=True)

        self.assertTrue(isinstance(raw_scores, dict))
        self.assertEqual(len(raw_scores), test.n_features)
        self.assertLessEqual(raw_scores['f_0'], 0.5)
        self.assertEqual(raw_scores['f_1'], 1.0)
        self.assertLessEqual(max([v for v in raw_scores.values()]), 1.0)

        scores = [v for k, v in raw_scores.items() if k not in ['f_0', 'f_1']]
        self.assertLessEqual(numpy.std(scores), 0.05)

    def test_sensitivity_raw_missing_regression(self,
                                                is_classification=False):
        sensitivity = SensitivityAnalysis()
        model = self.get_model(is_classification)

        train = self.get_data(is_classification)
        model.fit(train.values, train.target.ravel())

        test = self.get_data(is_classification, seed=1)

        raw_scores = sensitivity.sensitivity_analysis(model=model,
                                                      metric=Metrics.mae.name,
                                                      dmd_test=test,
                                                      method='missing',
                                                      raw_scores=True)

        self.assertTrue(isinstance(raw_scores, dict))
        self.assertEqual(len(raw_scores), test.n_features)
        self.assertEqual(raw_scores['f_1'], 0)
        self.assertGreaterEqual(raw_scores['f_0'], 0.8)
        self.assertGreaterEqual(min([v for v in raw_scores.values()]), 0)

        scores = [v for k, v in raw_scores.items() if k not in ['f_0', 'f_1']]
        self.assertLessEqual(numpy.std(scores), 0.05)

    def test_sensitivity_impact_regression(self, is_classification=False):
        sensitivity = SensitivityAnalysis()
        model = self.get_model(is_classification)

        train = self.get_data(is_classification)
        model.fit(train.values, train.target.ravel())

        test = self.get_data(is_classification, seed=1)

        scores = sensitivity.sensitivity_analysis(model=model,
                                                  metric=Metrics.mae.name,
                                                  dmd_test=test,
                                                  method='missing',
                                                  raw_scores=False)

        print(scores)
        self.assertTrue(isinstance(scores, dict))
        self.assertEqual(len(scores), test.n_features)
        self.assertGreaterEqual(numpy.round(sum([v for v in scores.values()]), 6), 1-1e-5)
        self.assertEqual(scores['f_1'], 0)
        self.assertGreaterEqual(scores['f_0'], 2 / len(scores))

        o_scores = [v for k, v in scores.items() if k not in ['f_0', 'f_1']]
        self.assertLessEqual(numpy.std(o_scores), 0.05)
        self.assertGreaterEqual(scores['f_0'], numpy.mean(o_scores))

    def test_sensitivity_impact_classification(self, is_classification=True):
        sensitivity = SensitivityAnalysis()
        model = self.get_model(is_classification)

        train = self.get_data(is_classification)
        model.fit(train.values, train.target.ravel())

        test = self.get_data(is_classification, seed=1)

        scores = sensitivity.sensitivity_analysis(model=model,
                                                  metric=Metrics.mae.name,
                                                  dmd_test=test,
                                                  method='missing',
                                                  raw_scores=False)

        print(scores)
        self.assertTrue(isinstance(scores, dict))
        self.assertEqual(len(scores), test.n_features)
        self.assertGreaterEqual(numpy.round(sum([v for v in scores.values()]), 6), 1-1e-5)
        self.assertEqual(scores['f_1'], 0)
        self.assertGreaterEqual(scores['f_0'], 2 / len(scores))

        o_scores = [v for k, v in scores.items() if k not in ['f_0', 'f_1']]
        self.assertLessEqual(numpy.std(o_scores), 0.05)
        self.assertGreaterEqual(scores['f_0'], numpy.mean(o_scores))

    def test_sensitivity_functions(self, is_classification=False):
        sensitivity = SensitivityAnalysis()
        model = self.get_model(is_classification)

        train = self.get_data(is_classification)
        model.fit(train.values, train.target.ravel())

        test = self.get_data(is_classification, seed=1)

        perturb = sensitivity.sensitivity_analysis(model=model,
                                                   metric=Metrics.mae.name,
                                                   dmd_test=test,
                                                   method='perturb',
                                                   raw_scores=False)

        missing = sensitivity.sensitivity_analysis(model=model,
                                                   metric=Metrics.mae.name,
                                                   dmd_test=test,
                                                   method='missing',
                                                   raw_scores=False)

        perturbed_sensitivity_meta = sensitivity._sensitivity_meta(perturb)
        n_features = perturbed_sensitivity_meta[ReportSensitivity.N_FEATURES]
        n_zero = perturbed_sensitivity_meta[ReportSensitivity.N_ZERO]
        n_low = perturbed_sensitivity_meta[ReportSensitivity.N_LOW]

        self.assertTrue(len(perturbed_sensitivity_meta) > 0)

        leakage_score = sensitivity._leakage(
            n_features=n_features,
            n_zero=n_zero)

        self.assertGreater(leakage_score, 0)
        self.assertLessEqual(leakage_score, 1)


        overfit_score = sensitivity._overfit(
            n_features=n_features,
            n_low=n_low,
            n_zero=n_zero)

        self.assertGreater(overfit_score, 0)
        self.assertLessEqual(overfit_score, 1)

        imputation_score = sensitivity._imputation_score(shuffled=perturb,
                                                         missing=missing)
        self.assertGreaterEqual(imputation_score, 0)
        self.assertLessEqual(imputation_score, 1)

        scores = sensitivity._sensitivity_scores(
            perturbed_sensitivity=perturb,
            perturbed_sensitivity_meta=perturbed_sensitivity_meta,
            missing_sensitivity=missing)
        self.assertTrue(len(scores) >= 3)
