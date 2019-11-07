import unittest

import numpy
import pandas
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from pytolemaic.pytrust import SklearnTrustBase
from pytolemaic.utils.constants import CLASSIFICATION, REGRESSION
from pytolemaic.utils.dmd import DMD
from pytolemaic.utils.general import GeneralUtils
from pytolemaic.utils.metrics import Metrics


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
            estimator = RandomForestClassifier
        else:
            estimator = RandomForestRegressor

        model = GeneralUtils.simple_imputation_pipeline(
            estimator(random_state=0, n_estimators=3))

        return model

    def get_pytrust(self, is_classification):
        if is_classification:
            metric = Metrics.recall.name
        else:
            metric = Metrics.mae.name

        model = self.get_model(is_classification)

        train = self.get_data(is_classification)
        model.fit(train.values, train.target.ravel())

        test = self.get_data(is_classification, seed=1)
        pytrust = SklearnTrustBase(
            model=model,
            xtrain=train.values, ytrain=train.target,
            xtest=test.values, ytest=test.target,
            sample_meta_train=None, sample_meta_test=None,
            columns_meta={DMD.FEATURE_NAMES: ['f' + str(k) for k in
                                              range(train.n_features)]},
            metric=metric)

        return pytrust

    def test_pytrust_sensitivity_classification(self):
        is_classification = True
        metric = Metrics.recall.name

        model = self.get_model(is_classification)

        train = self.get_data(is_classification)
        model.fit(train.values, train.target.ravel())

        test = self.get_data(is_classification, seed=1)
        pytrust = SklearnTrustBase(
            model=model,
            xtrain=train.values, ytrain=train.target,
            xtest=test.values, ytest=test.target,
            sample_meta_train=None, sample_meta_test=None,
            columns_meta={DMD.FEATURE_NAMES: ['f' + str(k) for k in
                                              range(train.n_features)]},
            metric=metric)

        sensitivity_report = pytrust.sensitivity_report()
        print(sensitivity_report)
        self.assertTrue(isinstance(sensitivity_report, dict))
        for key in ['perturbed_sensitivity', 'missing_sensitivity',
                    'perturbed_sensitivity_meta', 'missing_sensitivity_meta',
                    'perturbed_sensitivity_scores']:
            self.assertIn(key, sensitivity_report)
            self.assertTrue(sensitivity_report.get(key, None))

        pytrust = SklearnTrustBase(
            model=model,
            xtrain=pandas.DataFrame(train.values),
            ytrain=pandas.DataFrame(train.target),
            xtest=pandas.DataFrame(test.values),
            ytest=pandas.DataFrame(test.target),
            sample_meta_train=None, sample_meta_test=None,
            columns_meta={DMD.FEATURE_NAMES: ['f' + str(k) for k in
                                              range(train.n_features)]},
            metric=metric)

        sensitivity_report2 = pytrust.sensitivity_report()
        print(sensitivity_report)
        self.maxDiff = None
        self.assertEqual(sensitivity_report2, sensitivity_report)

    def test_pytrust_scoring_report(self):

        pytrust = self.get_pytrust(is_classification=True)

        report = pytrust.scoring_report()['Score']
        for metric in Metrics.supported_metrics().values():
            if metric.ptype == CLASSIFICATION:
                self.assertIn(metric.name, report)
                self.assertEqual(len(report[metric.name]), 3)
                low, score, high = report[metric.name]['ci_low'], \
                                   report[metric.name]['value'], \
                                   report[metric.name]['ci_high'],
                self.assertTrue(low < score < high)

        pytrust = self.get_pytrust(is_classification=False)

        report = pytrust.scoring_report()['Score']
        for metric in Metrics.supported_metrics().values():
            if metric.ptype == REGRESSION:
                self.assertIn(metric.name, report)
                self.assertEqual(len(report[metric.name]), 3)
                low, score, high = report[metric.name]['ci_low'], \
                                   report[metric.name]['value'], \
                                   report[metric.name]['ci_high'],
                self.assertTrue(low < score < high)
