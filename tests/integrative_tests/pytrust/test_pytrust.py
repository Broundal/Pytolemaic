import unittest

import numpy
import pandas
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from pytolemaic.pytrust import SklearnTrustBase
from pytolemaic.utils.constants import CLASSIFICATION, REGRESSION
from pytolemaic.utils.dmd import DMD
from pytolemaic.utils.general import GeneralUtils
from pytolemaic.utils.metrics import Metrics
from pytolemaic.utils.report_keys import ReportSensitivity, ReportScoring
from pytolemaic.utils.report import Report


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
        self.assertTrue(isinstance(sensitivity_report, Report))
        for key in ReportSensitivity.keys(members=True):
            print(key)
            self.assertTrue(sensitivity_report.get(key) is not None)

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
        self.assertEqual(sensitivity_report2.report, sensitivity_report.report)

    def test_pytrust_scoring_report(self):

        pytrust = self.get_pytrust(is_classification=True)

        scoring_report = pytrust.scoring_report()

        for metric in Metrics.supported_metrics().values():
            if metric.ptype == CLASSIFICATION:
                metric_report = scoring_report.get(metric.name)
                score_value = metric_report.get(ReportScoring.SCORE_VALUE)
                ci_low = metric_report.get(ReportScoring.CI_LOW)
                ci_high = metric_report.get(ReportScoring.CI_HIGH)

                self.assertTrue(ci_low < score_value < ci_high)

        pytrust = self.get_pytrust(is_classification=False)

        scoring_report = pytrust.scoring_report()
        for metric in Metrics.supported_metrics().values():
            if metric.ptype == REGRESSION:
                metric_report = scoring_report.get(metric.name)
                score_value = metric_report.get(ReportScoring.SCORE_VALUE)
                ci_low = metric_report.get(ReportScoring.CI_LOW)
                ci_high = metric_report.get(ReportScoring.CI_HIGH)

                self.assertTrue(ci_low < score_value < ci_high)
