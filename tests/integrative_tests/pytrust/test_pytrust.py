import unittest
from pprint import pprint

import numpy
import pandas
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from pytolemaic.analysis_logic.model_analysis.sensitivity.sensitivity_reports import SensitivityFullReport
from pytolemaic.pytrust import PyTrust
from pytolemaic.utils.constants import CLASSIFICATION, REGRESSION
from pytolemaic.utils.dmd import DMD
from pytolemaic.utils.general import GeneralUtils
from pytolemaic.utils.metrics import Metrics


class TestSensitivity(unittest.TestCase):

    def get_data(self, is_classification, seed=0):
        rs = numpy.random.RandomState(seed)
        x = rs.rand(200, 3)
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
        pytrust = PyTrust(
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
        pytrust = PyTrust(
            model=model,
            xtrain=train.values, ytrain=train.target,
            xtest=test.values, ytest=test.target,
            sample_meta_train=None, sample_meta_test=None,
            columns_meta={DMD.FEATURE_NAMES: ['f' + str(k) for k in
                                              range(train.n_features)]},
            metric=metric)

        sensitivity_report = pytrust.sensitivity_report
        print(sensitivity_report.to_dict())
        self.assertTrue(isinstance(sensitivity_report, SensitivityFullReport))
        for key, value in sensitivity_report.__dict__.items():
            if key.startswith('_'):
                continue
            print(key)
            self.assertTrue(value is not None)

        pytrust = PyTrust(
            model=model,
            xtrain=pandas.DataFrame(train.values),
            ytrain=pandas.DataFrame(train.target),
            xtest=pandas.DataFrame(test.values),
            ytest=pandas.DataFrame(test.target),
            sample_meta_train=None, sample_meta_test=None,
            columns_meta={DMD.FEATURE_NAMES: ['f' + str(k) for k in
                                              range(train.n_features)]},
            metric=metric)

        sensitivity_report2 = pytrust.sensitivity_report
        pprint(sensitivity_report.to_dict())
        self.maxDiff = None
        self.assertEqual(sensitivity_report2.shuffle_report.sensitivities, sensitivity_report.shuffle_report.sensitivities)
        self.assertEqual(sensitivity_report2.missing_report.sensitivities, sensitivity_report.missing_report.sensitivities)

    def test_pytrust_scoring_report(self):

        pytrust = self.get_pytrust(is_classification=True)

        scoring_report = pytrust.scoring_report

        for metric in Metrics.supported_metrics().values():
            if metric.ptype == CLASSIFICATION:
                metric_report = scoring_report.metric_scores[metric.name]
                score_value = metric_report.value
                ci_low = metric_report.ci_low
                ci_high = metric_report.ci_high

                self.assertTrue(ci_low < score_value < ci_high)

        pytrust = self.get_pytrust(is_classification=False)

        scoring_report = pytrust.scoring_report
        for metric in Metrics.supported_metrics().values():
            if metric.ptype == REGRESSION:
                metric_report = scoring_report.metric_scores[metric.name]
                score_value = metric_report.value
                ci_low = metric_report.ci_low
                ci_high = metric_report.ci_high

                self.assertTrue(ci_low < score_value < ci_high)
