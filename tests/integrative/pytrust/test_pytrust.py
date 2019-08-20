import unittest

import numpy
import pandas
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer

from pytolemaic.pytrust import SklearnTrustBase
from pytolemaic.utils.dmd import DMD
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

        estimators = []
        estimators.append(('Imputer', Imputer()))
        # estimators.append(('nn', MLPRegressor(hidden_layer_sizes=(5,), max_iter=1000)))
        estimators.append(('Estimator', estimator(random_state=0)))
        model = Pipeline(estimators)

        return model

    def test_pytrust_sensitivity_classification(self):
        is_classification = True
        metric = Metrics.recall.name

        model = self.get_model(is_classification)

        train = self.get_data(is_classification)
        model.fit(train.values, train.target)

        test = self.get_data(is_classification, seed=1)
        pytrust = SklearnTrustBase(
            model=model,
            Xtrain=train.values, Ytrain=train.target,
            Xtest=test.values, Ytest=test.target,
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
            Xtrain=pandas.DataFrame(train.values),
            Ytrain=pandas.DataFrame(train.target),
            Xtest=pandas.DataFrame(test.values),
            Ytest=pandas.DataFrame(test.target),
            sample_meta_train=None, sample_meta_test=None,
            columns_meta={DMD.FEATURE_NAMES: ['f' + str(k) for k in
                                              range(train.n_features)]},
            metric=metric)

        sensitivity_report2 = pytrust.sensitivity_report()
        print(sensitivity_report)
        self.maxDiff = None
        self.assertEqual(sensitivity_report2, sensitivity_report)
