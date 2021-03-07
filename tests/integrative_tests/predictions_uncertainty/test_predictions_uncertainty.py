import multiprocessing
import unittest
from pprint import pprint

import numpy
import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from pytolemaic.prediction_uncertainty.uncertainty_model import \
    UncertaintyModelClassifier, UncertaintyModelRegressor
from pytolemaic.utils.dmd import DMD
from pytolemaic.utils.general import GeneralUtils


class TestPredictionsUncertainty(unittest.TestCase):

    def get_data(self, is_classification, seed=0):
        rs = numpy.random.RandomState(seed)
        x = rs.rand(10000, 4)
        x[:, 1] = 0
        # 1st is double importance, 2nd has no importance
        y = numpy.sum(x, axis=1) + 2 * x[:, 0]
        y[x[:, 0] < 0.2] = x[x[:, 0] < 0.2, 2]
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
            estimator(random_state=0, n_jobs=multiprocessing.cpu_count() - 1, n_estimators=15))

        return model

    def _test(self, is_classification, method):
        model = self.get_model(is_classification)

        train = self.get_data(is_classification)
        model.fit(train.values, train.target.ravel())

        test = self.get_data(is_classification, seed=1)

        if is_classification:
            uncertainty_model = UncertaintyModelClassifier(model=model,
                                                           uncertainty_method=method)
        else:
            uncertainty_model = UncertaintyModelRegressor(model=model,
                                                          uncertainty_method=method)

        uncertainty_model.fit(dmd_test=test, n_jobs=5, n_estimators=5)

        new_data = self.get_data(is_classification, seed=2)
        yp = uncertainty_model.predict(new_data)
        uncertainty = uncertainty_model.uncertainty(new_data)

        metric = sklearn.metrics.recall_score if is_classification else sklearn.metrics.r2_score
        kwargs = {'average': 'macro'} if is_classification else {}
        base_score = metric(y_true=new_data.target,
                            y_pred=yp, **kwargs)

        p50 = numpy.percentile(numpy.unique(uncertainty), 50)

        good = (uncertainty < p50).ravel()
        subset_good_score = metric(
            y_true=new_data.target[good], y_pred=yp[good], **kwargs)

        bad = (uncertainty > p50).ravel()
        subset_bad_score = metric(
            y_true=new_data.target[bad], y_pred=yp[bad], **kwargs)

        print(subset_bad_score, base_score, subset_good_score)
        self.assertGreater(subset_good_score, base_score)
        self.assertLess(subset_bad_score, base_score)

        pprint(uncertainty_model.uncertainty_analysis(dmd_train=train))


    def test_classification_confidence(self):
        self._test(is_classification=True, method='confidence')

    def test_classification_probability(self):
        self._test(is_classification=True, method='probability')

    def test_regression_mae(self):
        self._test(is_classification=False, method='mae')

    def test_regression_rmse(self):
        self._test(is_classification=False, method='rmse')

    def test_regression_quantile(self):
        self._test(is_classification=False, method='quantile')

        # subset_bad_score = []
        # subset_good_score = []
        # for k in range(10):
        #     bad = (uncertainty > (k+1)/11).ravel()
        #     subset_bad_score.append(sklearn.metrics.recall_score(
        #         y_true=new_data.target[bad], y_pred=yp[bad], average='macro'))
        #
        #     good = (uncertainty < (k + 1) / 11).ravel()
        #     subset_good_score.append(sklearn.metrics.recall_score(
        #         y_true=new_data.target[good], y_pred=yp[good], average='macro'))
        #
        # from matplotlib import pyplot as plt
        # plt.plot(range(10), subset_bad_score,'.-r',
        #          list(reversed(range(10))), subset_good_score,'.-g')
        # plt.show()
