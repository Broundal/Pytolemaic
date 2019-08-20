import unittest

import numpy
import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from pytolemaic.analysis_logic.prediction_analysis.prediction_uncertainty.uncertainty_model import \
    UncertaintyModelClassifier
from pytolemaic.utils.dmd import DMD
from pytolemaic.utils.general import GeneralUtils


class TestPredictionsUncertainty(unittest.TestCase):

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
            estimator(random_state=0, n_jobs=-1))

        return model

    def test_classification_confidence(self, is_classification=True,
                                       method='confidence'):

        model = self.get_model(is_classification)

        train = self.get_data(is_classification)
        model.fit(train.values, train.target)

        test = self.get_data(is_classification, seed=1)

        uncertainty_model = UncertaintyModelClassifier(model=model,
                                                       uncertainty_method=method)

        uncertainty_model.fit(dmd_test=test)

        new_data = self.get_data(is_classification, seed=2)
        yp = uncertainty_model.predict_proba(new_data)
        uncertainty = uncertainty_model.uncertainty(new_data)

        print(sklearn.metrics.recall_score(y_true=new_data.target, y_pred=yp))
        print(sklearn.metrics.recall_score(y_true=new_data.target, y_pred=yp,
                                           sample_weight=uncertainty))
