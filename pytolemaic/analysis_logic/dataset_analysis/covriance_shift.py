import logging
import multiprocessing

import numpy
from sklearn.ensemble import RandomForestClassifier

from pytolemaic.utils.metrics import Metrics
from pytolemaic.utils.dmd import DMD
from pytolemaic.utils.general import GeneralUtils

from pytolemaic.analysis_logic.dataset_analysis.covariance_shift_report import CovarianceShiftReport



class CovarianceShiftCalculator():

    @classmethod
    def prepare_dataset_for_score_quality(cls, dmd_train: DMD, dmd_test: DMD):
        '''

        :param dmd_train: train set
        :param dmd_test: test set
        :return: dataset with target of test/train
        '''

        dmd = DMD.concat([dmd_train, dmd_test])
        new_label = [0] * dmd_train.n_samples + [1] * dmd_test.n_samples
        dmd.set_target(new_label)

        train, test = dmd.split(ratio=dmd_test.n_samples / (dmd_train.n_samples + dmd_test.n_samples))
        return train, test

    @classmethod
    def prepare_estimator(cls, train:DMD):
        classifier= GeneralUtils.simple_imputation_pipeline(
            estimator=RandomForestClassifier(random_state=0, n_estimators=30, n_jobs=multiprocessing.cpu_count() - 1))
        classifier.fit(train.values, train.target.ravel())
        return classifier

    @classmethod
    def calc_separation_quality(cls, classifier, test: DMD):
        yp = classifier.predict_proba(test.values)

        auc = Metrics.auc.function(y_true=test.target, y_pred=yp)
        auc = numpy.clip(auc, 0.5, 1)  # auc<0.5 --> 0.5

        #  High auc --> separable --> low quality.
        #  [0 to 1] --> [0.5 to 1] --> 1-(2*[0.5 to 1]-1) --> [1 to 0]
        separation_quality = numpy.round(1 - (2 * auc - 1), 5)
        return separation_quality

    @classmethod
    def calc_convriance_shift_auc(cls, classifier, test: DMD):
        return 1-cls.calc_separation_quality(classifier, test)

    @classmethod
    def calc_covariance_shift(cls, dmd_train: DMD, dmd_test: DMD):
        '''
        :param dmd_train: train set
        :param dmd_test: test set
        :return: estimation of score quality based on similarity between train and test sets
        '''

        train, test = cls.prepare_dataset_for_score_quality(dmd_train=dmd_train, dmd_test=dmd_test)
        classifier = cls.prepare_estimator(train=train)
        return cls.calc_convriance_shift_auc(classifier=classifier, test=test)

class CovarianceShift():
    def __init__(self):
        self._separation_quality = None
        self._cov_train = None
        self._cov_test = None
        self._classifier = None
        self._sensitivity = None

        self._dmd_train = None
        self._dmd_test = None

    def calc_covariance_shift(self, dmd_train: DMD, dmd_test: DMD):
        # save data for later report
        self._dmd_train = dmd_train
        self._dmd_test = dmd_test

        # split data to new train / test setes
        self._cov_train, self._cov_test = CovarianceShiftCalculator.prepare_dataset_for_score_quality(dmd_train=dmd_train, dmd_test=dmd_test)
        self._classifier = CovarianceShiftCalculator.prepare_estimator(train=self._cov_train)
        self._covariance_shift = CovarianceShiftCalculator.calc_convriance_shift_auc(classifier=self._classifier, test=self._cov_test)

    def covariance_shift_report(self):
        medium_lvl = 0.7
        high_lvl = 0.95
        if self.covariance_shift > medium_lvl:
            sensitivity_report=self.calc_sensitivity_report()
        else:
            sensitivity_report=None

        return CovarianceShiftReport(covariance_shift=self.covariance_shift,
                                     sensitivity_report=sensitivity_report,
                                     medium_lvl=medium_lvl, high_lvl=high_lvl,
                                     train=self._dmd_train,
                                     test=self._dmd_test)

    def calc_sensitivity_report(self):
        try:
            from pytolemaic.analysis_logic.model_analysis.sensitivity.sensitivity import SensitivityAnalysis
        except:
            logging.exception("Failed to import SensitivityAnalysis")
            return None

        self._sensitivity = SensitivityAnalysis()
        sensitivity_report = self._sensitivity.sensitivity_analysis(model=self.classifier, dmd_train=self.train, dmd_test=self.test,
                                               metric=Metrics.auc.name)
        return sensitivity_report

    @property
    def sensitivity(self):
        return self._sensitivity

    @property
    def separation_quality(self):
        return 1-self._covariance_shift

    @property
    def covariance_shift(self):
        return self._covariance_shift

    @property
    def train(self):
        return self._cov_train

    @property
    def test(self):
        return self._cov_test

    @property
    def classifier(self):
        return self._classifier
