import itertools
import logging

import numpy
import pandas

from pytolemaic.utils.dmd import DMD

try:
    from pytolemaic.analysis_logic.model_analysis.sensitivity.sensitivity_reports import SensitivityOfFeaturesReport
except:
    logging.warning('issue with import of SensitivityOfFeaturesReport')
    pass

from pytolemaic.utils.base_report import Report
from matplotlib import pyplot as plt

class CovarianceShiftReport(Report):
    def __init__(self, covariance_shift:float, sensitivity_report:SensitivityOfFeaturesReport=None, train:DMD=None, test:DMD=None,
                 medium_lvl=0.7, high_lvl=0.95):
        self.medium_lvl = medium_lvl
        self.high_lvl = high_lvl
        self.covariance_shift = covariance_shift
        self.sensitivity = sensitivity_report

        # train and test data is used for distribution plots
        self.train = train
        self.test = test

    def _features_to_look_at_msg(self):
        if self.sensitivity is None:
            return ''
        else:
            return 'Compare distributions for following features: {}'.format(self.sensitivity.most_important_features())

    def covariance_shift_insights(self):
        insights = []
        covariance_shift = numpy.round(self.covariance_shift, 2)
        if covariance_shift < self.medium_lvl:
            return  insights # ok

        features_to_look_at_msg = self._features_to_look_at_msg()

        if covariance_shift < self.high_lvl:
            insights.append('Covariance shift ={:.2g} is not negligible, there may be some issue with train/test distribution. {}'
                            .format(self.covariance_shift, features_to_look_at_msg))
        else:
            insights.append("Covariance shift ={:.2g} is high! Double check the way you've defined the test set! {}"
                            .format(self.covariance_shift, features_to_look_at_msg))

        return insights

    def to_dict(self, printable=False):
        out = dict(covariance_shift=self.covariance_shift)
        if self.sensitivity is not None:
            out.update(dict(covariance_shift_sensitivity=self.sensitivity.to_dict(printable=printable)))
        return self._printable_dict(out, printable=printable)

    @classmethod
    def to_dict_meaning(cls):
        return dict(covariance_shift = "Measure whether the test and train comes from same distribution. "
                                       "High values (max 1) means the test and train come from different distribution. "
                                       "Low score (min 0) means they come from same distribution (=is ok).",
                    covariance_shift_sensitivity = "Sensitivity report for a model which classify samples to train/test sets. "
                                                   "This report is automatically generated only if covariance shift is high. "
                                                   "To generate it manually, see CovarianceShiftCalculator.")

    def plot(self):
        if self.sensitivity is not None:
            fig, ax = plt.subplots(1,1, figsize=(8,10))
            self.sensitivity.plot_sorted_sensitivities(ax=ax, n_features_to_plot=10)
            ax.set(
                title='Feature Sensitivity for Covariance Shift model',
                xlabel='Sensitivity value')

            if self.train is not None and self.test is not None:
                features_to_look_at = self.sensitivity.most_important_features(n_features=3)
                train, _ = self.train.to_df()
                test, _ = self.test.to_df()

                features_to_look_at = [f for f in features_to_look_at if f in train.columns]
                if len(features_to_look_at) > 0:
                    fig, axs = plt.subplots(len(features_to_look_at), 1, figsize=(10, 10), sharex=True)
                for i, feature in enumerate(features_to_look_at):

                    tmp = pandas.DataFrame({'Distribution in train set' : train[feature],
                                            'Distribution in test set': test[feature]})

                    for col in tmp.columns:
                        axs[i].hist(tmp[col], alpha=0.5, label=col)
                    axs[i].legend()
                    axs[i].set_title('Feature "{}"'.format(feature))





    def insights(self):

        return self._add_cls_name_prefix(
            itertools.chain(self.covariance_shift_insights()))




