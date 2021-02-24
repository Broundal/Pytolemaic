import logging

import numpy
import scipy.stats

from pytolemaic.utils.constants import CLASSIFICATION
from pytolemaic.utils.dmd import DMD
from pytolemaic.analysis_logic.dataset_analysis.covriance_shift import CovarianceShift
from pytolemaic.analysis_logic.dataset_analysis.dataset_analysis_report import \
    DatasetAnalysisReport, MissingValuesReport



class DatasetAnalysis():
    def __init__(self, problem_Type, class_count_threshold=10, outliers_n_sigma=(3, 5),
                 nan_threshold_per_col=(0.1, 0.5, 0.9), nan_threshold_per_sample=(0.1, 0.5, 0.9)):
        self._problem_type = problem_Type
        self._class_count_threshold = class_count_threshold
        self._outliers_n_sigma = outliers_n_sigma
        self._nan_threshold_per_feature = nan_threshold_per_col
        self._nan_threshold_per_sample = nan_threshold_per_sample
        self._covariance_shift = None

    def count_unique_classes(self, dataset: DMD) -> dict:

        if dataset.categorical_features is None:
            logging.warning("Unable to analyze categorical features if feature types information is unavailable.")
            return {}

        if len(dataset.categorical_features) == 0:
            return {}

        out = {}
        x = numpy.zeros((dataset.n_samples, dataset.n_features + 1))
        x[:, :-1] = dataset.values
        x[:, -1] = dataset.target.ravel()

        feature_names = dataset.feature_names + ['target']

        nan_mask = numpy.zeros((dataset.n_samples, dataset.n_features + 1)).astype(bool)
        nan_mask[:, :-1] = dataset.nan_mask

        target_is_categorical = self._problem_type == CLASSIFICATION
        target_index = [dataset.n_features] if target_is_categorical else []

        categorical_features = dataset.categorical_features.tolist() + target_index

        for i in categorical_features:
            vec = x[~nan_mask[:, i], i]
            if len(vec) == 0:
                continue
            try:
                classes_, counts = numpy.unique(vec, return_counts=True)
            except:
                classes_ = set(vec)
                counts = [sum(vec == class_) for class_ in classes_]

            if i in dataset.categorical_encoding_by_icols:
                encoding = dataset.categorical_encoding_by_icols[i]
            elif i == target_index and dataset.target_encoding is not None:
                encoding = dataset.target_encoding
            else:
                encoding = None

            if encoding is not None:
                classes_ = [encoding[i] for i in classes_]

            if min(counts) <= self._class_count_threshold:
                out[feature_names[i]] = {class_: count for class_, count in zip(classes_, counts) if
                                         count <= self._class_count_threshold}

        return out

    def count_outliers(self, dataset: DMD) -> dict:
        if dataset.categorical_features is None:
            logging.warning(
                "Unable to analyze numerical features for outlier if feature types information is unavailable.")
            return {}

        if len(dataset.categorical_features) == dataset.n_features:
            return {}

        out = {}
        x = numpy.zeros((dataset.n_samples, dataset.n_features + 1))
        x[:, :-1] = dataset.values
        x[:, -1] = dataset.target.ravel()
        feature_names = dataset.feature_names + ['target']

        nan_mask = numpy.zeros((dataset.n_samples, dataset.n_features + 1)).astype(bool)
        nan_mask[:, :-1] = dataset.nan_mask

        target_is_categorical = self._problem_type == CLASSIFICATION
        target_index = {dataset.n_features} if target_is_categorical else set()
        numerical_features = set(numpy.arange(dataset.n_features + 1)) - set(
            dataset.categorical_features) - target_index

        expected_outliers_per_sigma = {sigma: int(0.75 + (1 - scipy.stats.norm.cdf(sigma)) * dataset.n_samples)
                                       for sigma in self._outliers_n_sigma}

        for i in numerical_features:
            vec = x[~nan_mask[:, i], i]
            if len(vec) == 0:
                continue

            for sigma in self._outliers_n_sigma:
                mean, std = self._calc_mean_and_std(sigma, vec)
                mn, mx = numpy.min(vec), numpy.max(vec)

                expected_outliers = expected_outliers_per_sigma[sigma]
                n_outliers = numpy.sum(vec > mean + sigma * std) + numpy.sum(vec < mean - sigma * std)

                if n_outliers > 2 * expected_outliers:
                    if feature_names[i] not in out:
                        out[feature_names[i]] = {}
                    out[feature_names[i]]['{}-sigma'.format(sigma)] = dict(n_outliers=n_outliers,
                                                                           n_sigma=sigma,
                                                                           expected_outliers=expected_outliers,
                                                                           mean=mean,
                                                                           std=std,
                                                                           min=mn,
                                                                           max=mx)
                    # n_sigma = '{}-sigma'.format(sigma)
                    # if n_sigma not in out:
                    #     out[n_sigma] = {}
                    # out[n_sigma][feature_names[i]] = n_outliers

        return out

    def _calc_mean_and_std(self, sigma, vec):
        # Outliers will strongly affect the mean and variance by which they are defined.
        # Thus, we throw away outliers according to sigma+2 before calculating the mean and std.
        # We repeat the process iteratively until converged.

        prev_indices = []
        indices = numpy.arange(len(vec))
        mean, std = 0, 0
        while len(prev_indices) != len(indices):
            vec_for_stats = vec[indices]
            std = numpy.std(vec_for_stats)
            mean = numpy.mean(vec_for_stats)
            outliers = ((vec_for_stats > mean + (2 + sigma) * std) + (
                    vec_for_stats < mean - (2 + sigma) * std)).astype(bool)
            prev_indices = indices
            indices = indices[~outliers]
        return mean, std

    def count_missing_values(self, dataset: DMD):
        nan_mask = dataset.nan_mask

        # features
        nan_cols = {}
        for nan_threshold in self._nan_threshold_per_feature:
            nan_ratios = numpy.sum(nan_mask, axis=0) / dataset.n_samples
            is_off_limit = nan_ratios >= nan_threshold
            nan_cols[nan_threshold] = dict(
                zip(numpy.array(dataset.feature_names)[is_off_limit], nan_ratios[is_off_limit]))

        # samples
        nan_rows = {}
        for nan_threshold in self._nan_threshold_per_sample:
            nan_ratios = numpy.sum(nan_mask, axis=1) / dataset.n_features
            is_off_limit = nan_ratios >= nan_threshold
            nan_rows[nan_threshold] = dict(zip(numpy.arange(dataset.n_samples)[is_off_limit], nan_ratios[is_off_limit]))

        return nan_cols, nan_rows

    @property
    def covariance_shift(self):
        return self._covariance_shift

    def dataset_analysis_report(self, train: DMD, test:DMD=None) -> DatasetAnalysisReport:
        nan_counts_features, nan_counts_samples = self.count_missing_values(dataset=train)

        self.count_unique_classes(dataset=train)
        self.count_outliers(dataset=train)
        if test is None:
            covariance_report = None
        else:
            self._covariance_shift = CovarianceShift()
            self._covariance_shift.calc_covariance_shift(dmd_train=train, dmd_test=test)
            covariance_report = self._covariance_shift.covariance_shift_report()

        return DatasetAnalysisReport(
            class_counts=self.count_unique_classes(dataset=train),
            outliers_count=self.count_outliers(dataset=train),
            missing_values_report=MissingValuesReport(nan_counts_features=nan_counts_features,
                                                      nan_counts_samples=nan_counts_samples),
            covariance_shift_report=covariance_report
        )
