import itertools

import numpy
from matplotlib import pyplot as plt

from pytolemaic.utils.base_report import Report


class MissingValuesReport(Report):
    def __init__(self, nan_counts_features, nan_counts_samples, n_samples_to_show=10):
        self._nan_counts_features = nan_counts_features
        self._nan_counts_samples = nan_counts_samples
        self._n_samples_to_show = n_samples_to_show

    @property
    def nan_counts_features(self):
        return self._nan_counts_features

    @property
    def nan_counts_samples(self):
        return self._nan_counts_samples

    def to_dict(self, printable=False):
        out = dict(per_feature={}, per_sample={})
        for nan_ratio, values in self.nan_counts_features.items():
            out['per_feature']['over {}%'.format(nan_ratio * 100)] = dict(count=len(values),
                                                                          threshold=nan_ratio,
                                                                          features=values)

        for nan_ratio, values in self.nan_counts_samples.items():
            if printable:
                n_samples_to_show = min(self._n_samples_to_show, len(values))
                values_in_dict = dict(
                    sorted(values.items(), key=lambda kv: kv[1], reverse=True)[:n_samples_to_show])
            else:
                values_in_dict = values

            out['per_sample']['over {}%'.format(nan_ratio * 100)] = dict(count=len(values),
                                                                         threshold=nan_ratio,
                                                                         samples=values_in_dict)

        # out['per_feature'] = {'scheme': {'threshold': {'count': 'int', 'features': {'feature': 'ratio'}}},
        #                       'values': out['per_feature']}
        # out['per_sample'] = {'scheme': '{threshold: {count: int, sampless: {samples_index: ratio}}} }',
        #                      'values': out['per_sample']}
        return self._printable_dict(out, printable=printable)

    @classmethod
    def to_dict_meaning(cls):
        return dict(per_feature={'over xx%': {'count': 'Number of features with more than xx% missing values',
                                              'threshold': 'threshold for missing values ratio',
                                              'features': 'dict of the form {feature_name : ratio_of_missing_values}.'}},
                    per_sample={'over xx%': {'count': 'Number of samples with more than xx% missing values',
                                             'threshold': 'threshold for missing values ratio',
                                             'samples': 'dict of the form {sample_index : ratio_of_missing_values}.'}})

    def plot(self, n_features_to_plot=10):
        as_str = lambda k: [str(s) for s in k]
        need_plot_for_features = any([len(v) > 0 for v in self.nan_counts_features.values()])
        need_plot_for_samples = any([len(v) > 0 for v in self.nan_counts_samples.values()])
        n_plots = int(need_plot_for_features) + int(need_plot_for_samples)
        if n_plots == 0:
            return

        # list of tuples
        f11 = sorted(self.nan_counts_features.items(), key=lambda kv: kv[0])  # sort by nan_ratio
        f21 = sorted(self.nan_counts_samples.items(), key=lambda kv: kv[0])  # sort by nan_ratio

        # list of tuples
        f12 = sorted(self.nan_counts_features[f11[0][0]].items(), key=lambda kv: kv[1], reverse=True)
        f22 = sorted(self.nan_counts_samples[f21[0][0]].items(), key=lambda kv: kv[1], reverse=True)
        f12 = f12[:min(len(f12), n_features_to_plot)]
        f22 = f22[:min(len(f22), n_features_to_plot)]

        if n_plots == 1:
            fig1, axs = plt.subplots(1, 2, figsize=(10, 5))

            ax11, ax12, ax21, ax22 = [None] * 4
            if need_plot_for_features:
                ax11, ax12 = axs
            else:
                ax21, ax22 = axs
        else:
            fig1, (ax11, ax12) = plt.subplots(1, 2, figsize=(10, 5))
            fig2, (ax21, ax22) = plt.subplots(1, 2, figsize=(10, 5))

        if ax11 is not None:
            keys, values = zip(*f11)
            x = list(reversed(keys))
            y = list(reversed([len(v) for v in values]))
            ax11.bar(x, y, width=0.2)
            ax11.set(
                title='Features with missing values above threshold',
                xlabel='Missing values threshold ',
                ylabel='Number of features',
                xticks=x)

            keys, values = zip(*f12)
            ax12.barh(as_str(reversed(keys)), list(reversed(values)))
            ax12.set(
                title='Features with highest missing ratio',
                xlabel='Missing values ratio',
                yticks=as_str(reversed(keys)))

        if ax22 is not None:
            keys, values = zip(*f21)
            x = list(reversed(keys))
            y = list(reversed([len(v) for v in values]))
            ax21.bar(x, y, width=0.2)
            ax21.set(
                title='Samples with missing values above threshold',
                xlabel='Missing values threshold ',
                ylabel='Number of samples',
                xticks=x)

            keys, values = zip(*f22)
            ax22.barh(as_str(reversed(keys)), list(reversed(values)))
            ax22.set(
                title='Samples with highest missing ratio',
                xlabel='Missing values ratio',
                yticks=as_str(reversed(keys)))

        plt.tight_layout()
        plt.draw()

    def insights(self):
        insights = []
        info = self.to_dict(printable=True)

        lvl1 = 0.1
        lvl2 = 0.5
        lvl3 = 0.9

        counts = [info['per_feature'][key]['count'] for key in info['per_feature']]
        if len(counts) > 0 and max(counts) > 0:
            max_key_features = max([key for key, values in info['per_feature'].items() if values['count'] > 0])
            missing_values = info['per_feature'][max_key_features]
            threshold = missing_values['threshold']

            sentence = "missing values ({}) were found in {} features".format(max_key_features, missing_values['count'])
            if threshold < lvl1:
                insights.append("Some {}.".format(sentence))
            elif threshold < lvl2:
                insights.append("Significant number of {}.".format(sentence))
            elif threshold < lvl3:
                insights.append(
                    "High number of {}. Check out these features:\n{}".format(sentence, missing_values['features']))
            else:
                insights.append("Extremely high number of {}. Check out these features:\n{}".format(sentence,
                                                                                                    missing_values[
                                                                                                        'features']))

        lvl1 = 0.1
        lvl2 = 0.5
        lvl3 = 0.9

        counts = [info['per_sample'][key]['count'] for key in info['per_sample']]
        if len(counts) > 0 and max(counts) > 0:
            max_key_samples = max([key for key, values in info['per_sample'].items() if values['count'] > 0])
            missing_values = info['per_sample'][max_key_samples]
            threshold = missing_values['threshold']

            sentence = "missing values ({}) were found in {} samples".format(max_key_samples, missing_values['count'])
            if threshold < lvl1:
                insights.append("Some {}.".format(sentence))
            elif threshold < lvl2:
                insights.append("Significant number of {}.".format(sentence))
            elif threshold < lvl3:
                insights.append(
                    "High number of {}. Check out these samples:\n{}".format(sentence, missing_values['samples']))
            else:
                insights.append("Extremely high number of {}. Check out these samples:\n{}".format(sentence,
                                                                                                   missing_values[
                                                                                                       'samples']))

        return self._add_cls_name_prefix(insights)


class DatasetAnalysisReport(Report):
    def __init__(self, class_counts, outliers_count, missing_values_report: MissingValuesReport):
        self._class_counts = class_counts
        self._outliers_count = outliers_count
        self._missing_values_report = missing_values_report

    @property
    def class_counts(self) -> dict:
        return self._class_counts

    @property
    def outliers_count(self) -> dict:
        return self._outliers_count

    @property
    def missing_values_report(self):
        return self._missing_values_report

    def to_dict(self, printable=False):
        out = dict(few_class_representatives=self.class_counts,
                   outliers_count=self.outliers_count,
                   missing_values=self.missing_values_report.to_dict())
        return self._printable_dict(out, printable=printable)

    @classmethod
    def to_dict_meaning(cls):
        return dict(
            few_class_representatives='{feature_name: {class: instances_count}} : listing categorical features that has at least 1 class which is under-represented (less than 10 instances).',
            outliers_count='{feature_name: {n-sigma: outliers_info}} : listing numerical features that has more outliers than is expected with respect to n-sigma. '
                           'E.g. for 3-sigma we expect ceil(0.0027 x n_samples) outliers.',
            missing_values=MissingValuesReport.to_dict_meaning())

    def _plot_class_counts(self):
        if len(self.class_counts) == 0:
            return

        plt.figure()
        ax = plt.subplot()
        features = sorted(self.class_counts.keys())
        n_low_count_classes = [len(self.class_counts[feature]) for feature in features]

        x = [str(k) for k in reversed(features)]
        y = list(reversed(n_low_count_classes))
        plt.barh(x, y)
        ax.set(
            title='Features with under represented classes',
            xlabel='Number of under represented classes',
            ylabel='Features',
            yticks=x)

    def _plot_outlier_counts(self):
        if len(self.outliers_count) == 0:
            return

        plt.figure()
        ax = plt.subplot()
        sigmas = set(list(itertools.chain(*[v.keys() for v in self.outliers_count.values()])))

        features = sorted([str(k) for k in self.outliers_count.keys()], reverse=True)
        for sigma in sorted(sigmas):
            fv = [(feature, values[sigma]['n_outliers']) for feature, values in self.outliers_count.items() if
                  sigma in values]

            x, y = zip(*fv)

            x = [str(k) for k in list(reversed(x))]
            y = list(reversed(y))
            plt.barh(x, y)
        ax.set(
            title='Features with outliers',
            xlabel='Number of outliers',
            ylabel='Features',
            yticks=features)
        ax.legend(sorted(sigmas))

    def plot(self):
        self.missing_values_report.plot()
        self._plot_class_counts()
        self._plot_outlier_counts()

    def _class_count_insights(self):
        insights = []
        for feature_name, class_counts in self.class_counts.items():
            if len(class_counts) == 1:
                sentence = "Feature '{}' contains a class with few representatives".format(feature_name)
            else:
                sentence = "Feature '{}' contains {} classes with few representatives - {} samples in total. " \
                           "With so few samples in each class there would be little to no learning.\n\tConsider merging all classes into a single 'other' class" \
                    .format(feature_name, len(class_counts), sum(class_counts.values()))

            list_of_classes = ["class '{}' has only {} representatives".format(key, value) for key, value in
                               class_counts.items()]
            if len(class_counts) == 1:
                list_of_classes = ": " + list_of_classes[0]
            else:
                n_lines_to_show = 5
                if len(list_of_classes) > n_lines_to_show:
                    suffix = '\n\t... {} more.'.format(len(list_of_classes) - n_lines_to_show)
                else:
                    suffix = ''
                list_of_classes = ". List of classes:\n\t" + ",\n\t".join(list_of_classes[:n_lines_to_show]) + suffix

            insights.append("{}{}".format(sentence, list_of_classes))

        return self._add_cls_name_prefix(insights)

    def _outlier_counts_insights(self):
        insights = []
        for feature_name, outliers_info in self.outliers_count.items():
            max_n_sigma = max(outliers_info.keys())

            n_sigma_info = outliers_info[max_n_sigma]
            n_sigma_range = "[{:.3g},{:.3g}]".format(
                n_sigma_info['mean'] - n_sigma_info['n_sigma'] * n_sigma_info['std'],
                n_sigma_info['mean'] + n_sigma_info['n_sigma'] * n_sigma_info['std'], )
            sentence = "Feature '{}' has {} outliers with respect to {} - a {} range." \
                .format(feature_name, n_sigma_info['n_outliers'], n_sigma_range, max_n_sigma)
            insights.append(sentence)

            log10_plus_one = numpy.log10(numpy.round(n_sigma_info['max'] + 1))

            if n_sigma_info['max'] >= 99 and log10_plus_one == int(log10_plus_one):
                insights.append("Feature '{}' has a maximal value of {} which doesn't seem to be a legit value".format(
                    feature_name, n_sigma_info['max']))

        return self._add_cls_name_prefix(insights)

    def insights(self):
        return list(itertools.chain(self.missing_values_report.insights(),
                                    self._outlier_counts_insights(),
                                    self._class_count_insights()))
