import numpy
import pandas as pd
from matplotlib import pyplot as plt

from pytolemaic.utils.dmd import DMD
from pytolemaic.utils.base_report import Report

from pytolemaic.utils.general import get_logger

logger = get_logger(__name__)

class AnomaliesInDataReport(Report):

    def __init__(self, anomalies_report : dict, categorical_encoding_by_feature_name):
        self.categorical_encoding = categorical_encoding_by_feature_name
        supported_keys = self.to_dict_meaning().keys()
        self._report = {}
        for feature_name, feature_report in anomalies_report.items():
            if len(set(feature_report.keys()) - set(supported_keys)):
                # excess of keys
                msg = ""
                for key, value in feature_report.items():
                    if key not in supported_keys:
                       msg += "key={} (value={}) is not supported.\n".format(key, value)
                raise NotImplementedError("Some keys are not supported:\n"+msg)

            self._report[feature_name] = {key: feature_report.get(key, None) for key in supported_keys}

    def to_dict(self, printable=False):
        return self._printable_dict(self._report, printable=printable)

    @classmethod
    def to_dict_meaning(cls):
        return dict(
            yt="Value that appear in data",
            yp="Expected value (model prediction)",
            probability_of_yt="y_proba of correct class (classification)",
            probability_of_yp="y_proba (classification)",
            n_stds = "Deviation from mean as # of stds (regression)",
            yrange="Range of feature: max(y) - min(y) (regression)",
            anomaly_score='Anomaly score. "1-1/n_stds" for numerical and "y_proba" for categorical',
            anomaly_score_ratio='Anomaly score divided by threshold',
            threshold='Threshold chosen/calculated, depending on analysis parameters',
            metric_score="Model score of model trained on 50% of data and tested on the other 50%",
            metric="Metric used to calculate metric_score")

    def plot(self, features_to_plot=None,
             plot_only_above_threshold=False,
             figsize=(10,10)):
        features_to_plot = features_to_plot or sorted(self._report.keys())

        for feature in features_to_plot:
            if feature not in self._report:
                raise KeyError("{} not recognized".format(feature))

            report = self._report[feature]


            v = report['anomaly_score_ratio']
            if numpy.all(v<1): # skip
                logger.info('Feature "{}" has no anomalies - nothing to plot.'.format(feature))
                continue

            if report['n_stds'] is not None:
                self._plot_regression(feature_name=feature, report=report, plot_only_above_threshold=plot_only_above_threshold)
            else:
                self._plot_classification(feature_name=feature, report=report, plot_only_above_threshold=plot_only_above_threshold)

            # plt.show()

    def _plot_regression(self, feature_name, report, plot_only_above_threshold):
        # 1 scatter
        ax = plt.figure(figsize=(10,5)).add_subplot()

        is_finite = numpy.isfinite(report['yt'])
        if plot_only_above_threshold:
            subset = report['anomaly_score_ratio'] >= 1
        else:
            subset = report['anomaly_score_ratio'] >= 0
        subset = subset & is_finite

        x,y = report['yt'][subset], report['yp'][subset]
        ax.scatter(x,y, c=report['anomaly_score_ratio'][subset])

        mn = numpy.min([x.min(), y.min()])
        mx = numpy.max([x.max(), y.max()])
        ax.plot([mn, mx], [mn, mx],':k')

        ax.set_xlabel('"{}": Value in data'.format(feature_name))
        ax.set_ylabel('"{}": Expected value'.format(feature_name))
        ax.set_title('Scatter plot for actual and expected values for feature "{}"'.format(feature_name))

        plt.tight_layout()

        # plt.show()

    def _plot_classification(self, feature_name, report, plot_only_above_threshold):
        from pytolemaic.analysis_logic.model_analysis.scoring.scoring_report import ConfusionMatrixReport
        class AnomalousConfusionMatrixReport(ConfusionMatrixReport):

            def plot(self, axs=None, figsize=(12, 5), title='Confusion Matrix'):
                if axs is None:
                    fig, axs = plt.subplots(1, 1, figsize=figsize)

                cm, labels, _ = self.cm_no_empty_classes(self.confusion_matrix, self.labels)

                self._plot_confusion_matrix(confusion_matrix=cm,
                                            labels=labels,
                                            title=title,
                                            ax=axs)

                plt.tight_layout()

        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10, 5)) # no sharex/sharey because 2nd matrix is more sparse.

        is_finite = numpy.isfinite(report['yt'].astype(float))

        if self.categorical_encoding and self.categorical_encoding.get(feature_name, None):
            labels = [self.categorical_encoding[feature_name][k] for k in
                      sorted(self.categorical_encoding[feature_name].keys())]
        else:
            labels = None

        for ax in [ax1, ax2]: #ax3
            # ax3 is used to weight confusion matrix samples by anomaly score. Doesn't seem to be interesting...
            yt = report['yt'][is_finite].astype(int)
            yp = report['yp'][is_finite].astype(int)
            anomaly_score_ratio = report['anomaly_score_ratio'][is_finite]


            if ax==ax1:
                title='Confusion matrix ("{}")'.format(feature_name)
                weights = None
            else:
                subset = anomaly_score_ratio >= 1
                yt, yp = yt[subset], yp[subset]
                # if plot_only_above_threshold:
                #     cm_labels = unique_labels(yt, yp)

                if ax==ax2:
                    title = 'Confusion matrix for anomalous values ("{}")'.format(feature_name)
                    weights = None
                else:
                    title = 'Confusion matrix for anomalous values weighted by anomaly score ("{}")'.format(feature_name)
                    weights = anomaly_score_ratio[subset]

            acmr = AnomalousConfusionMatrixReport(y_true=yt, y_pred=yp, labels=labels, sample_weight=weights)

            acmr.plot(axs=ax, title=title)



        plt.tight_layout()
        # plt.show()

    def insights(self, n_top_features=5, n_top_samples=5, train_data=None):
        insights_out = []
        n_anomalous_samples = {}
        scores = {}
        features = self._report.keys()
        anomaloues_samples={}

        # extract data
        for feature in features:
            report = self._report[feature]
            v = report['anomaly_score_ratio']

            n_anomalies = numpy.sum(v >= 1)
            if n_anomalies == 0:
                continue
            # if numpy.all(v < 1):  # skip
            #     continue
            n_anomalous_samples[feature] = n_anomalies
            scores[feature] = report['metric_score']

            inds = numpy.arange(len(v))
            inds = inds[v > 1]
            inds = sorted(inds, key=lambda i: v[i], reverse=True)[:n_top_samples]

            anomaloues_samples[feature] = inds

        if not n_anomalous_samples:
            triple_dots = "..." if len(features)>n_top_features else "."
            return ["No anomalous features found in features: {}".format(list(features)[:n_top_features]) + triple_dots]
        else:
            # only anomalous features
            features = n_anomalous_samples.keys()
            # sort by number of anomalies
            features = sorted(features, key=lambda f: n_anomalous_samples[f], reverse=True)

            # convert to array
            n_anomalous_samples_ = numpy.array([n_anomalous_samples[f] for f in features])
            n_features_with_anomaly = numpy.sum(n_anomalous_samples_>0)
            insight = "Found {} features with anomalous samples.".format(n_features_with_anomaly)
            insights_out.append(insight)

            insight = "Features with most anomalies are {} with {} anomalies respectively"\
                .format(features[:n_top_features], n_anomalous_samples_[:n_top_features])
            insights_out.append(insight)

            insight = ""
            metrics = []
            for feature in features:
                insight += 'Anomalous samples in feature "{}":\n'.format(feature)
                indices = anomaloues_samples[feature]

                report = self._report[feature]
                metrics.append(report['metric'])
                report = {key: report[key][indices] for key in report
                          if isinstance(report[key], numpy.ndarray) and len(report[key])==len(report['yt'])}
                report['indices'] = indices
                for key, value in report.items():
                    report[key] = numpy.round(numpy.array(report[key]).tolist(), 3).tolist() # tolist() before round is to get desired rounding

                # print(report)
                df = pd.DataFrame(report)
                if train_data is not None:
                    if isinstance(train_data, numpy.ndarray):
                        train_data = pd.DataFrame(train_data)
                    elif isinstance(train_data, pd.DataFrame):
                        train_data = train_data.copy()
                        train_data.reset_index(inplace=True)
                    elif isinstance(train_data, DMD):
                        train_data = pd.DataFrame(train_data.values,  columns=train_data.feature_names)
                    else:
                        raise ValueError("type {} is not supported".format(type(train_data)))

                    df = pd.merge(train_data, df)

                if self.categorical_encoding and self.categorical_encoding.get(feature, None):
                    df['yt'] = df['yt'].apply(lambda k: self.categorical_encoding[feature][k])
                    df['yp'] = df['yp'].apply(lambda k: self.categorical_encoding[feature][k])

                df.rename(columns={'yt': 'Value in data', 'yp': 'Value expected by model'},
                          inplace=True)

                df.sort_values('indices', inplace=True)
                df.set_index('indices',inplace=True, drop=True)
                mx = pd.get_option('display.max_columns')
                pd.set_option('display.max_columns',None)
                insight += "{}\n".format(df.head(n_top_samples) )
                pd.set_option('display.max_columns',mx)
            insights_out.append(insight)

            # sort by features easiest to predict by other features:
            features = sorted(features, key=lambda f: scores[f], reverse=True)
            scores = [scores[f] for f in features]
            n_anomalous_samples_ = [n_anomalous_samples[f] for f in features]

            metrics = list(set(metrics))
            metrics = " or ".join(metrics) + " metric" + "s"*(len(metrics)-1)

            insight = "Features easiest to predict based on other features are {} with scores={} ({}). Number of anomalies in these features: {}" \
                .format(features[:n_top_features],
                        numpy.round(scores[:n_top_features],3),
                        metrics, n_anomalous_samples_[:n_top_features])
            insights_out.append(insight)

        return insights_out

"""

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
    def __init__(self, class_counts, outliers_count, missing_values_report: MissingValuesReport,
                 covariance_shift_report:CovarianceShiftReport=None):
        self._class_counts = class_counts
        self._outliers_count = outliers_count
        self._missing_values_report = missing_values_report
        self._covariance_shift_report = covariance_shift_report


    @property
    def class_counts(self) -> dict:
        return self._class_counts

    @property
    def outliers_count(self) -> dict:
        return self._outliers_count

    @property
    def missing_values_report(self)->MissingValuesReport:
        return self._missing_values_report

    @property
    def covariance_shift_report(self)->CovarianceShiftReport:
        return self._covariance_shift_report

    def to_dict(self, printable=False):
        out = dict(few_class_representatives=self.class_counts,
                   outliers_count=self.outliers_count,
                   missing_values=self.missing_values_report.to_dict(printable=printable),
                   covariance_shift=None,
                   )
        if self.covariance_shift_report is not None:
            out.update(dict(covariance_shift=self.covariance_shift_report.to_dict(printable=printable)))
        return self._printable_dict(out, printable=printable)

    @classmethod
    def to_dict_meaning(cls):
        return dict(
            few_class_representatives='{feature_name: {class: instances_count}} : listing categorical features that has at least 1 class which is under-represented (less than 10 instances).',
            outliers_count='{feature_name: {n-sigma: outliers_info}} : listing numerical features that has more outliers than is expected with respect to n-sigma. '
                           'E.g. for 3-sigma we expect ceil(0.0027 x n_samples) outliers.',
            missing_values=MissingValuesReport.to_dict_meaning(),
            covariance_shift=CovarianceShiftReport.to_dict_meaning(),
        )

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
        if self.covariance_shift_report is not None:
            self.covariance_shift_report.plot()

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
                                    self.covariance_shift_report.insights(),
                                    self._outlier_counts_insights(),
                                    self._class_count_insights()))
"""