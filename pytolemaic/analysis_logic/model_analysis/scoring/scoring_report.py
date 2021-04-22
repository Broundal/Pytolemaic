import itertools

import numpy
import sklearn.calibration
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, \
    RocCurveDisplay, PrecisionRecallDisplay, \
    precision_recall_curve, average_precision_score, auc, roc_curve, \
    brier_score_loss
from sklearn.utils.multiclass import unique_labels

from pytolemaic.utils.base_report import Report
from pytolemaic.utils.general import GeneralUtils
from pytolemaic.utils.metrics import Metrics


class ROCCurveReport(Report):
    def __init__(self, y_true, y_proba, labels=None, sample_weight=None):
        self._labels = labels if labels is not None else unique_labels(y_true).tolist()
        self._roc_curve = {}
        self._auc = {}
        for class_index, label in enumerate(self.labels):
            fpr, tpr, thresholds = roc_curve(y_true == class_index, y_proba[:, class_index],
                                             pos_label=1, sample_weight=sample_weight,
                                             drop_intermediate=True)

            self._roc_curve[label] = dict(fpr=fpr, tpr=tpr, thresholds=thresholds)
            self._auc[label] = auc(fpr, tpr)

    @property
    def labels(self):
        return self._labels

    @property
    def roc_curve(self):
        return self._roc_curve

    @property
    def auc(self):
        return self._auc

    def to_dict(self, printable=False):
        out = dict(roc_curve=self.roc_curve,
                   auc=self.auc,
                   labels=self.labels)
        return self._printable_dict(out, printable=printable)

    @classmethod
    def to_dict_meaning(cls):
        return dict(
            roc_curve="ROC curve as dict (fpr, tpr, and thresholds), per label",
            auc='Area under curve per label',
            labels="The class labels"
        )

    def plot(self, ax=None, figsize=(10,5)):
        if ax is None:
            fig, ax = plt.subplots(1,1,figsize=figsize)

        ax.set_title("ROC Curve")
        possible_colors = GeneralUtils.shuffled_colors()
        for class_index, label in enumerate(self.labels):
            fpr, tpr = self._roc_curve[label]['fpr'], self._roc_curve[label]['tpr']
            roc_auc = self.auc[label]
            viz = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Classifier')

            viz.plot(ax=ax, name=label, color=possible_colors[class_index])

        plt.draw()

    def insights(self):
        min_points_threshold = 3 + 2  # 3 + '0' + '1'

        thresholds = {label: set(numpy.clip(self._roc_curve[label]['thresholds'], 0, 1)) for label in self.labels}

        insights = []

        for label in self.labels:
            n_points = len(thresholds[label])
            if n_points <= min_points_threshold:
                insights = ["Only {} probability values ({}) for class '{}'. "
                            "This may impede the tuning of prediction threshold and the calibartion curve. Such behavior may indicate a bug."
                                .format(n_points, sorted(thresholds[label]), label)]

        return self._add_cls_name_prefix(insights)


class PrecisionRecallCurveReport(Report):
    def __init__(self, y_true, y_proba, labels=None, sample_weight=None):
        self._labels = labels if labels is not None else unique_labels(y_true).tolist()
        self._recall_precision_curve = {}
        self._average_precision = {}
        for class_index, label in enumerate(self.labels):
            precision, recall, thresholds = precision_recall_curve(y_true == class_index, y_proba[:, class_index],
                                                                   pos_label=1, sample_weight=sample_weight, )

            self._recall_precision_curve[label] = dict(precision=precision, recall=recall, thresholds=thresholds)

            self._average_precision[label] = average_precision_score(y_true == class_index, y_proba[:, class_index],
                                                                     pos_label=1, sample_weight=sample_weight, )

    @property
    def labels(self):
        return self._labels

    @property
    def recall_precision_curve(self):
        return self._recall_precision_curve

    @property
    def average_precision(self):
        return self._average_precision

    def to_dict(self, printable=False):
        out = dict(recall_precision_curve=self.recall_precision_curve,
                   average_precision=self.average_precision,
                   labels=self.labels)
        return self._printable_dict(out, printable=printable)

    @classmethod
    def to_dict_meaning(cls):
        return dict(
            recall_precision_curve="Recall Precision Curve as dict (recall, precision, and thresholds), per label",
            average_precision='Average Precision per label',
            labels="The class labels"
        )

    def plot(self, ax=None, figsize=(10,5)):
        if ax is None:
            fig, ax = plt.subplots(1,1, figsize=figsize)

        ax.set_title("Precision Recall Curve")
        possible_colors = GeneralUtils.shuffled_colors()
        for class_index, label in enumerate(self.labels):
            precision = self._recall_precision_curve[label]['precision']
            recall = self._recall_precision_curve[label]['recall']
            average_precision = self._average_precision[label]

            viz = PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=average_precision,
                                         estimator_name='Classifier')

            viz.plot(ax=ax, name=label, color=possible_colors[class_index])

    def insights(self):
        return []


class CalibrationCurveReport(Report):
    def __init__(self, y_true, y_proba, labels=None, sample_weight=None,
                 n_bins=10):
        self._labels = labels if labels is not None else unique_labels(y_true).tolist()
        self._calibration_curve = {}
        self._brier_loss = {}
        self._y_proba = y_proba
        self._n_bins = n_bins

        for class_index, label in enumerate(self.labels):
            fraction_of_positives, mean_predicted_value = \
                sklearn.calibration.calibration_curve(
                    y_true=y_true == class_index,
                    y_prob=y_proba[:, class_index],
                    normalize=False,
                    n_bins=n_bins,
                    strategy='uniform')
            self._calibration_curve[label] = dict(
                fraction_of_positives=fraction_of_positives,
                mean_predicted_value=mean_predicted_value)
            self._brier_loss[label] = brier_score_loss(
                y_true=y_true == class_index,
                y_prob=y_proba[:, class_index],
                sample_weight=sample_weight,
                pos_label=1)

    @property
    def labels(self):
        return self._labels

    @property
    def calibration_curve(self):
        return self._calibration_curve

    @property
    def brier_loss(self):
        return self._brier_loss

    def to_dict(self, printable=False):
        out = dict(calibration_curve=self.calibration_curve,
                   brier_loss=self.brier_loss,
                   labels=self.labels)
        return self._printable_dict(out, printable=printable)

    @classmethod
    def to_dict_meaning(cls):
        return dict(
            calibration_curve="Calibration curve as a dict (fraction_of_positives, mean_predicted_value), per label",
            brier_loss="Brier loss (lower is better), per label",
            labels="The class labels"
        )

    def plot(self, axs=None, figsize=(10,10)):
        if axs is None:
            fig = plt.figure(figsize=figsize)
            ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
            ax2 = plt.subplot2grid((3, 1), (2, 0))
        else:
            ax1, ax2 = axs

        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

        ax1.set_title("Calibartion Curve")
        possible_colors = GeneralUtils.shuffled_colors()
        for class_index, label in enumerate(self.labels):
            mean_predicted_value = self._calibration_curve[label][
                'mean_predicted_value']
            fraction_of_positives = self._calibration_curve[label][
                'fraction_of_positives']

            brier_loss = self.brier_loss[label]
            ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                     color=possible_colors[class_index],
                     label="class %s (brier loss=%1.3f)" % (
                     label, brier_loss))

            # todo: remove y_proba from self
            ax2.hist(self._y_proba[:, class_index], range=(0, 1.),
                     bins=self._n_bins, label=label,
                     color=possible_colors[class_index],
                     histtype="step", lw=2)

        ax1.set_ylabel("Fraction of positives")
        ax1.set_ylim([-0.05, 1.05])
        ax1.legend(loc="lower right")
        ax1.set_title('Calibration plots (reliability curve)')

        ax2.set_xlabel("Mean predicted value")
        ax2.set_ylabel("Count")
        ax2.legend(loc="upper center", ncol=2)

    def insights(self):
        lvl1 = 0.25
        lvl2 = 0.5
        lvl3 = 0.75

        brier_loss = [(label, self._brier_loss[label]) for label in self.labels]
        label, max_loss = sorted(brier_loss, key=lambda pair: pair[1], reverse=True)[0]
        insights = []
        if max_loss <= lvl1:
            pass  # OK
        elif max_loss <= lvl2:
            insights = [
                'Brier loss for class {} indicates model is not well calibrated. Please look at Calibration Curve.'.format(
                    label)]
        elif max_loss <= lvl3:
            insights = [
                'Brier loss for class {} indicates model is badly calibrated! Check the Calibration Curve.'.format(
                    label)]
        else:
            insights = [
                'Brier loss for class {} indicates model is not calibrated at all! Check the Calibration Curve!'.format(
                    label)]

        return self._add_cls_name_prefix(insights)


class SklearnClassificationReport(Report):
    def __init__(self, y_true, y_pred, y_proba, labels=None,
                 sample_weight=None, digits=3):
        self._labels = labels if labels is not None else unique_labels(y_true,
                                                                       y_pred).tolist()

        self._sample_weight = sample_weight

        self._sklearn_performance_summary_text = classification_report(
            y_true=y_true, y_pred=y_pred.reshape(-1, 1),
            labels=None,
            target_names=[str(k) for k in self._labels],
            sample_weight=sample_weight, digits=digits,
            output_dict=False)

        self._sklearn_performance_summary_dict = classification_report(
            y_true=y_true, y_pred=y_pred,
            labels=None, target_names=self._labels,
            sample_weight=sample_weight, digits=digits,
            output_dict=True)

        self._roc_curve = ROCCurveReport(y_true=y_true, y_proba=y_proba,
                                         labels=self.labels,
                                         sample_weight=sample_weight)

        self._calibration_curve = CalibrationCurveReport(y_true=y_true,
                                                         y_proba=y_proba,
                                                         labels=self.labels,
                                                         sample_weight=sample_weight)

        self._precision_recall_curve = PrecisionRecallCurveReport(
            y_true=y_true, y_proba=y_proba,
            labels=self.labels, sample_weight=sample_weight)
        self._y_true = y_true
        self._y_pred = y_pred
        self._y_proba = y_proba

    @property
    def labels(self):
        return self._labels

    @property
    def roc_curve(self) -> ROCCurveReport:
        return self._roc_curve

    @property
    def precision_recall_curve(self) -> PrecisionRecallCurveReport:
        return self._precision_recall_curve

    @property
    def calibration_curve(self) -> CalibrationCurveReport:
        return self._calibration_curve

    @property
    def sklearn_performance_summary(self):
        return self._sklearn_performance_summary_dict

    def to_dict(self, printable=False):
        out = dict(
            sklearn_performance_summary=self.sklearn_performance_summary,
            roc_curve=self.roc_curve.to_dict(printable=printable),
            precision_recall_curve=self.precision_recall_curve.to_dict(printable=printable),
            calibration_curve=self.calibration_curve.to_dict(printable=printable),
            labels=self.labels)
        return self._printable_dict(out, printable=printable)

    @classmethod
    def to_dict_meaning(cls):
        return dict(
            sklearn_performance_summary="Accuracy score for various metrics produced by sklearn",
            roc_curve="ROC curve report",
            precision_recall_curve="Precision-Recall curve report",
            calibration_curve="Calibration curve report",
            labels="The class labels"
        )

    def _plot_classification_report(self):
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 1 + len(self.labels)))
        fig.text(0.5, 0.5, self._sklearn_performance_summary_text,
                 ha='center', va='center', size=20, fontname='courier', family='monospace')
        plt.tight_layout()


    def plot(self, figsize=(14,5)):
        self._plot_classification_report()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        self.precision_recall_curve.plot(ax1)
        self.roc_curve.plot(ax2)
        plt.tight_layout()

        # new figure
        self.calibration_curve.plot(figsize=figsize)

        plt.tight_layout()
        plt.draw()

    def _sklearn_summary_insights(self):
        lvl1 = 0.75
        lvl2 = 0.5
        lvl3 = 0.25

        insights = []

        metrics = ['f1-score', 'precision', 'recall']

        for label in list(self.labels) + ['macro avg']:
            for metric in metrics:
                score = self.sklearn_performance_summary[label][metric]
                if label == 'macro avg':
                    prefix = 'The overall performance ({}) is '.format(metric)
                else:
                    prefix = '{} score for class {} is {} which is '.format(metric, label, numpy.round(score, 2))

                if score >= lvl1:
                    pass  # ok
                elif score >= lvl2:
                    insights.append(prefix + 'quite low.')
                elif score >= lvl3:
                    insights.append(prefix + 'very low! Look at the confusion matrix.'.format(metric, label))
                elif score >= lvl2:
                    insights.append(prefix + 'extremely low! Check out the confusion matrix!'.format(metric, label))

        return self._add_cls_name_prefix(insights)

    def insights(self):
        return list(itertools.chain(self._sklearn_summary_insights(),
                                    self.roc_curve.insights(),
                                    self.precision_recall_curve.insights(),
                                    self.calibration_curve.insights(),
                                    ))


class ConfusionMatrixReport(Report):
    def __init__(self, y_true, y_pred, labels: list = None):
        self._confusion_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred,
                                                  labels=unique_labels(y_true, y_pred)).tolist()
        self._labels = labels if labels is not None else unique_labels(y_true, y_pred).tolist()
        if isinstance(self._labels, numpy.ndarray):
            self._labels = self._labels.tolist()

    @property
    def labels(self)->list:
        return self._labels

    @property
    def confusion_matrix(self):
        return self._confusion_matrix

    @property
    def normalized_confusion_matrix(self):
        cm = numpy.array(self.confusion_matrix)
        cm = cm / cm.sum(axis=1)[:, numpy.newaxis]

        return GeneralUtils.f3(cm).tolist()

    def to_dict(self, printable=False):
        out = dict(confusion_matrix=self.confusion_matrix,
                    normalized_confusion_matrix=self.normalized_confusion_matrix,
                    labels=self.labels)
        return self._printable_dict(out, printable=printable)

    @classmethod
    def to_dict_meaning(cls):
        return dict(
            confusion_matrix="Confusion Matrix - rows indicate true values, columns indicate predicted value (in contrast to the convention shown in https://en.wikipedia.org/wiki/Confusion_matrix)",
            normalized_confusion_matrix="Normalized confusion matrix - the sum of each rows is equal to 1",
            labels="The class labels"
            )

    @classmethod
    def _plot_confusion_matrix(cls, confusion_matrix, labels: list, title,
                               ax, cmap=plt.cm.Greens):

        cm = numpy.array(confusion_matrix)
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

        # ax.figure.colorbar(im, ax=ax)

        ax.set(xticks=[-0.5] + numpy.arange(cm.shape[1]).tolist() + [cm.shape[1] - 0.5],
               yticks=[-0.5] + numpy.arange(cm.shape[0]).tolist() + [cm.shape[0] - 0.5],
               xticklabels=[''] + labels + [''], yticklabels=[''] + labels + [''],
               title=title,
               ylabel='True labels',
               xlabel='Predicted labels')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2g' if numpy.min(cm[cm > 0]) < 1 else 'd'
        if fmt == 'd':
            cm = cm.astype(int)
        # noinspection PyArgumentList
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        return ax

    def plot(self, axs=None, figsize=(12,5)):
        if axs is None:
            fig, axs = plt.subplots(1, 2, figsize=figsize)

        ax1, ax2 = axs

        self._plot_confusion_matrix(confusion_matrix=self.confusion_matrix,
                                    labels=self.labels,
                                    title='Confusion Matrix',
                                    ax=ax1)
        self._plot_confusion_matrix(confusion_matrix=self.normalized_confusion_matrix,
                                    labels=self.labels,
                                    title='Normalized confusion Matrix',
                                    ax=ax2)

        plt.tight_layout()

        # plt.show()

    def insights(self):
        insights = []
        for i, label in enumerate(self.labels):
            if numpy.sum(numpy.array(self.confusion_matrix)[:, i]) < 10:
                insights.append('Model rarely predicts class {}! Is that ok??'.format(label))

        return self._add_cls_name_prefix(insights)

class ScatterReport(Report):
    def __init__(self, y_true, y_pred, error_bars=None):
        self._y_true = numpy.array(y_true).reshape(-1, 1)
        self._y_pred = numpy.array(y_pred).reshape(-1, 1)
        self._error_bars = error_bars

    @property
    def y_true(self):
        return self._y_true

    @property
    def y_pred(self):
        return self._y_pred

    def to_dict(self, printable=False):
        error_bars = self._error_bars.ravel() if self._error_bars is not None else None
        out = dict(y_true=self.y_true.ravel(),
                    y_pred=self.y_pred.ravel(),
                    error_bars=error_bars)
        return self._printable_dict(out, printable=printable)

    @classmethod
    def to_dict_meaning(cls):
        return dict(y_true="True values",
                    y_pred="Predicted values",
                    error_bars="The uncertainty of each prediction")

    def plot(self, max_points=500, figsize=(10,5), residual_plot=True):
        if max_points is None:
            max_points = len(self.y_true)

        rs = numpy.random.RandomState(0)
        inds = rs.permutation(len(self.y_true))[:max_points]

        if residual_plot:
            fig, (ax1, ax2) = plt.subplots(1,2,figsize=figsize)
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=figsize)



        y_true = self.y_true[inds]
        y_pred = self.y_pred[inds]
        error_bars = None if self._error_bars is None else self._error_bars[inds]

        ## scatter plot

        ax = ax1
        ax.errorbar(y_true, y_pred, xerr=None, yerr=error_bars, fmt='.b', ecolor='k')

        mn = numpy.min([y_true.min(), y_pred.min()])
        mx = numpy.max([y_true.max(), y_pred.max()])
        ax.plot([mn, mx], [mn, mx],':k')

        ax.set_xlabel('Y true')
        ax.set_ylabel('Y predicted')
        ax.set_title('Scatter plot')

        ## scatter plot
        if residual_plot:
            ax = ax2
            ax.errorbar(y_true, y_pred-y_true, xerr=None, yerr=error_bars, fmt='.b', ecolor='k')

            mn = numpy.min([y_true.min(), y_pred.min()])
            ax.plot([mn, mx], [0, 0], '-k')

            ax.set_xlabel('Y true')
            ax.set_ylabel('Ypred - Ytrue')
            ax.set_title('Residual plot')

        plt.draw()
        # plt.show()

    def insights(self):
        # todo: what insights can we derive?
        return []

class ScoringMetricReport(Report):
    def __init__(self, metric, value, ci_low, ci_high):
        self._metric = metric
        self._value = value
        self._ci_low = ci_low
        self._ci_high = ci_high

    def to_dict(self, printable=False):
        out = dict(
            metric=self.metric,
            value=self.value,
            ci_low=self.ci_low,
            ci_high=self.ci_high,
            ci_ratio=self.ci_ratio,
        )
        return self._printable_dict(out, printable=printable)

    @classmethod
    def to_dict_meaning(cls):
        return dict(metric="The metric which was used to calculate the score value",
                    value="The metric value - could be a score value (e.g. auc) or a loss value (e.g. mae), depending on metric",
                    ci_low="Lower confidence interval based on percentile 25",
                    ci_high="Higher confidence interval based on percentile 75",
                    ci_ratio="Measure confidence interval relative size - lower is better. Equation: (ci_high-ci_low)/(ci_low+ci_high)*2",
                    )

    def plot(self, ax=None, figsize=(10,5)):
        if ax is None:
            fig, ax = plt.subplots(1,1,figsize=figsize)


        ci_low = GeneralUtils.f5(self.ci_low)
        ci_high = GeneralUtils.f5(self.ci_high)
        value = GeneralUtils.f5(self.value)
        if ci_high == ci_low:
            n_digits = 5
        else:
            n_digits = -int(numpy.log10(ci_high - ci_low)) + 1  # 0.0011 --> -(-2) +1 = 3

        ax.plot([ci_low, ci_high], [1, 1], '-b',
                ci_low, 1, '|b',
                ci_high, 1, '|b',
                value, 1, 'or', )

        delta = (ci_high - ci_low) * 1e-1 + 10 ** -n_digits / 2

        metric_obj = Metrics.supported_metrics()[self.metric]
        r_lim = 1e100 if metric_obj.is_loss else 1
        l_lim = 0 if metric_obj.is_loss else -1e100

        l_lim = max(l_lim, numpy.round(ci_low - delta, n_digits))
        r_lim = min(r_lim, numpy.round(ci_high + delta, n_digits))

        ax.set_xlim(l_lim, r_lim)
        n_points = 1 + int(numpy.round(r_lim - l_lim, n_digits) / 10 ** -n_digits) % 10

        x = numpy.linspace(l_lim, r_lim, num=n_points)

        xlabels = ["%.5g" % numpy.round(k, n_digits) for k in x]
        ax.set(xticks=x.tolist(),
               xticklabels=xlabels,
               yticks=[0.5],
               yticklabels=[''],
               title='Confidence intervals for metric {}'.format(self.metric),
               ylabel='',
               xlabel='{}'.format(self.metric))

        # Loop over data dimensions and create text annotations.
        for x, label in [(ci_low, 'ci_low (25%)'),
                         (value, '{} value'.format(self.metric)),
                         (ci_high, 'ci_high  (75%)')]:
            y = 1.01 + 0.01 * (x == value)
            ax.text(x, y, label,
                    ha="center", va="center", )

        plt.draw()

    def insights(self):
        insights = []

        if self.value < self.ci_low or self.value > self.ci_high:
            insights.append('{} value {} is out of range of confidence interval [{},{}]'
                            .format(self.metric, self.value, self.ci_low, self.ci_high))

        lvl1 = 0.1
        lvl2 = 0.5
        ci_range_and_ratio = '[{:.3g}, {:.3g}], ci ratio of {:.3g}'.format(self.ci_low, self.ci_high, self.ci_ratio)
        if self.ci_ratio < lvl1:
            pass  # ok
        elif self.ci_ratio < lvl2:
            insights.append(
                'Confidence interval for metric {} is quite large ({})'.format(self.metric, ci_range_and_ratio))
        else:
            insights.append(
                'Confidence interval for metric {} is very large ({}). The score measurement of {} is inaccurate.'.format(
                    self.metric, ci_range_and_ratio, self.value))

        return self._add_cls_name_prefix(insights)

    @property
    def metric(self):
        return self._metric

    @property
    def value(self):
        return self._value

    @property
    def ci_low(self):
        return self._ci_low

    @property
    def ci_high(self):
        return self._ci_high

    @property
    def ci_ratio(self):
        # large ci difference is more of a concern if score is high
        ci_low = Metrics.metric_as_loss(value=self.ci_low, metric=self.metric)
        ci_high = Metrics.metric_as_loss(value=self.ci_high, metric=self.metric)

        if ci_low == ci_high:
            return 0
        else:
            ci_ratio = abs(ci_high - ci_low) / (ci_high + ci_low) * 2
            return ci_ratio


class ScoringFullReport(Report):
    def __init__(self, target_metric, metric_reports: [ScoringMetricReport],
                 confusion_matrix: ConfusionMatrixReport = None, scatter: ScatterReport = None,
                 classification_report: SklearnClassificationReport = None):
        self._target_metric = target_metric
        self._metric_scores_dict = {r.metric: r for r in metric_reports}
        self._confusion_matrix = confusion_matrix
        self._scatter = scatter
        self._classification_report = classification_report

    def plot(self, figsize=(12,5)):
        if self.confusion_matrix is not None:
            self.confusion_matrix.plot(figsize=figsize)
        if self.scatter is not None:
            self.scatter.plot(figsize=figsize)
        if self.classification_report is not None:
            self.classification_report.plot(figsize=figsize)

        n = len(self.metric_scores)
        fig, axs = plt.subplots(n, figsize=(12,n*2))
        for i, k in enumerate(sorted(self.metric_scores.keys())):
            self.metric_scores[k].plot(axs[i])

        plt.tight_layout()

    def to_dict(self, printable=False):
        metric_scores = {k: v.to_dict(printable=printable) for k, v in self.metric_scores.items()}
        for v in metric_scores.values():
            v.pop('metric', None)

        out = dict(
            metric_scores=metric_scores,
            target_metric=self.target_metric,
            scatter=None if self.scatter is None else self.scatter.to_dict(printable=printable),
            confusion_matrix=None if self.confusion_matrix is None else self.confusion_matrix.to_dict(
                printable=printable),
            classification_report=None if self.classification_report is None else self.classification_report.to_dict(
                printable=printable)
        )
        return self._printable_dict(out, printable=printable)

    @classmethod
    def to_dict_meaning(cls):
        return dict(
            target_metric="Metric of interest",
            metric_scores="Score information for various metrics saved in a dict structure where key is the metric name and value is of type {}".format(
                ScoringMetricReport.__name__),
            scatter="Scatter information (y_true vs y_pred). Available only for regressors.",
            confusion_matrix="Confusion matrix (y_true vs y_pred). Available only for classifiers.",
            classification_report="Sklearn's classification report"
        )

    def insights(self):
        insights_from_reports = [report.insights() for report in [self.metric_scores[self.target_metric],
                                                                  self.scatter, self.confusion_matrix,
                                                                  self.classification_report]
                                 if report is not None]

        return list(itertools.chain(*insights_from_reports))

        # metric_scores=metric_scores,
        # target_metric=self.target_metric,
        # scatter=None if self.scatter is None else self.scatter.to_dict(),
        # confusion_matrix=None if self.confusion_matrix is None else self.confusion_matrix.to_dict(),
        # classification_report=None if self.classification_report is None else self.classification_report.to_dict()


    @property
    def target_metric(self):
        return self._target_metric

    @property
    def metric_scores(self) -> dict:
        return self._metric_scores_dict

    @property
    def confusion_matrix(self) -> ConfusionMatrixReport:
        return self._confusion_matrix

    @property
    def scatter(self) -> ScatterReport:
        return self._scatter

    @property
    def classification_report(self) -> SklearnClassificationReport:
        return self._classification_report


if __name__ == '__main__':
    from pprint import pprint
    pprint(ScoringFullReport.to_dict_meaning(), width=160)