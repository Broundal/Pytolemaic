import numpy
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, RocCurveDisplay, PrecisionRecallDisplay, \
    precision_recall_curve, average_precision_score, auc, roc_curve
from sklearn.utils.multiclass import unique_labels

from pytolemaic.utils.general import GeneralUtils
from pytolemaic.utils.metrics import Metrics


class ROCCurveReport():
    def __init__(self, y_true, y_pred, labels=None, sample_weight=None):
        self._labels = labels if labels is not None else unique_labels(y_true, y_pred)
        self._roc_curve = {}
        self._auc = {}
        for class_index, label in enumerate(self.labels):
            fpr, tpr, thresholds = roc_curve(y_true == class_index, y_pred == class_index,
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

    def to_dict(self):
        return dict(roc_curve=self.roc_curve,
                    auc=self.auc,
                    labels=self.labels)

    @classmethod
    def to_dict_meaning(cls):
        return dict(
            roc_curve="ROC curve as dict (fpr, tpr, and thresholds), per label",
            auc='Area under curve per label',
            labels="The class labels"
        )

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplot()

        ax.set_title("ROC Curve")
        possible_colors = GeneralUtils.shuffled_colors()
        for class_index, label in enumerate(self.labels):
            fpr, tpr = self._roc_curve[label]['fpr'], self._roc_curve[label]['tpr']
            roc_auc = self.auc[label]
            viz = RocCurveDisplay(fpr, tpr, roc_auc, 'Classifier')

            viz.plot(ax=ax, name=label, color=possible_colors[class_index])

        plt.draw()


class PrecisionRecallCurveReport():
    def __init__(self, y_true, y_pred, labels=None, sample_weight=None):
        self._labels = labels if labels is not None else unique_labels(y_true, y_pred)
        self._recall_precision_curve = {}
        self._average_precision = {}
        for class_index, label in enumerate(self.labels):
            precision, recall, thresholds = precision_recall_curve(y_true == class_index, y_pred == class_index,
                                                                   pos_label=1, sample_weight=sample_weight, )

            self._recall_precision_curve[label] = dict(precision=precision, recall=recall, thresholds=thresholds)

            self._average_precision[label] = average_precision_score(y_true == class_index, y_pred == class_index,
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

    def to_dict(self):
        return dict(recall_precision_curve=self.recall_precision_curve,
                    auc=self.average_precision,
                    labels=self.labels)

    @classmethod
    def to_dict_meaning(cls):
        return dict(
            recall_precision_curve="Recall Precision Curve as dict (recall, precision, and thresholds), per label",
            auc='Area under curve per label',
            labels="The class labels"
        )

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplot()

        ax.set_title("Precision Recall Curve")
        possible_colors = GeneralUtils.shuffled_colors()
        for class_index, label in enumerate(self.labels):
            precision = self._recall_precision_curve[label]['precision']
            recall = self._recall_precision_curve[label]['recall']
            average_precision = self._average_precision[label]

            viz = PrecisionRecallDisplay(precision, recall, average_precision, 'Classifier')
            viz.plot(ax=ax, name=label, color=possible_colors[class_index])


class SklearnClassificationReport():
    def __init__(self, y_true, y_pred, labels=None,
                 sample_weight=None, digits=3):
        self._labels = labels if labels is not None else unique_labels(y_true, y_pred)

        self._sample_weight = sample_weight

        self._classification_report_text = classification_report(y_true=y_true, y_pred=y_pred.reshape(-1, 1),
                                                                 labels=None,
                                                                 target_names=[str(k) for k in self._labels],
                                                                 sample_weight=sample_weight, digits=digits,
                                                                 output_dict=False)

        self._classification_report_dict = classification_report(y_true=y_true, y_pred=y_pred,
                                                                 labels=None, target_names=self._labels,
                                                                 sample_weight=sample_weight, digits=digits,
                                                                 output_dict=True)

        self._roc_curve = ROCCurveReport(y_true=y_true, y_pred=y_pred,
                                         labels=self.labels, sample_weight=sample_weight)

        self._precision_recall_curve = PrecisionRecallCurveReport(y_true=y_true, y_pred=y_pred,
                                                                  labels=self.labels, sample_weight=sample_weight)
        self._y_true = y_true
        self._y_pred = y_pred

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
    def classification_report(self):
        return self._classification_report_dict

    def to_dict(self):
        return dict(classification_report_dict=self.classification_report,
                    roc_curve=self.roc_curve.to_dict(),
                    precision_recall_curve=self.precision_recall_curve.to_dict(),
                    labels=self.labels)

    @classmethod
    def to_dict_meaning(cls):
        return dict(
            classification_report_dict="Accuracy score for various metrics",
            roc_curve="Roc curve report",
            precision_recall_curve="Precision-Recall curve report",
            labels="The class labels"
        )

    def _plot_classification_report(self):
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 3 + len(self.labels)))
        fig.text(0.5, 0.5, self._classification_report_text,
                 ha='center', va='center', size=20, fontname='courier', family='monospace')


    def plot(self):
        self._plot_classification_report()

        fig, (ax1, ax2) = plt.subplots(1, 2)

        self.precision_recall_curve.plot(ax1)
        self.roc_curve.plot(ax2)

        plt.tight_layout()
        plt.draw()


class ConfusionMatrixReport():
    def __init__(self, y_true, y_pred, labels=None):
        self._confusion_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred,
                                                  labels=unique_labels(y_true, y_pred)).tolist()
        self._labels = labels if labels is not None else unique_labels(y_true, y_pred)

    @property
    def labels(self):
        return self._labels

    @property
    def confusion_matrix(self):
        return self._confusion_matrix

    @property
    def normalized_confusion_matrix(self):
        cm = numpy.array(self.confusion_matrix)
        cm = cm / cm.sum(axis=1)[:, numpy.newaxis]

        return GeneralUtils.f3(cm).tolist()

    def to_dict(self):
        return dict(confusion_matrix=self.confusion_matrix,
                    normalized_confusion_matrix=self.normalized_confusion_matrix,
                    labels=self.labels)

    @classmethod
    def to_dict_meaning(cls):
        return dict(
            confusion_matrix="Confusion Matrix - rows indicate true values, columns indicate predicted value (in contrast to the convention shown in https://en.wikipedia.org/wiki/Confusion_matrix)",
            normalized_confusion_matrix="Normalized confusion matrix - the sum of each rows is equal to 1",
            labels="The class labels"
            )

    @classmethod
    def _plot_confusion_matrix(cls, confusion_matrix, labels, title,
                               ax, cmap=plt.cm.Greens):

        cm = numpy.array(confusion_matrix)
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

        # ax.figure.colorbar(im, ax=ax)

        ax.set(xticks=[-0.5] + numpy.arange(cm.shape[1]).tolist() + [cm.shape[1] - 0.5],
               yticks=[-0.5] + numpy.arange(cm.shape[0]).tolist() + [cm.shape[0] - 0.5],
               xticklabels=[''] + labels.tolist(), yticklabels=[''] + labels.tolist(),
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
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        return ax

    def plot(self):
        fig, (ax1, ax2) = plt.subplots(1, 2)

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


class ScatterReport():
    def __init__(self, y_true, y_pred, error_bars=None):
        self._y_true = y_true
        self._y_pred = y_pred
        self._error_bars = error_bars


    @property
    def y_true(self):
        return self._y_true

    @property
    def y_pred(self):
        return self._y_pred

    def to_dict(self):
        return dict(y_true=GeneralUtils.f5(self.y_true),
                    y_pred=GeneralUtils.f5(self.y_pred))

    @classmethod
    def to_dict_meaning(cls):
        return dict(y_true="True values",
                    y_pred="Predicted values"
                    )

    def plot(self, max_points=500):
        if max_points is None:
            max_points = len(self.y_true)

        rs = numpy.random.RandomState(0)
        inds = rs.permutation(len(self.y_true))[:max_points]


        plt.figure()
        plt.errorbar(self.y_true[inds], self.y_pred[inds], xerr=None, yerr=self._error_bars[inds], fmt='.b', ecolor='k')

        plt.xlabel('Y true')
        plt.ylabel('Y predicted')
        plt.title('Scatter plot')
        plt.draw()
        # plt.show()


class ScoringMetricReport():
    def __init__(self, metric, value, ci_low, ci_high):
        self._metric = metric
        self._value = value
        self._ci_low = ci_low
        self._ci_high = ci_high

    def to_dict(self):
        return dict(
            metric=self.metric,
            value=GeneralUtils.f5(self.value),
            ci_low=GeneralUtils.f5(self.ci_low),
            ci_high=GeneralUtils.f5(self.ci_high),
            ci_ratio=GeneralUtils.f5(self.ci_ratio),
        )

    @classmethod
    def to_dict_meaning(cls):
        return dict(metric="The metric which was used to calculate the score value",
                    value="The metric value - could be a score value (e.g. auc) or a loss value (e.g. mae), depending on metric",
                    ci_low="Lower confidence interval based on percentile 25",
                    ci_high="Higher confidence interval based on percentile 75",
                    ci_ratio="Measure confidence interval relative size - lower is better. Equation: (ci_high-ci_low)/(ci_low+ci_high)*2",
                    )

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1)


        ci_low = GeneralUtils.f5(self.ci_low)
        ci_high = GeneralUtils.f5(self.ci_high)
        value = GeneralUtils.f5(self.value)
        n_digits = -int(numpy.log10(ci_high - ci_low)) + 1  # 0.0011 --> -(-2) +1 = 3

        ax.plot([ci_low, ci_high], [1, 1], '-b',
                ci_low, 1, '|b',
                ci_high, 1, '|b',
                value, 1, 'or', )

        delta = (ci_high - ci_low) * 1e-1 + 10 ** -n_digits / 2
        l_lim = numpy.round(max(0, ci_low - delta), n_digits)
        r_lim = numpy.round(min(1, ci_high + delta), n_digits)

        ax.set_xlim(l_lim, r_lim)
        x = numpy.linspace(l_lim, r_lim, num=1 + int(numpy.round(r_lim - l_lim, n_digits) / 10 ** -n_digits))

        xlabels = ["%.5g" % numpy.round(k, n_digits) for k in x]
        ax.set(xticks=x.tolist(),
               xticklabels=xlabels,
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

        ci_ratio = abs(ci_high - ci_low) / (ci_high + ci_low) * 2
        return ci_ratio


class ScoringFullReport():
    def __init__(self, target_metric, metric_reports: [ScoringMetricReport], separation_quality: float,
                 confusion_matrix: ConfusionMatrixReport = None, scatter: ScatterReport = None,
                 classification_report: SklearnClassificationReport = None):
        self._target_metric = target_metric
        self._separation_quality = separation_quality
        self._metric_scores_dict = {r.metric: r for r in metric_reports}
        self._confusion_matrix = confusion_matrix
        self._scatter = scatter
        self._classification_report = classification_report

    def plot(self):
        if self.confusion_matrix is not None:
            self.confusion_matrix.plot()
        if self.scatter is not None:
            self.scatter.plot()
        if self.classification_report is not None:
            self.classification_report.plot()

        n = len(self.metric_scores)
        fig, axs = plt.subplots(n)
        for i, k in enumerate(sorted(self.metric_scores.keys())):
            self.metric_scores[k].plot(axs[i])

        plt.tight_layout()

    def to_dict(self):
        metric_scores = {k: v.to_dict() for k, v in self.metric_scores.items()}
        for v in metric_scores.values():
            v.pop('metric', None)

        return dict(
            metric_scores=metric_scores,
            target_metric=self.target_metric,
            separation_quality=GeneralUtils.f5(self.separation_quality),
            scatter=None if self.scatter is None else self.scatter.to_dict(),
            confusion_matrix=None if self.confusion_matrix is None else self.confusion_matrix.to_dict(),
            classification_report=None if self.classification_report is None else self.classification_report.to_dict()
        )

    @classmethod
    def to_dict_meaning(cls):
        return dict(
            target_metric="Metric of interest",
            metric_scores="Score information for various metrics saved in a dict structure where key is the metric name and value is of type {}".format(ScoringMetricReport.__name__),
            separation_quality="Measure whether the test and train comes from same distribution. High quality (max 1) means the test and train come from the same distribution. Low score (min 0) means the test set is problematic.",
            scatter="Scatter information (y_true vs y_pred). Available only for regressors.",
            confusion_matrix="Confusion matrix (y_true vs y_pred). Available only for classifiers.",
            classification_report="Sklearn's classification report"
        )

    @property
    def separation_quality(self):
        return self._separation_quality

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
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(5, 1.5))
    text = fig.text(0.5, 0.5, 'Hello path effects world!\nThis is the normal '
                              'path effect.\nPretty dull, huh?',
                    ha='center', va='center', size=20)
    plt.show()
