import numpy
from sklearn.metrics.classification import confusion_matrix

from pytolemaic.utils.metrics import Metrics
from matplotlib import pyplot as plt

class ConfusionMatrixReport():
    def __init__(self, y_true, y_pred, labels=None):
        self._confusion_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels).tolist()
        self._labels = labels

    @property
    def labels(self):
        return self._labels

    @property
    def confusion_matrix(self):
        return self._confusion_matrix

    def to_dict(self):
        return dict(confusion_matrix=self.confusion_matrix,
                    labels=self.labels)

    @classmethod
    def _plot_confusion_matrix(cls, confusion_matrix, labels,
                               normalize, title,
                               ax,
                               cmap=plt.cm.Greens):

        cm = numpy.array(confusion_matrix)
        if normalize:
            cm = cm / cm.sum(axis=1)[:, numpy.newaxis]



        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

        # ax.figure.colorbar(im, ax=ax)

        ax.set(xticks=[-0.5]+numpy.arange(cm.shape[1]).tolist()+[cm.shape[1]-0.5],
               yticks=[-0.5]+numpy.arange(cm.shape[0]).tolist()+[cm.shape[0]-0.5],
               xticklabels=['']+labels.tolist(), yticklabels=['']+labels.tolist(),
               title=title,
               ylabel='True labels',
               xlabel='Predicted labels')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        return ax

    def plot(self):
        fig, (ax1, ax2) = plt.subplots(1,2)

        self._plot_confusion_matrix(confusion_matrix=self.confusion_matrix,
                                    labels=self.labels,
                                    normalize=False,
                                    title='Confusion Matrix',
                                    ax=ax1)
        self._plot_confusion_matrix(confusion_matrix=self.confusion_matrix,
                                    labels=self.labels,
                                    normalize=True,
                                    title='Normalized confusion Matrix',
                                    ax=ax2)

        plt.tight_layout()

        # plt.show()



class ScatterReport():
    def __init__(self, y_true, y_pred):
        self._y_true = y_true
        self._y_pred = y_pred

    @property
    def y_true(self):
        return self._y_true

    @property
    def y_pred(self):
        return self._y_pred

    def to_dict(self):
        return dict(y_true=self.y_true,
                    y_pred=self.y_pred)

    def plot(self):
        plt.figure()
        plt.plot(self.y_true, self.y_pred, '.b')
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
            value=self.value,
            ci_low=self.ci_low,
            ci_high=self.ci_high,
            ci_ratio=self.ci_ratio,
        )

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1)

        ax.plot([self.ci_low, self.ci_high], [1, 1], '-b', self.ci_low, 1, '|b', self.ci_high, 1, '|b', self.value, 1, 'or', )

        delta = (self.ci_high - self.ci_low)*1e-1
        l_lim = max(0, self.ci_low-delta)
        r_lim = min(1, self.ci_high+delta)

        ax.set_xlim(l_lim, r_lim)
        x = numpy.round(numpy.linspace(l_lim, r_lim, num=5),3)
        xlabels = ["%.5g" % k for k in x]
        ax.set(xticks=x.tolist(),
               xticklabels=xlabels,
               yticklabels=[''],
               title='Confidence intervals for metric {}'.format(self.metric),
               ylabel='',
               xlabel='{}'.format(self.metric))

        # Loop over data dimensions and create text annotations.
        for x, label in [(self.ci_low, 'ci_low (25%)'), (self.value, '{} value'.format(self.metric)), (self.ci_high, 'ci_high  (75%)')]:
            y = 1.01
            ax.text(x, y, label,
                    ha="center", va="center",)

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
    def __init__(self, metric_reports: [ScoringMetricReport], separation_quality: float, confusion_matrix=None, scatter=None):
        self._separation_quality = separation_quality
        self._metric_scores_dict = {r.metric: r for r in metric_reports}
        self._confusion_matrix = confusion_matrix
        self._scatter = scatter

    def plot(self):
        if self.confusion_matrix is not None:
            self.confusion_matrix.plot()
        if self.scatter is not None:
            self.scatter.plot()

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
            separation_quality=self.separation_quality,
            scatter=None if self.scatter is None else self.scatter.to_dict(),
            confusion_matrix=None if self.confusion_matrix is None else self.confusion_matrix.to_dict()
        )

    @property
    def separation_quality(self):
        return self._separation_quality

    @property
    def metric_scores(self) -> dict:
        return self._metric_scores_dict

    @property
    def confusion_matrix(self)->ConfusionMatrixReport:
        return self._confusion_matrix

    @property
    def scatter(self)->ScatterReport:
        return self._scatter

if __name__ == '__main__':
    plt.figure()
    plt.plot(1,1,'|', 3,1,'|', 1.5,1,'.', [1,3], [1,1],'-')
    plt.show()