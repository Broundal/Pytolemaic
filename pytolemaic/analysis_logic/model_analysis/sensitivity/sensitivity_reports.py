import itertools

from matplotlib import pyplot as plt

from pytolemaic.utils.base_report import Report


class SensitivityTypes():
    shuffled = 'shuffled'
    missing = 'missing'


class SensitivityStatsReport(Report):
    def __init__(self, n_features: int, n_low: int, n_very_low: int, n_zero: int):
        self._n_features = n_features
        self._n_low = n_low
        self._n_very_low = n_very_low
        self._n_zero = n_zero

    def to_dict(self, printable=False):
        out = dict(
            n_features=self.n_features,
            n_low=self.n_low,
            n_very_low=self.n_very_low,
            n_zero=self.n_zero,
        )
        return self._printable_dict(out, printable=printable)

    @classmethod
    def to_dict_meaning(cls):
        return dict(
            n_features="Number of features in dataset",
            n_low="Number of feature with low sensitivity (sensitivity lower than 5% of max sensitivity)",
            n_very_low="Number of feature with low sensitivity (sensitivity lower than 5% of max sensitivity)",
            n_zero="Number of feature with zero sensitivity",
        )

    def plot(self, ax=None, method=None):
        if ax is None:
            fig, ax = plt.subplots(1)

        keys, values = zip(*sorted(self.to_dict().items()))

        title_prefix = '"{}" '.format(method) if method else ''
        ax.bar(keys, values)
        ax.set(
            title='{}Sensitivity Statistics'.format(title_prefix),
            ylabel='# of features')
        plt.draw()

    @property
    def n_features(self):
        return self._n_features

    @property
    def n_low(self):
        return self._n_low

    @property
    def n_very_low(self):
        return self._n_very_low

    @property
    def n_zero(self):
        return self._n_zero

    def insights(self):

        insights = []

        lvl = 0.5
        sentence = "More than {} of features have".format(int(lvl * self.n_features))
        if self.n_zero > lvl * self.n_features:
            insights.append("{} no sensitivity at all".format(sentence))
        elif self.n_very_low > lvl * self.n_features:
            insights.append("{} very low sensitivity".format(sentence))
        elif self.n_low > lvl * self.n_features:
            insights.append("{} low sensitivity".format(sentence))

        # insights = self._add_cls_name_prefix(insights)

        return insights


class SensitivityVulnerabilityReport(Report):
    def __init__(self, imputation: float, leakage: float, too_many_features: float):
        self._too_many_features = too_many_features
        self._imputation = imputation
        self._leakage = leakage

    def to_dict(self, printable=False):
        out = dict(
            too_many_features=self.too_many_features,
            imputation=self.imputation,
            leakage=self.leakage
        )
        return self._printable_dict(out, printable=printable)

    @classmethod
    def to_dict_meaning(cls):
        return dict(
            too_many_features="Using many features slow down model's training, affect data pipeline and preprocessing and may also make model more susceptible to overfit. Higher value means there are many redundant features in dataset.",
            imputation="Sensitivity of the features should be similar in all methods. If not, this may indicate an issue with the imputation process. Higher value means the imputation process has higher impact on the model.",
            leakage="Measure how likely the model is prone to data leakage."
        )

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1)

        keys, values = zip(*sorted(self.to_dict().items()))

        ax.bar(keys, values)
        ax.set(
            title="Model's vulnerability metrics",
            ylabel='Vulnerability scores',
            ylim=[0, 1])
        plt.draw()

    @property
    def imputation(self):
        return self._imputation

    @property
    def leakage(self):
        return self._leakage

    @property
    def too_many_features(self):
        return self._too_many_features

    def insights(self):

        insights = []
        lvl1, lvl2, lvl3 = 0.1, 0.5, 0.9
        sentence = "Vulnerability to imputation is {:.3g} indicating your model is %s to the imputation technique.".format(
            self.imputation)
        if self.imputation < lvl1:
            pass  # ok
        elif self.imputation < lvl2:
            insights.append(sentence % "somewhat sensitive")
        elif self.imputation < lvl3:
            insights.append(sentence % "sensitive")
        else:
            insights.append(sentence % "very sensitive" + "You should check what's going on, there may be a bug.")

        lvl1, lvl2 = 0.1, 0.75
        sentence = "Vulnerability to data leakage is {:.3g} indicating %s data leakage.".format(self.leakage)
        if self.leakage < lvl1:
            pass  # ok
        elif self.leakage < lvl2:
            insights.append(
                sentence % "there is a chance of" + " Check the features with the highest sensitivity scores.")
        else:
            insights.append(
                sentence % "a very high chance of" + " Check the features with the highest sensitivity scores.")

        lvl1, lvl2 = 0.1, 0.5
        sentence = "Vulnerability to number of features is {:.3g} indicating %s.".format(self.too_many_features)
        if self.too_many_features < lvl1:
            pass  # ok
        elif self.too_many_features < lvl2:
            insights.append(
                sentence % "there are substantial number of features which could be discarded / regularized")
        else:
            insights.append(
                sentence % "that many features have little value and may cause overfit. Discard these feature or increase regularization.")

        return self._add_cls_name_prefix(insights)

class SensitivityOfFeaturesReport(Report):
    def __init__(self, method: str, sensitivities: dict,
                 stats_report: SensitivityStatsReport):
        self._method = method
        self._sensitivities = sensitivities
        self._stats_report = stats_report

    def to_dict(self, printable=False):
        out = dict(
            method=self.method,
            sensitivities=self.sensitivities,
            stats=self.stats_report.to_dict(printable=printable)
        )
        return self._printable_dict(out, printable=printable)

    @classmethod
    def to_dict_meaning(cls):
        return dict(
            method="Method used to calculate sensitivity",
            sensitivities="key-value dictionary where the key is feature name and value is feature sensitivity",
            stats=SensitivityStatsReport.to_dict_meaning()
        )

    def plot(self, axs=None, n_features_to_plot=10):
        if axs is None:
            fig, axs = plt.subplots(1, 2, figsize=(10, 10))

        ax1, ax2 = axs

        sorted_features = self.sorted_sensitivities  # sorted(self.sensitivities.items(), key=lambda kv: -kv[1])

        if n_features_to_plot is not None:
            sorted_features = sorted_features[:min(n_features_to_plot, len(sorted_features))]

        keys, values = zip(*sorted_features)
        ax1.barh(list(reversed(keys)), list(reversed(values)))
        ax1.set(
            title='"{}" Feature Sensitivity'.format(self.method),
            xlabel='Sensitivity value')

        self.stats_report.plot(ax=ax2, method=self.method)
        plt.draw()

    @property
    def method(self):
        return self._method

    @property
    def sensitivities(self):
        return self._sensitivities

    @property
    def sorted_sensitivities(self):
        return [(k, v) for k, v in sorted(self._sensitivities.items(), key=lambda kv: -kv[1])]

    @property
    def stats_report(self) -> SensitivityStatsReport:
        return self._stats_report

    def insights(self):
        stats_report_insights = self.stats_report.insights()
        insights = []
        insights.append("The most important feature is '{}', followed by '{}' and '{}'."
                        .format(self.sorted_sensitivities[0][0],
                                self.sorted_sensitivities[1][0],
                                self.sorted_sensitivities[2][0]))
        if self.stats_report.n_zero > 0:
            zero_sensitivity = [feature for feature, value in self.sorted_sensitivities[-self.stats_report.n_zero:]]
            if len(zero_sensitivity) == 1:
                list_of_features = " '{}'".format(zero_sensitivity[0])
            else:
                list_of_features = "\n\t" + ", ".join(["'{}'".format(feature) for feature in zero_sensitivity])

            insights.append("The following features has 0 sensitivity:{}".format(list_of_features))

        if self.stats_report.n_very_low - self.stats_report.n_zero > 0:
            very_low_sensitivity = [feature for feature, value in
                                    self.sorted_sensitivities[
                                    -self.stats_report.n_very_low - self.stats_report.n_zero: -self.stats_report.n_zero]]
            if len(very_low_sensitivity) == 1:
                list_of_features = " '{}'".format(very_low_sensitivity[0])
            else:
                list_of_features = "\n\t" + ", ".join(["'{}'".format(feature) for feature in very_low_sensitivity])

            insights.append(
                "The following features can be discarded due to very low sensitivity:{}".format(list_of_features))

        insights = itertools.chain(stats_report_insights, insights)
        insights = ["{}.{}: {}".format(type(self).__name__, self.method, insight) for insight in insights]
        return insights

class SensitivityFullReport(Report):

    def __init__(self,
                 shuffle_report: SensitivityOfFeaturesReport,
                 missing_report: SensitivityOfFeaturesReport,
                 vulnerability_report: SensitivityVulnerabilityReport
                 ):
        self._shuffle_report = shuffle_report
        self._missing_report = missing_report
        self._vulnerability_report = vulnerability_report

    def plot(self):
        if self.missing_report is not None:
            fig, (axs1, axs2) = plt.subplots(2, 2, figsize=(10, 10))
            self.shuffle_report.plot(axs=axs1)
            self.missing_report.plot(axs=axs2)

        else:
            fig, axs = plt.subplots(1, 2, figsize=(10, 10))
            self.shuffle_report.plot(axs=axs)

        plt.tight_layout()

        self.vulnerability_report.plot()

    @classmethod
    def to_dict_meaning(cls):
        return dict(
            shuffle_report=SensitivityOfFeaturesReport.to_dict_meaning(),
            missing_report=SensitivityOfFeaturesReport.to_dict_meaning(),
            vulnerability_report=SensitivityVulnerabilityReport.to_dict_meaning(),
        )

    def to_dict(self, printable=False):
        out = dict(
            shuffle_report=self.shuffle_report.to_dict(),
            missing_report=None if self.missing_report is None else self.missing_report.to_dict(),
            vulnerability_report=self.vulnerability_report.to_dict(),
        )
        return self._printable_dict(out, printable=printable)

    @property
    def shuffle_report(self) -> SensitivityOfFeaturesReport:
        return self._shuffle_report

    @property
    def missing_report(self) -> SensitivityOfFeaturesReport:
        return self._missing_report

    @property
    def vulnerability_report(self) -> SensitivityVulnerabilityReport:
        return self._vulnerability_report

    def insights(self):
        return list(itertools.chain(self.shuffle_report.insights(),
                                    [] if self.missing_report is None else self.missing_report.insights(),
                                    self.vulnerability_report.insights()))


if __name__ == '__main__':
    from pprint import pprint

    pprint(SensitivityFullReport.to_dict_meaning(), width=160)
