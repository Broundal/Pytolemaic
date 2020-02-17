from matplotlib import pyplot as plt

from pytolemaic.utils.base_report import Report
from pytolemaic.utils.general import GeneralUtils


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


class SensitivityOfFeaturesReport(Report):
    def __init__(self, method: str, sensitivities: dict):
        self._method = method
        self._sensitivities = sensitivities

    def to_dict(self, printable=False):
        out = dict(
            method=self.method,
            sensitivities=self.sensitivities,
        )
        return self._printable_dict(out, printable=printable)

    @classmethod
    def to_dict_meaning(cls):
        return dict(
            method="Method used to calculate sensitivity",
            sensitivities="key-value dictionary where the key is feature name and value is feature sensitivity",
        )

    def plot(self, ax=None, n_features_to_plot=10):
        if ax is None:
            fig, ax = plt.subplots(1)

        sorted_features = self.sorted_sensitivities  # sorted(self.sensitivities.items(), key=lambda kv: -kv[1])
        
        if n_features_to_plot is not None:
            sorted_features = sorted_features[:min(n_features_to_plot, len(sorted_features))]

        keys, values = zip(*sorted_features)
        ax.barh(list(reversed(keys)), list(reversed(values)))
        ax.set(
            title='"{}" Feature Sensitivity'.format(self.method),
            xlabel='Sensitivity value')
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


class SensitivityFullReport(Report):

    def __init__(self,
                 shuffle_report: SensitivityOfFeaturesReport,
                 shuffle_stats_report: SensitivityStatsReport,
                 missing_report: SensitivityOfFeaturesReport,
                 missing_stats_report: SensitivityStatsReport,
                 vulnerability_report: SensitivityVulnerabilityReport
                 ):
        self._shuffle_report = shuffle_report
        self._shuffle_stats_report = shuffle_stats_report
        self._missing_report = missing_report
        self._missing_stats_report = missing_stats_report
        self._vulnerability_report = vulnerability_report

    def plot(self):
        if self.missing_report is not None:

            fig, ((a11, a12), (a21, a22)) = plt.subplots(2, 2, figsize = (10, 10))

            self.shuffle_report.plot(ax=a11)
            self.shuffle_stats_report.plot(ax=a12, method=self.shuffle_report.method)
            self.missing_report.plot(ax=a21)
            self.missing_stats_report.plot(ax=a22, method=self.missing_report.method)

        else:
            fig, (a11, a12) = plt.subplots(1, 2, figsize = (10, 10))

            self.shuffle_report.plot(ax=a11)
            self.shuffle_stats_report.plot(ax=a12, method=self.shuffle_report.method)

        plt.tight_layout()

        self.vulnerability_report.plot()

    @classmethod
    def to_dict_meaning(cls):
        return dict(
            shuffle_report=SensitivityOfFeaturesReport.to_dict_meaning(),
            shuffle_stats_report=SensitivityStatsReport.to_dict_meaning(),
            missing_report=SensitivityOfFeaturesReport.to_dict_meaning(),
            missing_stats_report=SensitivityStatsReport.to_dict_meaning(),
            vulnerability_report=SensitivityVulnerabilityReport.to_dict_meaning(),
        )

    def to_dict(self, printable=False):
        out = dict(
            shuffle_report=self.shuffle_report.to_dict(),
            shuffle_stats_report=self.shuffle_stats_report.to_dict(),
            missing_report=None if self.missing_report is None else self.missing_report.to_dict(),
            missing_stats_report=None if self.missing_report is None else self.missing_stats_report.to_dict(),
            vulnerability_report=self.vulnerability_report.to_dict(),
        )
        return self._printable_dict(out, printable=printable)

    @property
    def shuffle_report(self) -> SensitivityOfFeaturesReport:
        return self._shuffle_report

    @property
    def shuffle_stats_report(self) -> SensitivityStatsReport:
        return self._shuffle_stats_report

    @property
    def missing_report(self) -> SensitivityOfFeaturesReport:
        return self._missing_report

    @property
    def missing_stats_report(self) -> SensitivityStatsReport:
        return self._missing_stats_report

    @property
    def vulnerability_report(self) -> SensitivityVulnerabilityReport:
        return self._vulnerability_report

if __name__ == '__main__':
    from pprint import pprint
    pprint(SensitivityFullReport.to_dict_meaning(), width=160)