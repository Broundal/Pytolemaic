class SensitivityStatsReport():
    def __init__(self, n_features: int, n_low: int, n_zero: int, n_non_zero: int):
        self._n_features = n_features
        self._n_low = n_low
        self._n_zero = n_zero
        self._n_non_zero = n_non_zero

        if not n_zero + n_non_zero == n_features:
            raise ValueError(" n_zero + n_non_zero != n_features")

    @property
    def n_features(self):
        return self._n_features

    @property
    def n_low(self):
        return self._n_low

    @property
    def n_zero(self):
        return self._n_zero

    @property
    def n_non_zero(self):
        return self._n_non_zero


class SensitivityVulnerabilityReport():
    def __init__(self, imputation: float, leakage: float, too_many_features: float):
        self._too_many_features = too_many_features
        self._imputation = imputation
        self._leakage = leakage

    @property
    def imputation(self):
        return self._imputation

    @property
    def leakage(self):
        return self._leakage

    @property
    def too_many_features(self):
        return self._too_many_features


class SensitivityOfFeaturesReport():
    def __init__(self, method: str, sensitivities: dict):
        self._method = method
        self._sensitivities = sensitivities


    @property
    def method(self):
        return self._method

    @property
    def sensitivities(self):
        return self._sensitivities


class SensitivityFullReport():

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

    @property
    def shuffle_report(self):
        return self._shuffle_report

    @property
    def shuffle_stats_report(self):
        return self._shuffle_stats_report

    @property
    def missing_report(self):
        return self._missing_report

    @property
    def missing_stats_report(self):
        return self._missing_stats_report

    @property
    def vulnerability_report(self):
        return self._vulnerability_report
