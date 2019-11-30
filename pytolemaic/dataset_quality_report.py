from pytolemaic.analysis_logic.model_analysis.scoring.scoring_report import ScoringFullReport, ScoringMetricReport
from pytolemaic.analysis_logic.model_analysis.sensitivity.sensitivity_reports import SensitivityVulnerabilityReport
from pytolemaic.utils.constants import REGRESSION
from pytolemaic.utils.general import GeneralUtils
from pytolemaic.utils.metrics import Metrics


class TestSetQualityReport():
    def __init__(self, scoring_report: ScoringFullReport):
        self._scoring_report = scoring_report
        self._metric = scoring_report.target_metric

        self._test_set_quality = self._calculate_test_set_quality()

    def to_dict(self):
        return dict(
            ci_ratio=GeneralUtils.f5(self.ci_ratio),
            separation_quality=GeneralUtils.f5(self.separation_quality),
            test_set_quality=GeneralUtils.f5(self.test_set_quality),
        )

    @classmethod
    def to_dict_meaning(cls):
        return dict(
            ci_ratio=ScoringMetricReport.to_dict_meaning()['ci_ratio'],
            separation_quality=ScoringFullReport.to_dict_meaning()['separation_quality'],
            test_set_quality="Overall test set quality - higher is better",
        )

    def _calculate_test_set_quality(self):
        test_set_quality = 1.0
        test_set_quality = test_set_quality - self.ci_ratio - (1 - self.separation_quality)
        test_set_quality = max(test_set_quality, 0)
        return test_set_quality

    @property
    def ci_ratio(self):
        return self._scoring_report.metric_scores[self._metric].ci_ratio

    @property
    def metric(self):
        return self._metric

    @property
    def separation_quality(self):
        return self._scoring_report.separation_quality

    @property
    def scoring_report(self) -> ScoringFullReport:
        return self._scoring_report

    @property
    def test_set_quality(self):
        return self._test_set_quality


class TrainSetQualityReport():
    def __init__(self, vulnerability_report: SensitivityVulnerabilityReport):
        self._vulnerability_report = vulnerability_report

        self._train_set_quality = self._calculate_train_set_quality()

    def _calculate_train_set_quality(self):
        train_set_quality = 1.0
        train_set_quality = train_set_quality \
                            - self._vulnerability_report.leakage \
                            - self._vulnerability_report.too_many_features
        train_set_quality = max(train_set_quality, 0)

        return train_set_quality

    def to_dict(self):
        return dict(
            vulnerability_report=self.vulnerability_report.to_dict(),
            train_set_quality=self.train_set_quality,
        )

    @classmethod
    def to_dict_meaning(cls):
        return dict(vulnerability_report=SensitivityVulnerabilityReport.to_dict_meaning(),
                    train_set_quality="Overall train set quality - higher is better")

    @property
    def vulnerability_report(self) -> SensitivityVulnerabilityReport:
        return self._vulnerability_report

    @property
    def train_set_quality(self):
        return self._train_set_quality


class ModelQualityReport():
    def __init__(self, vulnerability_report: SensitivityVulnerabilityReport, scoring_report: ScoringFullReport):
        self._vulnerability_report = vulnerability_report
        self._scoring_report = scoring_report
        self._model_loss = self._get_model_loss()

        self._model_quality = self._calculate_model_quality()

    def _get_model_loss(self):
        metric = Metrics.supported_metrics()[self.scoring_report.target_metric]

        if metric.ptype == REGRESSION and Metrics.normalized_rmse.name in self.scoring_report.metric_scores:
            loss = self.scoring_report.metric_scores[Metrics.normalized_rmse.name].value
        else:
            loss = Metrics.metric_as_loss(
                self.scoring_report.metric_scores[self.scoring_report.target_metric].value,
                self.scoring_report.target_metric)

        return loss

    def _calculate_model_quality(self):
        model_quality = 1.0
        model_quality = model_quality \
                        - self._vulnerability_report.imputation \
                        - self.model_loss
        model_quality = max(model_quality, 0)

        return model_quality

    def to_dict(self):
        return dict(
            vulnerability_report=self.vulnerability_report.to_dict(),
            model_loss=self.model_loss,
            model_quality=self.model_quality,
        )

    @classmethod
    def to_dict_meaning(cls):
        return dict(
            vulnerability_report=SensitivityVulnerabilityReport.to_dict_meaning(),
            model_loss="Error of the model - lower is better",
            model_quality="Overall model quality - higher is better")

    @property
    def vulnerability_report(self) -> SensitivityVulnerabilityReport:
        return self._vulnerability_report

    @property
    def model_quality(self):
        return self._model_quality

    @property
    def scoring_report(self):
        return self._scoring_report

    @property
    def model_loss(self):
        return self._model_loss



class QualityReport():
    def __init__(self, train_quality_report: TrainSetQualityReport, test_quality_report: TestSetQualityReport,
                 model_quality_report: ModelQualityReport):
        self._train_quality_report = train_quality_report
        self._test_quality_report = test_quality_report
        self._model_quality_report = model_quality_report

    def to_dict(self):
        return dict(
            test_quality_report=self.test_quality_report.to_dict(),
            train_quality_report=self.train_quality_report.to_dict(),
            model_quality_report=self.model_quality_report.to_dict(),
        )

    @classmethod
    def to_dict_meaning(cls):
        return dict(
            test_quality_report=TestSetQualityReport.to_dict_meaning(),
            train_quality_report=TrainSetQualityReport.to_dict_meaning(),
            model_quality_report=ModelQualityReport.to_dict_meaning(),
        )

    @property
    def train_quality_report(self):
        return self._train_quality_report

    @property
    def test_quality_report(self):
        return self._test_quality_report

    @property
    def model_quality_report(self):
        return self._model_quality_report
