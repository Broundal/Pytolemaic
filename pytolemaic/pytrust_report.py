import itertools

from pytolemaic.analysis_logic.dataset_analysis.dataset_analysis_report import DatasetAnalysisReport
from pytolemaic.analysis_logic.model_analysis.scoring.scoring_report import ScoringFullReport
from pytolemaic.analysis_logic.model_analysis.sensitivity.sensitivity_reports import SensitivityFullReport
from pytolemaic.dataset_quality_report import QualityReport
from pytolemaic.utils.base_report import Report


class PyTrustReport(Report):

    def __init__(self, pytrust):
        self.pytrust = pytrust

    @classmethod
    def _try_catch_report(cls, pytrust, attr):

        try:
            # some report may fail due to lack of available information
            out = getattr(pytrust, attr, None)
        except:
            out = None

        return out

    def _reports(self):
        return {key: self._try_catch_report(self.pytrust, key) for key in self.to_dict_meaning().keys()}

    def to_dict(self, printable=False):
        return {key: report.to_dict(printable=printable) for key, report in self._reports().items() if
                report is not None}

    @classmethod
    def to_dict_meaning(cls):
        return {'scoring_report': ScoringFullReport.to_dict_meaning(),
                'quality_report': QualityReport.to_dict_meaning(),
                'dataset_analysis_report': DatasetAnalysisReport.to_dict_meaning(),
                'sensitivity_report': SensitivityFullReport.to_dict_meaning(),
                }

    def plot(self):
        for report in self._reports().values():
            if report is not None:
                report.plot()

    def insights(self) -> list:
        reports = self._reports()
        reports.pop('quality_report')
        l = [report.insights() for report in reports.values() if report is not None]
        if any(l):
            return list(itertools.chain(*l))
        else:
            return []
