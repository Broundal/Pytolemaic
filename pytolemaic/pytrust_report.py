import itertools
import time

from pytolemaic.analysis_logic.dataset_analysis.anomaly_values_in_data_report import AnomaliesInDataReport
from pytolemaic.analysis_logic.dataset_analysis.dataset_analysis_report import DatasetAnalysisReport
from pytolemaic.analysis_logic.model_analysis.scoring.scoring_report import ScoringFullReport
from pytolemaic.analysis_logic.model_analysis.sensitivity.sensitivity_reports import SensitivityFullReport
from pytolemaic.analysis_logic.quality_report import QualityReport

from pytolemaic.utils.base_report import Report
from pytolemaic.utils.general import get_logger

logger = get_logger(__name__)



class PyTrustReport(Report):

    def __init__(self, pytrust):
        self.pytrust = pytrust

    @classmethod
    def _get_report(cls, pytrust, attr, verbose=True):

        ts = time.time()
        try:
            # some report may fail due to lack of available information
            if verbose:
                logger.info("Calculating {}...".format(attr))
            out = getattr(pytrust, attr, None)
            if verbose:
                logger.info("Calculating {}... Done ({:.1f} seconds)".format(attr, time.time() - ts))
        except:
            logger.info("Failed to calculate {}".format(attr))
            out = None

        return out

    @property
    def _reports(self):
        logger.info("Calculating all possible reports, this may take some time.\n"
                    "It's possible to access the reports directly through the following attributes:\n"
                    +"\n".join(["* pytrust.{} or pytrust.report.{}".format(key, key)
                                for key in self.to_dict_meaning().keys()]))
        return {key: self._get_report(self.pytrust, key, verbose=True) for key in self.to_dict_meaning().keys()}

    def to_dict(self, printable=False):
        return {key: report.to_dict(printable=printable) for key, report in self._reports.items() if
                report is not None}

    @classmethod
    def to_dict_meaning(cls):
        return {'scoring_report': ScoringFullReport.to_dict_meaning(),
                'quality_report': QualityReport.to_dict_meaning(),
                'dataset_analysis_report': DatasetAnalysisReport.to_dict_meaning(),
                'sensitivity_report': SensitivityFullReport.to_dict_meaning(),
                'anomalies_in_data_report': AnomaliesInDataReport.to_dict_meaning()
                }

    def plot(self):
        for report in self._reports.values():
            if report is not None:
                report.plot()

    def insights(self) -> list:
        reports = self._reports.copy()
        reports.pop('quality_report')
        l = [report.insights() for report in reports.values() if report is not None]
        if any(l):
            return list(itertools.chain(*l))
        else:
            return []

    def __getattr__(self, item):
        return self._get_report(item, verbose=False)