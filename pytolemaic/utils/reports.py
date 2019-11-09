from enum import Enum


class EnumBase(Enum):

    @classmethod
    def keys(cls, members=False):
        if members:
            return [item for item in cls]
        else:
            return [item.name for item in cls]

    @classmethod
    def values(cls):
        return [item.value for item in cls]


    @classmethod
    def asdict(cls):
        return {item.name: item.value for item in cls}


class ReportScoring(EnumBase):
    CI_LOW = 1
    CI_HIGH = 2
    SCORE_VALUE = 3
    QUALITY = 4

class ReportSensitivity(EnumBase):
    SHUFFLE = 1
    MISSING = 2
    SENSITIVITY = 6
    META = 7
    LEAKAGE = 3
    IMPUTATION = 4
    OVERFIIT = 5

    N_FEATURES = 8
    N_LOW = 9
    N_ZERO = 10
    N_NON_ZERO = 11


class Report(object):

    def __init__(self, report: dict):

        # flatten to to simple dict
        for k, v in report.items():
            if isinstance(v, Report):
                report[k] = v.report

        self.report = report

    def get(self, key):
        def get_from_report(report, lookup_key):
            if lookup_key in report:
                value = report[lookup_key]
                if isinstance(value, dict):
                    return Report(value)
                else:
                    return report[lookup_key]
            else:
                for key, value in report.items():
                    if isinstance(value, dict):
                        recursion = get_from_report(value, lookup_key)
                        if recursion is not None:
                            return recursion

            # key not found
            return None

        return get_from_report(self.report, lookup_key=key)

    def __repr__(self):
        return self.report.__repr__()


