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

    QUALITY = 12
    LEAKAGE = 3
    IMPUTATION = 4
    OVERFIIT = 5


    N_FEATURES = 8
    N_LOW = 9
    N_ZERO = 10
    N_NON_ZERO = 11




