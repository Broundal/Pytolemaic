from pytolemaic.utils.general import GeneralUtils
from typing import List

class Report():
    def to_dict(self, printable=False):
        raise NotImplementedError

    @classmethod
    def to_dict_meaning(cls)->dict:
        raise NotImplementedError

    def plot(self)->None:
        raise NotImplementedError

    def insights(self)->List[str]:
        raise NotImplementedError

    def _add_cls_name_prefix(self, l):
        prefix = str(type(self).__name__)
        return [prefix + ': ' + item for item in l]

    @classmethod
    def _printable_dict(cls, out, printable=False):
        if printable:
            return GeneralUtils.make_dict_printable(out)
        else:
            return out
