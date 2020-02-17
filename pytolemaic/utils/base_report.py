from pytolemaic.utils.general import GeneralUtils


class Report():
    def to_dict(self, printable=False):
        raise NotImplementedError

    @classmethod
    def to_dict_meaning(cls):
        raise NotImplementedError

    def plot(self):
        raise NotImplementedError

    @classmethod
    def _printable_dict(cls, out, printable=False):
        if printable:
            return GeneralUtils.make_dict_printable(out)
        else:
            return out

