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
