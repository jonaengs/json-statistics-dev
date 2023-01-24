from collections import namedtuple
from dataclasses import asdict, dataclass
from enum import Enum
import json


class StatType(Enum):
    BASIC = 1
    BASIC_NDV = 2
    HYPERLOG = 3
    HISTOGRAM = 4


@dataclass
class KeyStat:
    count: int = 0
    null_count: int = 0
    min_val: (None | int) = None
    max_val: (None | int) = None
    ndv: (None | int) = None
    histogram: (None | list[int]) = None

    def __repr__(self) -> str:
        return str({k:v for k, v in asdict(self).items() if v is not None})

    @property
    def valid_count(self):
        return self.count - self.null_count


class KeyStatEncoder(json.JSONEncoder):
    def default(self, o):
        if type(o) == KeyStat:
            # return asdict(o)
            return {k:v for k, v in asdict(o).items() if v is not None}  # exclude None-fields
        return super().default(o)   

HistBucket = namedtuple("HistBucket", ["upper_bound", "count", "ndv"])
