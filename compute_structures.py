from collections import namedtuple
from dataclasses import asdict, dataclass
from enum import Enum
import json

class StatType(Enum):
    BASIC = 1
    BASIC_NDV = 2
    HYPERLOG = 3
    HISTOGRAM = 4


class PruneStrat(Enum):
    MIN_FREQ = 1
    MAX_NO_PATHS = 2
    MAX_PREFIX_LENGTH = 3

HistBucket = namedtuple("HistBucket", ["upper_bound", "count", "ndv"])

@dataclass
class KeyStat:
    count: int = 0
    null_count: int = 0
    min_val: (None | int) = None
    max_val: (None | int) = None
    ndv: (None | int) = None
    histogram: (None | list[HistBucket]) = None

    def __repr__(self) -> str:
        return str({k:v for k, v in asdict(self).items() if v is not None})

    @property
    def valid_count(self):
        return self.count - self.null_count


class KeyStatEncoder(json.JSONEncoder):
    def default(self, o):
        if type(o) == KeyStat:
            return {k:v for k, v in asdict(o).items() if v is not None}  # exclude None-valued fields
        if isinstance(o, Enum):
            return o.name

        return super().default(o)   

