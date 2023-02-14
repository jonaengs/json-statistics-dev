from collections import namedtuple
from dataclasses import asdict, dataclass
from enum import Enum
import json

Json_Number = int | float
Json_Null = type(None)
Json_Primitive = Json_Number | bool | str | Json_Null
Json_Primitive_No_Null = Json_Number | bool | str 
Json_value = Json_Primitive | list | dict

class StatType(Enum):
    BASIC = 1
    BASIC_NDV = 2
    NDV_HYPERLOG = 3
    NDV_WITH_MODE = 4
    HISTOGRAM = 5


class PruneStrat(Enum):
    MIN_FREQ = 1
    MAX_NO_PATHS = 2
    MAX_PREFIX_LENGTH = 3
    UNIQUE_SUFFIX = 4

HistBucket = namedtuple("HistBucket", ["upper_bound", "count", "ndv"])
ModeInfo = namedtuple("ModeInfo", ["value", "count"])

@dataclass
class KeyStat:
    count: int = 0
    null_count: int = 0
    min_val: (None | int) = None
    max_val: (None | int) = None
    ndv: (None | int) = None
    histogram: (None | list[HistBucket]) = None
    mode_info: (None | ModeInfo) = None

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

