from dataclasses import dataclass, asdict
from enum import Enum
import json
from collections import defaultdict, namedtuple
import math
from hyperloglog import HyperLogLog

import struct
from compute_structures import StatType

from utils import timer, print_total_memory_use, mem_tracker

from settings import settings

from logger import log



STATS_TYPE = settings.stats.stat_type
HYPERLOGLOG_ERROR = settings.stats.hyperloglog_error
log("compute: ", STATS_TYPE)


# TODO: Check that the case of repeat keys ({a: 1, a: 2}) is covered. Should be fine as long a json library is used
# TODO: Figure out a key path format that is unlikely to collide with existing keys
    # Problem: {"a": {"a": []}} and {"a_dict.a": []} gives "a_dict.a_list" = 2
# TODO: Instead of parsing json files. Parse MySQL's json binary format

# ==========================================================================================
# ==================================================
# Statistics design
# ==================================================
"""
* Q: Keep only most frequent type for all key paths (like JSON Tiles)? What if two types are split 50/50 in freq?
* Idea: Calculate some mean cardinality of all pruned (due to infrequency) key-paths, and use that as a stand-in
* Q: Do we keep statistics for inner nodes, or just leaf/primitive nodes? Users can of course query for inner nodes as well
    We can likely reconstruct inner node statistics by looking at children node statistics. 
        Combined with pruning of infrequent keypaths, this can cause an issue where a parent with lots of different children
        had all its children pruned and so cannot be reconstructed, while in actuality having a very high frequency
* Q: Can we do any of the pattern mining that JSON Tiles does?
* Q: Should we differentiate between floats and ints? JSON doesn't, and MySQL doesn't when querying. So we shouldn't either 
    On the other hand, histograms are much simpler to create for ints than for floats. 
    So if all vals in a keypath are ints, that could enable better/simpler statistics
* Q: Do we track min and max values for strings? Less useful than for numbers, and more expensive to calculate (and possibly store)
    I say no, for now.
"""
# ==========================================================================================


# THE PLAN:
# Generate statistics for document collection
# Compute actual selectivity of some clause, then compare to the estimation
    # First: Compare basics like no. els, no. null, min val, max val
    # Then: Compare for clauses like ... > 3, ... = "asdasd",
    # Final (unrealistic): Compare for clauses like JOIN WHERE A.B = ...


def compute_cardinality(path, accessor):
    with open(path) as f:
        docs = json.load(f)

    count = 0
    for doc in docs:
        try:
            accessor(doc)
            count += 1
        except:
            pass

    return count

def compute_ndv(path, accessor):
    with open(path) as f:
        docs = json.load(f)

    els = set()
    for doc in docs:
        try:
            els.add(accessor(doc)) 
        except:
            pass

    return len(els)

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


class KeyStatEncoder(json.JSONEncoder):
    def default(self, o):
        if type(o) == KeyStat:
            # return asdict(o)
            return {k:v for k, v in asdict(o).items() if v is not None}  # exclude None-fields
        return super().default(o)


HistBucket = namedtuple("HistBucket", ["upper_bound", "height", "ndv"])


# Creates an equi-height histogram for the data in the array
# Very naive algorithm. See MySQL src (sql/histograms/equi_height.cc) for a better approach
# Note: Mutates the argument array
@mem_tracker.record_peak_memory
@timer.record_time_used
def compute_histogram(arr, nbins=10):
    min_bucket_size = len(arr)/nbins
    arr.sort()

    histogram = []
    counter = 1
    prev_el = arr[0]
    bucket = [prev_el]
    for el in arr[1:]:
        if counter >= min_bucket_size and el != prev_el:
            histogram.append(
                HistBucket(prev_el, len(bucket), len(set(bucket)))
            )
            bucket = [el]
            counter = 1
        else:
            bucket.append(el)
            counter += 1

        prev_el = el

    histogram.append(
        HistBucket(prev_el, len(bucket), len(set(bucket)))
    )

    return histogram


@mem_tracker.record_peak_memory
@timer.record_time_used
def make_base_statistics(collection):
    """
    STATS TYPES:
    1. count, null_count. For int & float: min and max. 
        Simply counts up as values are encountered. So there is little memory overhead
    2. 1, but also keeps track of the ratio of unique values.
        This requires storing all values encountered, so memory overhead (and computation cost) is higher
    3. 2, but using HyperLogLog to estimate the number of distinct values
    4. Histograms. 
        Naive approach, so this will again require us to store all values
    """

    def make_key_path(_parent_path, key, val)-> tuple[str, str]:
        type_str = {
            list: "array",
            dict: "object",
            None.__class__: "", 
        }.get(type(val), type(val).__name__)

        parent_path = _parent_path + (_parent_path and ".")

        return parent_path + str(key) + ("_" + type_str if type_str else ""), parent_path + str(key)
    
    def record_path_stats(stats, parent_path, key, val) -> str:
        match STATS_TYPE:
            case StatType.BASIC:
                # record keypath both with and without type information
                key_str, base_key_str = make_key_path(parent_path, key, val)
                stats[key_str].count += 1
                stats[base_key_str].count += val is not None  # Change this if null values get a type suffix
                stats[key_str].null_count += val is None

                if type(val) == int or type(val) == float:
                    stats[key_str].min_val = min(val, stats[key_str].min_val or math.inf)
                    stats[key_str].max_val = max(val, stats[key_str].max_val or -math.inf)

                return key_str
            case StatType.BASIC_NDV | StatType.HISTOGRAM:
                # stats[key] is either a list or a KeyStat object
                key_str, base_key_str = make_key_path(parent_path, key, val)
                
                # Handle base_key_str here. key_str is handled differently for 
                if base_key_str not in stats:
                    stats[base_key_str] = KeyStat()
                stats[base_key_str].count += val is not None  # Change this if null values get a type suffix

                if type(val) in (int, float, str):
                    # Store all primitive values
                    if key_str not in stats:
                        stats[key_str] = []

                    stats[key_str].append(val)
                else:
                    # Count arrays, objects and nulls
                    if key_str not in stats:
                        stats[key_str] = KeyStat()

                    stats[key_str].count += 1
                    stats[key_str].null_count += val is None

                return key_str

            case StatType.HYPERLOG:
                # Very similar to basic, but we 
                key_str, base_key_str = make_key_path(parent_path, key, val)
                stats[key_str].count += 1
                stats[base_key_str].count += val is not None  # Change this if null values get a type suffix
                stats[key_str].null_count += val is None

                if type(val) in (int, float, str):
                    if not hasattr(stats[key_str], "hll"):
                        stats[key_str].hll = HyperLogLog(HYPERLOGLOG_ERROR)
                    
                    hll_val = int.to_bytes(8) if type(val) == int else struct.pack("!f", val) if type(val) == float else val
                    stats[key_str].hll.add(hll_val)

                    if type(val) == int or type(val) == float:
                        stats[key_str].min_val = min(val, stats[key_str].min_val or math.inf)
                        stats[key_str].max_val = max(val, stats[key_str].max_val or -math.inf)

                return key_str
            case _:
                assert False, f"statistics {STATS_TYPE} type not supported"
            

    
    stats = {
        StatType.BASIC: defaultdict(KeyStat),
        StatType.BASIC_NDV: dict(), # stats[key] is either a list or a KeyStat object
        StatType.HYPERLOG: defaultdict(KeyStat), # stats[key] is either a list or a KeyStat object
        StatType.HISTOGRAM: dict() # stats[key] is either a list or a KeyStat object
    }[STATS_TYPE]

    def traverse(doc, parent_path=""):

        if type(doc) == list:
            for key, val in enumerate(doc):  # use index as key
                new_parent_path = record_path_stats(stats, parent_path, key, val)
                
                if type(val) == list or type(val) == dict:
                    traverse(val, new_parent_path)

        if type(doc) == dict:
            for key, val in doc.items():
                new_parent_path = record_path_stats(stats, parent_path, key, val)

                if type(val) == list or type(val) == dict:
                    traverse(val, new_parent_path)


    log(f"creating statistics for a collection of {len(collection)} documents...")
    print_total_memory_use()
    for doc in collection:
        traverse(doc)
    
    print_total_memory_use()
    if STATS_TYPE == StatType.BASIC_NDV:
        # Compute KeyStat object for primitive value lists
        for key in stats:
            if type(stats[key]) == list:
                vals = stats[key]
                stats[key] = KeyStat(  # obvious performance improvement: Calculate these in a single pass instead of four
                    count=len(vals),
                    null_count=0,
                    # Don't calculate min and max for strings
                    min_val=min(vals) if type(vals[0]) != str else None,
                    max_val=max(vals) if type(vals[0]) != str else None,
                    ndv=len(set(vals))
                )  
    elif STATS_TYPE == StatType.HYPERLOG:
        # Compute ndv estimate from hyperloglog object
        for val in stats.values():
            if hasattr(val, "hll"):
                val.ndv = len(val.hll)
                del val.hll
    
    elif STATS_TYPE == StatType.HISTOGRAM:
        # Compute basic stats
        for key in stats:
            if type(stats[key]) == list:
                vals = stats[key]
                histogram = compute_histogram(vals) if type(stats[key][0]) != str else None  # drop string histograms for now
                stats[key] = KeyStat(  # obvious performance improvement: Calculate these in a single pass instead of four
                    count=len(vals),
                    null_count=0,
                    # Don't calculate min and max for strings
                    min_val=min(vals) if type(vals[0]) != str else None,
                    max_val=max(vals) if type(vals[0]) != str else None,
                    ndv=len(set(vals)),
                    histogram=histogram
                )
        

    print_total_memory_use()

    return dict(stats)


# Removes uncommon paths. Returns some summary statistics as well
def make_statistics(collection) -> list[dict, dict]:
    # Tunable vars:
    MIN_FREQ_THRESHOLD = 0.001
    # MAX_NUM_PATHS_TRACKED
    # MAX_PATH_DEPTH (seems terrible, but eh)

    base_stats = make_base_statistics(collection)

    min_count_threshold = int(MIN_FREQ_THRESHOLD * len(collection))

    pruned_path_stats = {}
    min_count_included = None
    for key_path, path_stats in base_stats.items():
        if path_stats.count >= min_count_threshold:
            min_count_included = min(min_count_included or math.inf, path_stats.count)
            pruned_path_stats[key_path] = path_stats

    
    num_pruned = len(base_stats) - len(pruned_path_stats)
    log("num_pruned", num_pruned, f"({len(base_stats)} unique paths total)")

    # TODO: Store sampling ratio/frequency?
    summary_stats = {
        "min_count_included": min_count_included,
        "min_count_threshold": min_count_threshold,
        "collection_size": len(collection)
    }
    return [pruned_path_stats, summary_stats]


# Maybe there should be one of these for each approach
def get_stats_data(path, accessor):
    with open(path) as f:
        path_stats, summary_stats = json.load(f)

    try:
        return accessor(path_stats)
    except:
        return {"count": summary_stats["min_count_threshold"]}


def run():
    log(f"using {StatType(STATS_TYPE).name} StatType")

    data_path = f"data/recsys/{settings.stats.filename}.json"
    stats_path = f"stats/{settings.stats.filename}.json"

    with open(data_path) as f:
        stats = make_statistics(json.load(f)[:1000])
        # plog(stats[0])
        # log("-"*50)
        # plog(stats[1])
        # log("-"*50)

    with open(stats_path, mode="w") as f:
        log(len(json.dumps(stats, cls=KeyStatEncoder)))
        json.dump(stats, f, cls=KeyStatEncoder)

    # log(compute_cardinality(data_path, lambda d: d["entities"]["urls"][0]["url"]))
    # log(get_stats_data(stats_path, lambda d: d["entities_object.urls_array.0_object.url_str"])["count"])
    # log()
    
    # log(compute_ndv(data_path, lambda d: d["entities"]["urls"][0]["url"]))
    # log(get_stats_data(stats_path, lambda d: d["entities_object.urls_array.0_object.url_str"])["ndv"])
    # log()

    # # "entities_object.hashtags_array.0_object.text_str"
    # log(compute_ndv(data_path, lambda d: d["entities"]["hashtags"][0]["text"]))
    # log(get_stats_data(stats_path, lambda d: d["entities_object.hashtags_array.0_object.text_str"])["ndv"])
    # log()