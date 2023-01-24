from dataclasses import dataclass, asdict
import json
from collections import defaultdict, namedtuple
import math
import os
import random
from typing import Any, Callable
from hyperloglog import HyperLogLog

import struct
from compute_structures import HistBucket, KeyStat, KeyStatEncoder, StatType

from trackers import time_tracker, local_mem_tracker, global_mem_tracker
from settings import settings
from logger import log



STATS_TYPE = settings.stats.stat_type
HYPERLOGLOG_ERROR = settings.stats.hyperloglog_error
SAMPLING_RATIO = 0


# TODO: Check that the case of repeat keys ({a: 1, a: 2}) is covered. Should be fine as long a json library is used
# TODO: Figure out a key path format that is unlikely to collide with existing keys
    # Problem: {"a": {"a": []}} and {"a_dict.a": []} gives "a_dict.a_list" = 2
    # Maybe dashes can be a good idea to use, in addition to dots. Because they will be interpreted as minus signs they shouldn't be used as member names
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
    Problem: using sampling, we can't really know if all values associated with a key are ints. 
* Q: Do we track min and max values for strings? Less useful than for numbers, and more expensive to calculate (and possibly store)
    I say no, for now.
"""
# ==========================================================================================


# THE PLAN:
# Generate statistics for document collection
# Compute actual selectivity of some clause, then compare to the estimation
    # First: Compare basics like no. els, no. null, min val, max val
    # Then: Compare for clauses like ... > 3, ... = "asdasd",
        # TODO:: check clauses which demand a string, a number, an object or an array
    # Final (unrealistic): Compare for clauses like JOIN WHERE A.B = ...



# VARIABLES that can be tuned
# Histogram size  (look into histogram techniques. Can we optimize for uniformity, for example?)
# Statistics approach
# Pruning approach (also: combinations of them, like prefix/postfix length and min freq)
# Sampling ratio
# (Kinda) Data set makeup (heterogeneity, sparseness)
# Combining floats and ints into "number"

# Undersøk: sampling osv. 
# Compression av nøkler. Veldig like nøkler rett ved siden av hverandre burde ikke være så vanskelig
    # Typ: "retweeted_status_object.entities_object.hashtags_array.1_object.indices_array.0_int" 
    # og   "retweeted_status_object.entities_object.hashtags_array.1_object.indices_array.1_int"

@time_tracker.record_time_used
def get_operation_cardinality(file_path, key_path, operation: Callable[[Any], bool]):
    def traverse(doc, path: list[str]):
        if not path: 
            return doc
        return traverse(doc[path[0]], path[1:])


    with open(file_path) as f:
        collection = json.load(f)

    count = 0
    for doc in collection:
        try:
            count += operation(traverse(doc, key_path))
        except:
            pass

    return count


# NOTE: Only works for floats and ints
def compare_lt_estimate(data_path, key_path: list[str], stats, stat_path: str, compare_value):
    def get_cardinality_estimate():
        # return compare_range_estimate(data_path, key_path, stats, stat_path, range(stats[stat_path].min_val, compare_value))
        
        data = stats[stat_path]
        if compare_value <= data.min_val:
            return 0

        match STATS_TYPE:
            case StatType.BASIC | StatType.BASIC_NDV | StatType.HYPERLOG:
                data_range = data.max_val - data.min_val
                compare_range = (compare_value - 1) - data.min_val
                overlap = compare_range / data_range

                return int(overlap * data.valid_count)
            case StatType.HISTOGRAM:
                # NOTE: Assumed bucket structure [upper_bound, val_count, ndv]

                estimate = 0
                bucket_lower_bound = data.min_val
                for bucket in map(lambda b: HistBucket(*b), data.histogram):
                    if bucket.upper_bound < compare_value:
                        estimate += bucket.count
                    elif bucket.upper_bound >= compare_value:
                        bucket_range = bucket.upper_bound - bucket_lower_bound
                        valid_value_range = (compare_value - 1) - bucket_lower_bound
                        overlap = valid_value_range / bucket_range
                        estimate += bucket.count * overlap
                        break

                    bucket_lower_bound = bucket.upper_bound


                return int(estimate)

    true_card = get_operation_cardinality(data_path, key_path, lambda x: x < compare_value)
    estimate = get_cardinality_estimate()
    
    log(true_card, estimate)

# NOTE: Only works for floats and ints
# Assume range is inclusive lower excluse upper, like a python range
def compare_range_estimate(data_path, key_path: list[str], stats: dict, stat_path: str, q_range: range):
    def get_cardinality_estimate():
        data: KeyStat = stats[stat_path]

        match STATS_TYPE:
            case StatType.BASIC | StatType.BASIC_NDV | StatType.HYPERLOG:
                valid_range = min(data.max_val, q_range.stop - 1) - max(data.min_val, q_range.start)
                data_range = data.max_val - data.min_val
                overlap = valid_range / (data_range)

                return int(overlap * data.valid_count)

            case StatType.HISTOGRAM:
                # NOTE: Assumed bucket structure [upper_bound, val_count, ndv]

                estimate = 0
                bucket_lower_bound = data.min_val
                for bucket in map(lambda b: HistBucket(*b), data.histogram):
                    valid_range = min(bucket.upper_bound, q_range.stop - 1) - max(bucket_lower_bound, q_range.start)
                    bucket_range = bucket.upper_bound - bucket_lower_bound
                    overlap = valid_range / bucket_range
                    
                    estimate += overlap * bucket.count

                    bucket_lower_bound = bucket.upper_bound
                    
                    if bucket_lower_bound >= q_range.stop:
                        break

                return int(estimate)
    
    true_card = get_operation_cardinality(data_path, key_path, lambda x: x in q_range)
    estimate = get_cardinality_estimate()
    
    log(true_card, estimate)

# NOTE: Only works for floats and ints
def compare_eq_estimate(data_path, key_path: list[str], stats, stat_path: str, compare_value):
    def get_cardinality_estimate():
        data = stats[stat_path]
        if compare_value > data.max_val or compare_value < data.min_val:
            return 0

        match STATS_TYPE:
            case StatType.BASIC:
                return None
            case StatType.BASIC_NDV | StatType.HYPERLOG:
                return data.valid_count//data.ndv

            case StatType.HISTOGRAM:
                Bucket = namedtuple("Bucket", ["upper_bound", "count", "ndv"])

                for bucket in map(lambda b: Bucket(*b), data.histogram):
                    if bucket.upper_bound >= compare_value:
                        # Assume uniform distribution, so return count divided by ndv
                        return int(bucket.count / bucket.ndv)

                return estimate

    true_card = get_operation_cardinality(data_path, key_path, lambda x: x == compare_value)
    estimate = get_cardinality_estimate()
    
    log(true_card, estimate)


def compare_card_estimate(data_path, key_path: list[str], stats, stat_path: str):
    def get_cardinality_estimate():
        data = stats[stat_path]
        return data.count

    true_card = get_operation_cardinality(data_path, key_path, lambda:1)
    estimate = get_cardinality_estimate()
    
    log(true_card, estimate)


def compute_ndv(path, accessor):
    with open(path) as f:
        collection = json.load(f)

    els = set()
    for doc in collection:
        try:
            els.add(accessor(doc)) 
        except:
            pass

    return len(els)



# Creates an equi-height histogram for the data in the array
# Bucket structure is as follows: [upper_bound, count, ndv]. Upper bound is inclusive
# Very naive algorithm. See MySQL src (sql/histograms/equi_height.cc) for a better approach
# Note: Mutates the argument array
@local_mem_tracker.record_peak_memory
@time_tracker.record_time_used
def compute_histogram(arr, nbins=200):
    min_bucket_size = math.ceil(len(arr)/nbins)
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


@local_mem_tracker.record_peak_memory
@time_tracker.record_time_used
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
            # int: "number", float: "number"
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
    global_mem_tracker.record_global_memory()
    for doc in collection:
        # TODO: Should maybe be done in blocks of N docs at a time, to emulate page sampling as used in real systems
        # TODO: Is python's random faulty in any way? It should be fine for this purpose, right?
        if (random.random() > SAMPLING_RATIO):
            traverse(doc)
    
    global_mem_tracker.record_global_memory()
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
        

    global_mem_tracker.record_global_memory()

    return dict(stats)


# Removes uncommon paths. Returns some summary statistics as well
def make_statistics(collection) -> list[dict, dict]:
    # Tunable vars:
    MIN_FREQ_THRESHOLD = 0.00
    # MAX_NUM_PATHS_TRACKED
    # MAX_PATH_DEPTH (seems terrible, but eh)
    # Max Prefix length, Max postfix length (prune middle keys)
        # Look at what JSON PATH in MySQL allows. Like wildcards

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

    summary_stats = {
        "min_count_included": min_count_included,
        "min_count_threshold": min_count_threshold,
        "collection_size": len(collection),
        "stats_type": STATS_TYPE.name,
        "sampling_ratio": SAMPLING_RATIO,
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

    data_path = settings.stats.data_path
    out_path = settings.stats.out_path

    def store_stats():
        log("Creating new stats...")
        with open(data_path) as f:
            stats = make_statistics(json.load(f))

        log("Storing new stats...")
        with open(out_path, mode="w") as f:
            log(len(json.dumps(stats, cls=KeyStatEncoder)))
            json.dump(stats, f, cls=KeyStatEncoder)

        return stats

    if os.path.exists(out_path) and not settings.stats.force_new:
        with open(out_path, "r") as f:
            existing_stats = json.load(f)
            if existing_stats and existing_stats[1]["stats_type"] == STATS_TYPE.name:
                stats = existing_stats
                log("Using pre-existing stats...")
                for k in stats[0]:
                    stats[0][k] = KeyStat(**stats[0][k])

            else:
                stats = store_stats()
    else:
        stats = store_stats()


    # "entities_object.hashtags_array.0_object.indices_array.0_int"
    stat_path = "entities_object.hashtags_array.0_object.indices_array.0_int"
    key_path = ["entities", "hashtags", 0, "indices", 0]
    compare_lt_estimate(data_path, key_path, stats[0], stat_path, 70)
    compare_eq_estimate(data_path, key_path, stats[0], stat_path, 100)
    compare_range_estimate(data_path, key_path, stats[0], stat_path, range(-1, 70))

    # log(compute_cardinality(data_path, lambda d: d["entities"]["urls"][0]["url"]))
    # log(get_stats_data(out_path, lambda d: d["entities_object.urls_array.0_object.url_str"])["count"])
    # log()
    
    # log(compute_ndv(data_path, lambda d: d["entities"]["urls"][0]["url"]))
    # log(get_stats_data(out_path, lambda d: d["entities_object.urls_array.0_object.url_str"])["ndv"])
    # log()

    # # "entities_object.hashtags_array.0_object.text_str"
    # log(compute_ndv(data_path, lambda d: d["entities"]["hashtags"][0]["text"]))
    # log(get_stats_data(out_path, lambda d: d["entities_object.hashtags_array.0_object.text_str"])["ndv"])
    # log()