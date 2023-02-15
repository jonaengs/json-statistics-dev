import json
from collections import Counter, defaultdict, namedtuple
import math
import os
import random
import re
from typing import Any, Callable
from hyperloglog import HyperLogLog

import struct
from compute_structures import EquiHeightBucket, Json_Primitive_No_Null, KeyStat, KeyStatEncoder, ModeInfo, PruneStrat, StatType

from trackers import time_tracker, local_mem_tracker, global_mem_tracker
from settings import settings
from logger import log

import stats_cache
import data_cache


# TODO: Check that the case of repeat keys ({a: 1, a: 2}) is covered. Should be fine as long a json library is used
# TODO: Figure out a key path format that is unlikely to collide with existing keys
    # Problem: {"a": {"a": []}} and {"a_dict.a": []} gives "a_dict.a_list" = 2
    # Maybe dashes can be a good idea to use, in addition to dots. Because they will be interpreted as minus signs they shouldn't be used as member names

# TODO?: Change from using cardinality to selectivity, accounting for sampling. 

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
* Q: Do we add support for date strings? In some ways, it should be simpler to support than regular stringss,
    but it is not required and the JSON RFC makes no mention of date strings

* Q: With numeric histograms, should we be smart when deciding whether to create a singleton histogram vs an 
    equi-height histogram? For example, if we encounter each value only once, it's quite unlikely 
    that all of the values would fit in a singleton histogram (assuming we're sampling heavily). In this case,
    using an equi-width histogram might be better as it likely represents the data better. On the other hand,
    if we encounter a low number of different values very many times, then its much more likely that
    all the values (again, assuming sampling) would fit inside a singleton histogram. 
    I'm sure there has to be some literature on this topic. Look it up!
* Q: Do we keep track of min and max strings? Could be expensive to perform all the comparisons. I guess we should test it

"""
# ==========================================================================================


# THE PLAN:
# Generate statistics for document collection
# Compute actual selectivity of some clause, then compare to the estimation
    # First: Compare basics like no. els, no. null, min val, max val
    # Then: Compare for clauses like ... > 3, ... = "asdasd",
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
            case StatType.BASIC | StatType.BASIC_NDV | StatType.NDV_HYPERLOG:
                data_range = data.max_val - data.min_val
                compare_range = (compare_value - 1) - data.min_val
                overlap = compare_range / data_range

                return int(overlap * data.valid_count)
            case StatType.HISTOGRAM:
                # NOTE: Assumed bucket structure [upper_bound, val_count, ndv]

                estimate = 0
                bucket_lower_bound = data.min_val
                for bucket in map(lambda b: EquiHeightBucket(*b), data.histogram):
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
            case StatType.BASIC | StatType.BASIC_NDV | StatType.NDV_HYPERLOG:
                valid_range = min(data.max_val, q_range.stop - 1) - max(data.min_val, q_range.start)
                data_range = data.max_val - data.min_val
                overlap = valid_range / (data_range)

                return int(overlap * data.valid_count)

            case StatType.HISTOGRAM:
                # NOTE: Assumed bucket structure [upper_bound, val_count, ndv]

                estimate = 0
                bucket_lower_bound = data.min_val
                for bucket in map(lambda b: EquiHeightBucket(*b), data.histogram):
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
                # TODO: Can we estimate something here?
                # For integers, data.valid_count / (data.max_val - data.min_val)
                    # Obviously this won't work for floats (as there are infinitely many values between whatever max and min are -- unless they are equal, of course)
                return None
                # assert False, "cannot estimate EQ for 
            case StatType.BASIC_NDV | StatType.NDV_HYPERLOG:
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
def compute_histogram(arr, _nbins=None) -> list[EquiHeightBucket] | None:
    nbins = _nbins or settings.stats.num_histogram_buckets
    min_bucket_size = math.ceil(len(arr)/nbins)
    arr.sort()

    # Try to create singleton-ish histogram if possible
    # Make sure to check after sorting the array to get a sorted histogram
    if len(set(arr)) <= nbins:
        return [EquiHeightBucket(v, c, 1) for v,c in Counter(arr).items()]
    elif type(arr[0]) == str:
        # Cannot make non-singleton string histograms
        return None

    histogram = []
    counter = 1
    prev_el = arr[0]
    bucket = [prev_el]
    for el in arr[1:]:
        if counter >= min_bucket_size and el != prev_el:
            histogram.append(
                EquiHeightBucket(prev_el, len(bucket), len(set(bucket)))
            )
            bucket = [el]
            counter = 1
        else:
            bucket.append(el)
            counter += 1

        prev_el = el

    histogram.append(
        EquiHeightBucket(prev_el, len(bucket), len(set(bucket)))
    )

    return histogram

type_suffixes = {
    list: "array",
    dict: "object",
    None.__class__: "",
    bool: "boolean",
    str: "string",
    int: "number", float: "number",
}

def make_key_path(_parent_path, key, val)-> tuple[str, str]:
    """returns key path with and without the value type suffix, respectively"""
    key_sep = settings.stats.key_path_key_sep 
    type_sep = settings.stats.key_path_type_sep

    type_str = type_suffixes[type(val)]

    # ands and if-elses are to avoid having a path start with the key_sep
    # or end with the type_sep 

    parent_path = _parent_path + (_parent_path and key_sep)

    key_path = parent_path + str(key) + (type_sep + type_str if type_str else "")
    base_path = parent_path + str(key)
    return key_path, base_path


@local_mem_tracker.record_peak_memory
@time_tracker.record_time_used
def _make_base_statistics(collection, _STATS_TYPE=None, _SAMPLING_RATE=None) -> dict[str, KeyStat]:
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
    STATS_TYPE = _STATS_TYPE or settings.stats.stats_type
    SAMPLING_RATE = _SAMPLING_RATE if _SAMPLING_RATE is not None else settings.stats.sampling_rate
    HYPERLOGLOG_ERROR = settings.stats.hyperloglog_error
    MAX_PREFIX_LENGTH = settings.stats.prune_params.max_prefix_length_threshold
    
    def record_path_stats(stats, parent_path, key, val) -> str:
        match STATS_TYPE:
            case StatType.BASIC:
                # record keypath both with and without type information
                key_str, base_key_str = make_key_path(parent_path, key, val)
                stats[key_str].count += 1
                stats[base_key_str].count += val is not None  # Change this if null values get a type suffix
                stats[key_str].null_count += val is None

                if type(val) in (int, float, bool):
                    stats[key_str].min_val = min(val, stats[key_str].min_val if stats[key_str].min_val is not None else math.inf)
                    stats[key_str].max_val = max(val, stats[key_str].max_val if stats[key_str].max_val is not None else -math.inf)

                return key_str
            case StatType.BASIC_NDV | StatType.HISTOGRAM | StatType.NDV_WITH_MODE:
                # stats[key] is either a list or a KeyStat object
                key_str, base_key_str = make_key_path(parent_path, key, val)
                
                # Handle common base_key logic
                if base_key_str not in stats:
                    stats[base_key_str] = KeyStat()
                stats[base_key_str].count += 1
                     
                # key_str is handled differently for non-null primitives, nulls and arrays/objects
                if isinstance(val, Json_Primitive_No_Null):                    
                    # Store all primitive values in a list
                    if key_str not in stats:
                        stats[key_str] = []
                    stats[key_str].append(val)
                elif val == None:
                    # Change if None get a suffix
                    assert key_str == base_key_str
                    stats[base_key_str].null_count += 1
                else:
                    # Count arrays and objects
                    if key_str not in stats:
                        stats[key_str] = KeyStat()

                    stats[key_str].count += 1

                return key_str
            case StatType.NDV_HYPERLOG:
                # Very similar to basic, but we 
                key_str, base_key_str = make_key_path(parent_path, key, val)
                stats[key_str].count += 1
                stats[key_str].null_count += val is None
                stats[base_key_str].count += val is not None  # Change this if null values get a type suffix

                # Add value to HyperLogLog structure
                # if type(val) in (int, float, str):
                if type(val) in (int, float, str, bool):
                    if not hasattr(stats[key_str], "hll"):
                        stats[key_str].hll = HyperLogLog(HYPERLOGLOG_ERROR)
                    
                    # hll_val = int.to_bytes(8) if type(val) == int else struct.pack("!f", val) if type(val) == float else val
                    # For some reason, int.to_bytes() seems to produce values that don't work nicely with hyperloglog
                    # Resulting in ndv=1 for sets of values with true ndv much(?) higher
                    hll_val = str(val) if type(val) == int else struct.pack("!f", val) if type(val) == float else val
                    stats[key_str].hll.add(hll_val)

                # Record min and max values
                if type(val) in (int, float, bool):
                    stats[key_str].min_val = min(val, stats[key_str].min_val if stats[key_str].min_val is not None else math.inf)
                    stats[key_str].max_val = max(val, stats[key_str].max_val if stats[key_str].max_val is not None else -math.inf)

                return key_str
            case _:
                assert False, f"statistics {STATS_TYPE} type not supported"
            

    stats = {
        StatType.BASIC: defaultdict(KeyStat),
        StatType.BASIC_NDV: dict(), # stats[key] is either a list or a KeyStat object
        StatType.NDV_HYPERLOG: defaultdict(KeyStat), # stats[key] is either a list or a KeyStat object
        StatType.NDV_WITH_MODE: dict(), # stats[key] is either a list or a KeyStat object
        StatType.HISTOGRAM: dict(), # stats[key] is either a list or a KeyStat object
    }[STATS_TYPE]

    def traverse(doc, parent_str_path="", parent_path=tuple()):
        def get_max_prefix_pruned_parent_paths(parent_str_path, parent_path):
            if PruneStrat.MAX_PREFIX_LENGTH in settings.stats.prune_strats and len(parent_path) == MAX_PREFIX_LENGTH:
                # Prune parent path to be max_length - 1, so max_length is reached when adding child
                parent_path = parent_path[1:]
                if MAX_PREFIX_LENGTH > 1:
                    type_sep = settings.stats.key_path_type_sep
                    key_sep = settings.stats.key_path_key_sep
                    re_special_chars = "^$.\|+*?{}[]()"
                    key_sep = '\\'+key_sep if key_sep in re_special_chars else key_sep

                    pattern = f".+{type_sep}[A-Za-z]+{key_sep}"
                    oldest_key = next(re.finditer(pattern, parent_str_path))
                    parent_str_path = parent_str_path[len(oldest_key.group()):]
                else:
                    parent_str_path = ""

            return parent_str_path, parent_path


        parent_str_path, parent_path = get_max_prefix_pruned_parent_paths(parent_str_path, parent_path)
        for key, val in (doc.items() if type(doc) == dict else enumerate(doc)):
            new_parent_str_path = record_path_stats(stats, parent_str_path, key, val)
            
            if type(val) == list or type(val) == dict:
                traverse(val, new_parent_str_path, parent_path=parent_path + (key, ))


    log(f"creating {STATS_TYPE.name} statistics for a collection of {len(collection)} documents...")
    global_mem_tracker.record_global_memory()
    for doc in collection:
        # TODO?: Should maybe be done in blocks of N docs at a time, to emulate page sampling as used in real systems
        # TODO: Is python's random faulty in any way? It should be fine for this purpose, right?
        if (random.random() >= SAMPLING_RATE):
            traverse(doc)
    
    global_mem_tracker.record_global_memory()
    if STATS_TYPE == StatType.BASIC_NDV:
        # Compute KeyStat object for primitive value lists
        for key in stats:
            if type(stats[key]) == list:  # List means that primitive values are recorded here
                vals = stats[key]
                stats[key] = KeyStat(  # obvious performance improvement: Calculate these in a single pass instead of four
                    count=len(vals),
                    null_count=0,
                    # Don't calculate min and max for strings
                    min_val=min(vals) if type(vals[0]) != str else None,
                    max_val=max(vals) if type(vals[0]) != str else None,
                    ndv=len(set(vals))
                )
    elif STATS_TYPE == StatType.NDV_WITH_MODE:
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
                    ndv=len(set(vals)),

                    mode_info=ModeInfo(*Counter(vals).most_common(1)[0]),
                    # Don't store mode_info for strings. Takes lots of space
                    # mode_info=(ModeInfo(*Counter(vals).most_common(1)[0]) if type(vals[0]) != str else None),
                )
    elif STATS_TYPE == StatType.NDV_HYPERLOG:
        # Compute ndv estimate from hyperloglog object
        for keystat in stats.values():
            if hasattr(keystat, "hll"):
                keystat.ndv = len(keystat.hll)
                del keystat.hll
    
    elif STATS_TYPE == StatType.HISTOGRAM:
        # Compute basic stats
        for key in stats:
            if type(stats[key]) == list:
                vals = stats[key]
                histogram = compute_histogram(vals)
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
@time_tracker.record_time_used
def make_statistics(collection) -> list[dict, dict]:
    STATS_TYPE = settings.stats.stats_type
    SAMPLING_RATE = settings.stats.sampling_rate
    
    MIN_FREQ_THRESHOLD = settings.stats.prune_params.min_freq_threshold
    MAX_NUM_PATHS = settings.stats.prune_params.max_no_paths_threshold
    MAX_PREFIX_LENGTH = settings.stats.prune_params.max_prefix_length_threshold
    # Max Prefix length, Max postfix length (prune middle keys)
        # Look at what JSON PATH in MySQL allows. Like wildcards

    base_stats = _make_base_statistics(collection)

    min_count_threshold = int(MIN_FREQ_THRESHOLD * len(collection)) * (PruneStrat.MIN_FREQ in settings.stats.prune_strats)

    pruned_path_stats = {}

    if PruneStrat.MIN_FREQ in settings.stats.prune_strats:
        log("Performing min_freq pruning...")
        for key_path, path_stats in base_stats.items():
            if path_stats.count > min_count_threshold:
                pruned_path_stats[key_path] = path_stats
    else:
        pruned_path_stats = base_stats

    max_count_excluded = None
    if PruneStrat.MAX_NO_PATHS in settings.stats.prune_strats:
        if len(pruned_path_stats) > MAX_NUM_PATHS:
            log("Performing max_no_paths pruning...")
            sorted_by_count = list(sorted(pruned_path_stats.items(), key=lambda t: t[1].count, reverse=True))
            pruned_path_stats = dict(sorted_by_count[:MAX_NUM_PATHS])
            max_count_excluded = sorted_by_count[MAX_NUM_PATHS+1][1].count

    if PruneStrat.UNIQUE_SUFFIX in settings.stats.prune_strats:
        approach = 2
        
        key_sep = settings.stats.key_path_key_sep
        type_sep = settings.stats.key_path_type_sep

        # Sort key paths, so that similar key_paths (e.g., array indices) come straight after one another
        # This will also make base paths appear before their typed counterparts
        all_key_paths = list(sorted(k for k in base_stats.keys()))


        # Approach 1:
        # Very simple. Check if the final typed key already exists in the set. 
        # If not, replace the key-value pair [tk_1.tk_2...tk_n => <stats>] with [tk_n => <stats>]
        # If it already exists, try tk_n-1.tk_n, and so on

        # Problem: On lookup, if multiple key-paths have the same ending, we don't know which
        # of those we are retrieving stats for. Leading occasional massively wrong estimates
        if approach == 1:
            for key_path in all_key_paths:

                # Only do typed key-paths, so we can retrieve the corresponding base path as well
                # is_typed = any(key_path.endswith(type_sep + type_str) for type_str in type_suffixes.values() if type_str)
                # if not is_typed:
                #     continue

                reduced_path = ""
                for key in key_path.split(key_sep)[1::-1]:
                    reduced_path = f"{key}.{reduced_path}" if reduced_path else key
                    
                    if reduced_path not in base_stats:
                        # Test that we can move the base path as well
                        # ...
                        
                        # We've found a unique, shortened version of the key-path
                        base_stats[reduced_path] = base_stats[key_path]
                        del base_stats[key_path]
                        
                        break

        # Approach 2:
        # Check that no other key_path ends with the typed key. Only then do we shorten.
        # As with the prev approach, we iteratively increase the key size going backwards.
        # With this approach, we can lookup by starting with the whole key-path and reducing iteratively until we find a match
        elif approach == 2:
            for key_path in all_key_paths:
                reduced_path = ""
                for key in key_path.split(key_sep)[::-1][:-1]:
                    reduced_path = f"{key}.{reduced_path}" if reduced_path else key
                    
                    collisions = [
                        cmp_key_path
                        for cmp_key_path in all_key_paths
                        if cmp_key_path != key_path and cmp_key_path.endswith(reduced_path)
                    ]

                    if not any(collisions):
                        base_stats[reduced_path] = base_stats[key_path]
                        del base_stats[key_path]
                        break

        
        else:
            assert False



    
    num_pruned = len(base_stats) - len(pruned_path_stats)
    log("num_pruned", num_pruned, f"({len(base_stats)} unique paths total)")

    summary_stats = {
        "highest_count_skipped": min_count_threshold if max_count_excluded is None else max_count_excluded,
        "collection_size": len(collection),
        "stats_type": STATS_TYPE,
        "sampling_rate": SAMPLING_RATE,
    } | ({
        "max_prefix_length": MAX_PREFIX_LENGTH
    } if PruneStrat.MAX_PREFIX_LENGTH in settings.stats.prune_strats else {})

    return [pruned_path_stats, summary_stats]


# Maybe there should be one of these for each approach
def get_stats_data(path, accessor):
    with open(path) as f:
        path_stats, summary_stats = json.load(f)

    try:
        return accessor(path_stats)
    except:
        return {"count": summary_stats["min_count_threshold"]}

@time_tracker.record_time_used
def get_statistics():
    if settings.stats.force_new or not stats_cache.check_cached_stats_exists():
        log("Creating fresh statistics...")
        collection = data_cache.load_data()
        stats = make_statistics(collection)
        stats_cache.add_stats(stats)
        
        # Write to stats to a human-readable format for inspection
        with open(settings.stats.out_path, mode="w") as f:
            json.dump(stats, f, cls=KeyStatEncoder)

    else:
        log("Retrieving cached statistics...")
        stats = stats_cache.get_cached_stats()
    
    return stats

def get_statistics_old(STATS_TYPE):
    data_path = settings.stats.data_path
    out_path = settings.stats.out_path

    def create_and_store_stats():
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
                
                # Convert stats back into KeyStat objects
                for k in stats[0]:
                    stats[0][k] = KeyStat(**stats[0][k])
                
                # Convert from enum name to enum object
                existing_stats[1]["stats_type"] = StatType[existing_stats[1]["stats_type"]]

            else:
                stats = create_and_store_stats()
    else:
        stats = create_and_store_stats()

    return stats

def run():
    log(f"using {StatType(STATS_TYPE).name} StatType")
    
    stats = get_statistics_old()


    # "entities_object.hashtags_array.0_object.indices_array.0_int"
    # stat_path = "entities_object.hashtags_array.0_object.indices_array.0_int"
    # key_path = ["entities", "hashtags", 0, "indices", 0]
    # compare_lt_estimate(data_path, key_path, stats[0], stat_path, 70)
    # compare_eq_estimate(data_path, key_path, stats[0], stat_path, 100)
    # compare_range_estimate(data_path, key_path, stats[0], stat_path, range(-1, 70))

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

if __name__ == '__main__':
    # TODO: TEST Stat creation

    for st in StatType:
        stats_1 = _make_base_statistics([
            {"a": 1},
            {"a": 2},
            {"a": None},

            {"b": {"c": 8}},
            {"b": {"c": -1}},
            {"b": {"c": -5}},
            {"b": {"c": -1}},
            {"b": {"c": 99}},
            {"b": {"c": -99}},
            {"b": {"c": -99}},
            {"b": {"c": 8}},
            
            # Should trigger min_val bug that hyperlog_ndv had (using 'or' instead of 'if is not null else')
            {"d": {"e": 1}},
            {"d": {"e": 0}},
            {"d": {"e": 1}},
        ], _STATS_TYPE=st, _SAMPLING_RATE=0)

        assert stats_1["a"].count == 3, (st, stats_1)
        assert stats_1["a"].null_count == 1, (st, stats_1)
        assert stats_1["a"].valid_count == 2, (st, stats_1)

        assert stats_1["a_number"].count == 2, (st, stats_1)
        assert stats_1["a_number"].null_count == 0, (st, stats_1)
        assert stats_1["a_number"].valid_count == 2, (st, stats_1)
        assert stats_1["a_number"].max_val == 2, (st, stats_1)
        assert stats_1["a_number"].min_val == 1, (st, stats_1)
        if st != StatType.BASIC:
            assert stats_1["a_number"].ndv == 2, (st, stats_1)

        assert stats_1["b_object.c_number"].count == 8, (st, stats_1)
        assert stats_1["b_object.c_number"].null_count == 0, (st, stats_1)
        assert stats_1["b_object.c_number"].valid_count == 8, (st, stats_1)
        assert stats_1["b_object.c_number"].max_val == 99, (st, stats_1)
        assert stats_1["b_object.c_number"].min_val == -99, (st, stats_1)
        
        assert stats_1["d_object.e_number"].count == 3, (st, stats_1)
        assert stats_1["d_object.e_number"].min_val == 0, (st, stats_1)
        assert stats_1["d_object.e_number"].max_val == 1, (st, stats_1)
        if st != StatType.BASIC:
            assert stats_1["d_object.e_number"].ndv == 2, (st, stats_1)

    
    test_hist = compute_histogram(
         ([0]*20)\
        +([1]*10)\
        +([2]*4)\
        +([5]*15)\

        +([7]*30)\
        +([8]*1)\
        +([9]*10)\

        +([10]*10)\
        +([15]*20),
        _nbins=3
    )

    assert test_hist == [
        (5, 49, 4),
        (9, 41, 3),
        (15, 30, 2)
    ]

    bool_hist_1 = compute_histogram(([False]*5) + ([True]*40), _nbins=2)
    bool_hist_2 = compute_histogram(([True]*40) + ([False]*5))
    assert bool_hist_1 == [(False, 5, 1), (True, 40, 1)] == bool_hist_2, (bool_hist_1, bool_hist_2)

    str_hist = compute_histogram(list("aaabaaadaaabaaadx"), _nbins=len("abdx"))
    assert str_hist == [("a", 12, 1), ("b", 2, 1), ("d", 2, 1), ("x", 1, 1)], str_hist


    collection = [
        {
            "a": True,
            "b": 1
        },
        {
            "a": False,
            "b": 1
        },
        {
            "a": None,
            "b": 1
        },
        {
            "a": None,
            "b": 1
        },
        {
            "b": 1
        },
        {
            "a": False,
            "b": 1
        },
        {
            "a": True,
            "b": 1
        },
    ]

    # print(_make_base_statistics(collection, StatType.HISTOGRAM))