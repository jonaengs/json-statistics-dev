import math
from compute_stats import get_statistics

from compute_structures import HistBucket, KeyStat, PruneStrat, StatType

from settings import settings
from trackers import time_tracker

"""
Contains functions for estimating cardinalities based on gathered statistics.
Before using, compute_and_set_stats (or _update_stats_info) must be called
so that statistics are collected and can be used by the functions. 
"""

STATS_TYPE = settings.stats.stats_type
stats = None
meta_stats = None

# TODO: CHECK VALUES
EQ_MULTIPLIER = 0.1
EQ_BOOL_MULTIPLIER = 0.5
INEQ_MULTIPLIER = 0.3  # (gt, lt, gte, lte)
RANGE_MULTIPLIER = 0.3
IS_NULL_MULTIPLIER = 0.1


def load_and_apply_stats():
    stats, meta_stats = get_statistics()
    _update_stats_info(settings.stats.stats_type, stats, meta_stats)
    return stats, meta_stats

def _update_stats_info(_STATS_TYPE=None, _stats=None, _meta_stats=None):
    """Set STATS_TYPE, stats and meta_stats manually. Should only be done for testing."""
    global STATS_TYPE, stats, meta_stats
    STATS_TYPE = _STATS_TYPE or STATS_TYPE
    stats = _stats or stats
    meta_stats = _meta_stats or meta_stats

    assert 1 >= meta_stats["sampling_rate"] >= 0
    assert not (PruneStrat.MAX_PREFIX_LENGTH in settings.stats.prune_strats and PruneStrat.UNIQUE_SUFFIX in settings.stats.prune_strats)

def _pre_post_process(f):
    def prune_stat_path(stat_path):
        max_len = settings.stats.prune_params.max_prefix_length_threshold
        sep = settings.stats.key_path_key_sep
        separated = stat_path.split(sep)
        if len(separated) > max_len:
            stat_path = sep.join(separated[-max_len:])

        return stat_path

    def find_unique_reduced_key_path(stat_path):
        key_sep = settings.stats.key_path_key_sep
        kp_parts = stat_path.split(key_sep)[::-1]

        reduced_kp = ""
        for kp_part in kp_parts:
            reduced_kp = kp_part + ((key_sep + reduced_kp) if reduced_kp else "")
            if reduced_kp in stats:
                return reduced_kp

    def wrapper(*args, **kwargs):
        if PruneStrat.MAX_PREFIX_LENGTH in settings.stats.prune_strats:
            # Prune the stat path to be of the maximum allowed length
            stat_path = args[0]
            stat_path = prune_stat_path(stat_path)
            args = (stat_path, ) + args[1:]

        if PruneStrat.UNIQUE_SUFFIX in settings.stats.prune_strats:
            stat_path = args[0]
            stat_path = find_unique_reduced_key_path(stat_path)
            args = (stat_path, ) + args[1:]

        estimate = f(*args, **kwargs)

        # Adjust estimate result for sampling
        adjusted = estimate / (1 - meta_stats["sampling_rate"])
        # And round the result in a consistent way
        rounded = math.ceil(adjusted)
        return rounded

    return wrapper



@_pre_post_process
@time_tracker.record_time_used
def estimate_exists_cardinality(stat_path):
    assert meta_stats["stats_type"] == STATS_TYPE, f"{meta_stats['stats_type']=}, {STATS_TYPE=}"

    if stat_path not in stats:
        est_card = meta_stats["highest_count_skipped"]
        return est_card

    return stats[stat_path].count

@_pre_post_process
@time_tracker.record_time_used
def estimate_not_null_cardinality(stat_path):
    assert meta_stats["stats_type"] == STATS_TYPE, f"{meta_stats['stats_type']=}, {STATS_TYPE=}"

    if stat_path not in stats:
        est_card = meta_stats["highest_count_skipped"]
        return est_card * (1 - IS_NULL_MULTIPLIER)

    return stats[stat_path].valid_count

@_pre_post_process
@time_tracker.record_time_used
def estimate_is_null_cardinality(stat_path):
    assert meta_stats["stats_type"] == STATS_TYPE, f"{meta_stats['stats_type']=}, {STATS_TYPE=}"

    if stat_path not in stats:
        est_card = meta_stats["highest_count_skipped"]
        return est_card * IS_NULL_MULTIPLIER

    return stats[stat_path].null_count


@_pre_post_process
@time_tracker.record_time_used
def estimate_gt_cardinality(stat_path: str, compare_value: float|int):
    assert meta_stats["stats_type"] == STATS_TYPE, f"{meta_stats['stats_type']=}, {STATS_TYPE=}"

    # If statistics are missing for this key-path, stop early
    if stat_path not in stats:
        est_card = meta_stats["highest_count_skipped"]
        return est_card * INEQ_MULTIPLIER 

    data = stats[stat_path]
    # max_val not gathered for str and bool types
    if compare_value >= data.max_val or data.max_val == data.min_val:
        return 0

    match STATS_TYPE:
        case StatType.BASIC | StatType.BASIC_NDV | StatType.HYPERLOG:
            data_range = data.max_val - data.min_val
            compare_range = min(data.max_val - compare_value, data_range)
            overlap = compare_range / data_range

            return overlap * data.valid_count
        case StatType.HISTOGRAM:
            estimate = 0
            bucket_lower_bound = data.min_val
            for bucket in map(lambda b: HistBucket(*b), data.histogram):
                if bucket_lower_bound > compare_value:
                    estimate += bucket.count
                elif bucket.upper_bound > compare_value:
                    if bucket.ndv == 1:
                        # Bucket contains only a single value: the upper bound. 
                        # Because the upper_bound >= compare_val, the entire bucket
                        # satisfies the gt predicate
                        estimate += bucket.count
                    else:
                        bucket_range = bucket.upper_bound - bucket_lower_bound
                        valid_value_range = bucket.upper_bound - compare_value
                        overlap = valid_value_range / bucket_range
                        estimate += bucket.count * overlap

                bucket_lower_bound = bucket.upper_bound

            return estimate

@_pre_post_process
@time_tracker.record_time_used
def estimate_lt_cardinality(stat_path: str, lt_value: float|int):
    assert meta_stats["stats_type"] == STATS_TYPE, f"{meta_stats['stats_type']=}, {STATS_TYPE=}"

    if stat_path not in stats:
        est_card = meta_stats["highest_count_skipped"]
        return est_card * INEQ_MULTIPLIER 

    data = stats[stat_path]
    if lt_value <= data.min_val or data.max_val == data.min_val:
        return 0

    match STATS_TYPE:
        case StatType.BASIC | StatType.BASIC_NDV | StatType.HYPERLOG:
            data_range = data.max_val - data.min_val
            compare_range = min(lt_value - data.min_val, data_range)
            overlap = compare_range / data_range

            assert 1 >= overlap >= 0

            return overlap * data.valid_count
        case StatType.HISTOGRAM:
            # NOTE: Assumed bucket structure [upper_bound, val_count, ndv]

            estimate = 0
            bucket_lower_bound = data.min_val
            for bucket in map(lambda b: HistBucket(*b), data.histogram):
                if bucket.upper_bound < lt_value:
                    estimate += bucket.count
                elif bucket.upper_bound >= lt_value:
                    if bucket.ndv == 1:  
                        # Bucket contains only a single value: the upper bound. 
                        # Because the upper_bound >= compare_val, the bucket doesn't satisfy the predicate
                        # and should not be counted
                        continue
                    else:
                        bucket_range = bucket.upper_bound - bucket_lower_bound
                        valid_value_range = lt_value - bucket_lower_bound
                        overlap = valid_value_range / bucket_range
                        estimate += bucket.count * overlap

                    # No more buckets will satisfy lt pred, so stop here
                    break

                bucket_lower_bound = bucket.upper_bound


            return estimate


# NOTE: Only works for floats and ints
# For floats, a non-inclusive upper range makes little sense. So this is inclusive at both ends.
@_pre_post_process
@time_tracker.record_time_used
def estimate_range_cardinality(stat_path: str, q_range: range):
    if stat_path not in stats:
        est_card = meta_stats["highest_count_skipped"]
        return est_card * INEQ_MULTIPLIER 

    data: KeyStat = stats[stat_path]
    match STATS_TYPE:
        case StatType.BASIC | StatType.BASIC_NDV | StatType.HYPERLOG:
            overlapping_range = min(data.max_val, q_range.stop) - max(data.min_val, q_range.start)

            if overlapping_range < 0:
                return 0

            data_range = data.max_val - data.min_val
            overlap = overlapping_range / data_range

            return overlap * data.valid_count

        case StatType.HISTOGRAM:
            estimate = 0
            bucket_lower_bound = data.min_val
            for bucket in map(lambda b: HistBucket(*b), data.histogram):
                overlapping_range = min(bucket.upper_bound, q_range.stop) - max(bucket_lower_bound, q_range.start)
                overlapping_range = max(overlapping_range, 0)
                bucket_range = bucket.upper_bound - bucket_lower_bound
                overlap = overlapping_range / bucket_range
                
                estimate += overlap * bucket.count

                bucket_lower_bound = bucket.upper_bound
                

            return estimate
    
@_pre_post_process
@time_tracker.record_time_used
def estimate_eq_cardinality(stat_path: str, compare_value):
    if stat_path not in stats:
        est_card = meta_stats["highest_count_skipped"]
        return est_card * (EQ_MULTIPLIER if type(compare_value) != bool else EQ_BOOL_MULTIPLIER)

    data = stats[stat_path]
    # We currently don't track min and max for strings and bools. So we can't check if we're outside the range 
    # for those values
    if type(compare_value) not in (str, bool) and (compare_value > data.max_val or compare_value < data.min_val):
        return 0

    match STATS_TYPE:
        case StatType.BASIC:
            # TODO: On EQ float, should we just return 0? 
            return stats[stat_path].valid_count * (EQ_MULTIPLIER if type(compare_value) != bool else EQ_BOOL_MULTIPLIER)
        case StatType.BASIC_NDV | StatType.HYPERLOG:
            return data.valid_count/(data.ndv if type(compare_value) != bool else 2)  # No ndv data for bools

        case StatType.HISTOGRAM:
            # Check for string singleton histograms
            if type(compare_value) == str:
                if data.histogram:
                    return next((b.count for b in data.histogram if b.upper_bound == compare_value), 0)
                else:
                    # No histogram data collected. Fall back to ndv
                    return data.valid_count/data.ndv

            for bucket in map(lambda b: HistBucket(*b), data.histogram):
                if bucket.upper_bound >= compare_value:
                    # Assume uniform distribution, so return count divided by ndv
                    return bucket.count / bucket.ndv

            # If we went past all buckets and never found anything with our value, estimate is 0
            return 0



"""
Example usage:
    data = json.load(f)
    compute_stats(data, StatType.HISTOGRAM)
    estimate_eq_cardinality("key_obj.0_num", 5)
    ...
"""