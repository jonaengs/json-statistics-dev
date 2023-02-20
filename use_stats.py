import math
from compute_stats import get_statistics, make_enum_array_stat_path, type_suffixes

from compute_structures import EquiHeightBucket, HistogramType, Json_Primitive, KeyStat, PruneStrat, StatType

from settings import settings
from trackers import time_tracker
from logger import log

"""
Contains functions for estimating cardinalities based on gathered statistics.
Before using, compute_and_set_stats (or _update_stats_info) must be called
so that statistics are collected and can be used by the functions. 
"""

STATS_TYPE = settings.stats.stats_type
stats: (dict[str, KeyStat] | None) = None
meta_stats = None

# TODO: CHECK VALUES
EQ_MULTIPLIER = 0.1
EQ_BOOL_MULTIPLIER = 0.5
INEQ_MULTIPLIER = 0.3  # (gt, lt, gte, lte)
RANGE_MULTIPLIER = 0.3
IS_NULL_MULTIPLIER = 0.1

JSON_MEMBEROF_MULTIPLIER = 0.1
JSON_CONTAINS_MULTIPLIER = 0.1
JSON_OVERLAPS_MULTIPLIER = 0.3


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

def _find_unique_reduced_key_path(stat_path):
    key_sep = settings.stats.key_path_key_sep
    kp_parts = stat_path.split(key_sep)[::-1]

    approach = 2

    # Approach 1:
    # Build up key-path from shortest to longest
    if approach == 1:
        reduced_kp = ""
        for kp_part in kp_parts:
            reduced_kp = f"{kp_part}.{reduced_kp}" if reduced_kp else kp_part
            if reduced_kp in stats:
                return reduced_kp


    # Approach 2:
    # Reduce key-path from longest to shortest
    elif approach == 2:
        reduced_kp = stat_path
        while reduced_kp:
            if reduced_kp in stats:
                return reduced_kp
            
            next_sep_idx = reduced_kp.find(key_sep)
            if next_sep_idx == -1:
                break

            reduced_kp = reduced_kp[next_sep_idx+1:]

    
    # Return stat path if we can't find any reduced paths
    return stat_path


def _apply_common_pre_post_processing(f):
    def prune_stat_path(stat_path):
        max_len = settings.stats.prune_params.max_prefix_length_threshold
        sep = settings.stats.key_path_key_sep
        separated = stat_path.split(sep)
        if len(separated) > max_len:
            stat_path = sep.join(separated[-max_len:])

        return stat_path

    def wrapper(*args, **kwargs):
        if PruneStrat.MAX_PREFIX_LENGTH in settings.stats.prune_strats:
            # Prune the stat path to be of the maximum allowed length
            stat_path = args[0]
            stat_path = prune_stat_path(stat_path)
            args = (stat_path, ) + args[1:]

        if PruneStrat.UNIQUE_SUFFIX in settings.stats.prune_strats:
            stat_path = args[0]
            stat_path = _find_unique_reduced_key_path(stat_path)
            args = (stat_path, ) + args[1:]

        estimate = f(*args, **kwargs)

        if estimate < 0:
            log()
            log("Estimate below zero:")
            log(f"{estimate=}")
            log(f.__name__, args, kwargs)

        # Adjust estimate result for sampling
        adjusted = estimate / (1 - meta_stats["sampling_rate"])
        # And round the result in a consistent way
        rounded = math.ceil(adjusted)
        return rounded

    return wrapper



@_apply_common_pre_post_processing
@time_tracker.record_time_used
def estimate_exists_cardinality(stat_path):
    assert meta_stats["stats_type"] == STATS_TYPE, f"{meta_stats['stats_type']=}, {STATS_TYPE=}"

    if stat_path not in stats:
        est_card = meta_stats["highest_count_skipped"]
        return est_card

    return stats[stat_path].count

@_apply_common_pre_post_processing
@time_tracker.record_time_used
def estimate_not_null_cardinality(stat_path):
    assert meta_stats["stats_type"] == STATS_TYPE, f"{meta_stats['stats_type']=}, {STATS_TYPE=}"

    if stat_path not in stats:
        est_card = meta_stats["highest_count_skipped"]
        return est_card * (1 - IS_NULL_MULTIPLIER)

    return stats[stat_path].valid_count

@_apply_common_pre_post_processing
@time_tracker.record_time_used
def estimate_is_null_cardinality(stat_path):
    assert meta_stats["stats_type"] == STATS_TYPE, f"{meta_stats['stats_type']=}, {STATS_TYPE=}"

    if stat_path not in stats:
        est_card = meta_stats["highest_count_skipped"]
        return est_card * IS_NULL_MULTIPLIER

    return stats[stat_path].null_count


@_apply_common_pre_post_processing
@time_tracker.record_time_used
def estimate_gt_cardinality(stat_path: str, gt_value: float|int):
    assert meta_stats["stats_type"] == STATS_TYPE, f"{meta_stats['stats_type']=}, {STATS_TYPE=}"

    # If statistics are missing for this key-path, stop early
    if stat_path not in stats:
        est_card = meta_stats["highest_count_skipped"]
        return est_card * INEQ_MULTIPLIER 

    data: KeyStat = stats[stat_path]
    # max_val not gathered for str and bool types
    if gt_value >= data.max_val or data.max_val == data.min_val:
        return 0

    match STATS_TYPE:
        case StatType.BASIC | StatType.BASIC_NDV | StatType.NDV_HYPERLOG:
            data_range = data.max_val - data.min_val
            compare_range = min(data.max_val - gt_value, data_range)
            overlap = compare_range / data_range

            return overlap * data.valid_count
        case StatType.NDV_WITH_MODE:
            # same as BASIC | BASIC_NDV | HYPERLOG, except we add/subtract
            # based on whether the mode value is included. 

            mode_modifier = 0
            if data.mode_info:
                mode, mode_count = data.mode_info
                expected_count = data.valid_count/data.ndv

                # If mode is included, increase result by
                # how much more the mode occurs than the average value.
                # If it's not included, decrease by that value.
                if mode > gt_value:
                    mode_modifier = mode_count - expected_count
                else:
                    mode_modifier = expected_count - mode_count

            data_range = data.max_val - data.min_val
            compare_range = min(data.max_val - gt_value, data_range)
            overlap = compare_range / data_range
            # Say that overlap must be at least as much as if we did x=max_val
            # overlap = max(overlap, (data.valid_count/data.ndv) * (gt_value < data.max_val))

            estimate = (overlap * data.valid_count) + mode_modifier
            return max(estimate, gt_value < data.max_val)  # Return at least 1 if compare value is valid
        case StatType.HISTOGRAM:
            match data.histogram.type:
                case HistogramType.EQUI_HEIGHT:
                    estimate = 0
                    bucket_lower_bound = data.min_val
                    for bucket in map(lambda b: EquiHeightBucket(*b), data.histogram.buckets):
                        if bucket_lower_bound > gt_value:
                            estimate += bucket.count
                        elif bucket.upper_bound > gt_value:
                            if bucket.ndv == 1:
                                estimate += bucket.count  # Upper_bound must be single value in bucket
                            else:
                                bucket_range = bucket.upper_bound - bucket_lower_bound
                                valid_value_range = bucket.upper_bound - gt_value
                                overlap = valid_value_range / bucket_range
                                estimate += bucket.count * overlap

                        bucket_lower_bound = bucket.upper_bound
                case HistogramType.SINGLETON:
                    estimate = 0
                    for bucket in data.histogram.buckets:
                        if bucket.value > gt_value:
                            estimate += bucket.count

                case _:
                    assert False, f"MISSING CASE FOR: {data.histogram.type}"

            return estimate

        case _:
            assert False, f"MISSING CASE FOR: {STATS_TYPE}"

@_apply_common_pre_post_processing
@time_tracker.record_time_used
def estimate_lt_cardinality(stat_path: str, lt_value: float|int):
    assert meta_stats["stats_type"] == STATS_TYPE, f"{meta_stats['stats_type']=}, {STATS_TYPE=}"

    if stat_path not in stats:
        est_card = meta_stats["highest_count_skipped"]
        return est_card * INEQ_MULTIPLIER 

    data: KeyStat = stats[stat_path]
    if lt_value <= data.min_val or data.max_val == data.min_val:
        return 0

    match STATS_TYPE:
        case StatType.BASIC | StatType.BASIC_NDV | StatType.NDV_HYPERLOG:
            data_range = data.max_val - data.min_val
            compare_range = min(lt_value - data.min_val, data_range)
            overlap = compare_range / data_range

            return overlap * data.valid_count
        case StatType.NDV_WITH_MODE:
            # same as BASIC | BASIC_NDV | HYPERLOG, except we add/subtract
            # based on whether the mode value is included. 

            mode_modifier = 0
            if data.mode_info:
                mode, mode_count = data.mode_info
                expected_count = data.valid_count/data.ndv

                # If mode is included, increase result by
                # how much more the mode occurs than the average value.
                # If it's not included, decrease by that value.
                if mode < lt_value:
                    mode_modifier = mode_count - expected_count
                else:
                    mode_modifier = expected_count - mode_count

            data_range = data.max_val - data.min_val
            compare_range = min(lt_value - data.min_val, data_range)
            overlap = compare_range / data_range
            # Say that overlap must be at least as much as if we did x=min_val
            # overlap = max(overlap, (data.valid_count/data.ndv) * (lt_value > data.min_val))
            estimate = (overlap * data.valid_count) + mode_modifier

            return max(estimate, 1)
        case StatType.HISTOGRAM:
            # NOTE: Assumed bucket structure [upper_bound, val_count, ndv]

            match data.histogram.type:
                case HistogramType.EQUI_HEIGHT:
                    estimate = 0
                    bucket_lower_bound = data.min_val
                    for bucket in data.histogram.buckets:
                        if bucket.upper_bound < lt_value:
                            estimate += bucket.count
                        elif bucket.upper_bound >= lt_value:
                            # Bucket contains only a single value: the upper bound. 
                            # Because the upper_bound >= compare_val, the bucket doesn't satisfy the predicate
                            # and should not be counted
                            if bucket.ndv > 1:  
                                bucket_range = bucket.upper_bound - bucket_lower_bound
                                valid_value_range = lt_value - bucket_lower_bound
                                overlap = valid_value_range / bucket_range
                                estimate += bucket.count * overlap

                            # No more buckets will satisfy lt pred, so stop here
                            break

                        bucket_lower_bound = bucket.upper_bound
                case HistogramType.SINGLETON:
                    estimate = 0
                    for bucket in data.histogram.buckets:
                        if bucket.value < lt_value:
                            estimate += bucket.count
                        else:
                            break

                case _:
                    assert False, f"MISSING CASE FOR: {data.histogram.type}"

            return estimate
        
        case _:
            assert False, f"MISSING CASE FOR: {STATS_TYPE}"


# NOTE: Only works for floats and ints
# For floats, a non-inclusive upper range makes little sense. So this is inclusive at both ends.
@_apply_common_pre_post_processing
@time_tracker.record_time_used
def estimate_range_cardinality(stat_path: str, q_range: range):
    if stat_path not in stats:
        est_card = meta_stats["highest_count_skipped"]
        return est_card * INEQ_MULTIPLIER 

    data: KeyStat = stats[stat_path]
    match STATS_TYPE:
        case StatType.BASIC | StatType.BASIC_NDV | StatType.NDV_HYPERLOG:
            overlapping_range = min(data.max_val, q_range.stop) - max(data.min_val, q_range.start)

            if overlapping_range < 0:
                return 0

            data_range = data.max_val - data.min_val
            overlap = overlapping_range / data_range

            return overlap * data.valid_count

        case StatType.HISTOGRAM:
            assert data.histogram.type == HistogramType.EQUI_HEIGHT

            estimate = 0
            bucket_lower_bound = data.min_val
            for bucket in data.histogram.buckets:
                overlapping_range = min(bucket.upper_bound, q_range.stop) - max(bucket_lower_bound, q_range.start)
                overlapping_range = max(overlapping_range, 0)
                bucket_range = bucket.upper_bound - bucket_lower_bound
                overlap = overlapping_range / bucket_range
                
                estimate += overlap * bucket.count

                bucket_lower_bound = bucket.upper_bound
                

            return estimate

        case _:
            assert False, f"MISSING CASE FOR: {STATS_TYPE}"
    
@_apply_common_pre_post_processing
@time_tracker.record_time_used
def estimate_eq_cardinality(stat_path: str, compare_value):
    if stat_path not in stats:
        est_card = meta_stats["highest_count_skipped"]
        return est_card * (EQ_MULTIPLIER if type(compare_value) != bool else EQ_BOOL_MULTIPLIER)

    data: KeyStat = stats[stat_path]
    # We currently don't track min and max for strings and bools. So we can't check if we're outside the range 
    # for those values
    if type(compare_value) not in (str, bool) and (compare_value > data.max_val or compare_value < data.min_val):
        return 0

    match STATS_TYPE:
        case StatType.BASIC:
            # On EQ float, should we just return 0? 
            return stats[stat_path].valid_count * (EQ_MULTIPLIER if type(compare_value) != bool else EQ_BOOL_MULTIPLIER)
        case StatType.BASIC_NDV | StatType.NDV_HYPERLOG:
            return data.valid_count/data.ndv
            # return data.valid_count/(data.ndv if type(compare_value) != bool else 2)  # No ndv data for bools
        case StatType.NDV_WITH_MODE:
            if not data.mode_info:
                # Fallback in case mode_info not stored for some reason
                return data.valid_count/data.ndv

            mode, mode_count = data.mode_info
            if compare_value == mode:
                return mode_count
            # Avoid division by zero error below.
            if data.ndv == 1:
                # If we've only seen the mode, and the query value is not the mode, then
                # we're safe to assume a cardinality of 0
                return 0

            # If not eq mode. Divide by ndv-1 as we can exclude one possible value
            estimate = (data.valid_count - mode_count) / (data.ndv - 1)
            return estimate
        case StatType.HISTOGRAM:
            if not data.histogram:
                # No histogram data collected. Fall back to ndv
                return data.valid_count/data.ndv

            match data.histogram.type:
                case HistogramType.SINGLETON:
                    return next((b.count for b in data.histogram.buckets if b.value == compare_value), 0)

                case HistogramType.EQUI_HEIGHT:
                    for bucket in data.histogram.buckets:
                        if bucket.upper_bound >= compare_value:
                            # Assume uniform distribution, so return count divided by ndv
                            return bucket.count / bucket.ndv

                    # If we went past all buckets and never found anything with our value, estimate is 0
                    return 0

                case HistogramType.SINGLETON_PLUS:
                    normal_buckets = data.histogram.buckets[:-1]
                    special_bucket: EquiHeightBucket = data.histogram.buckets[-1]
                    fallback = special_bucket.count / special_bucket.ndv
                    return next((b.count for b in normal_buckets if b.value == compare_value), fallback)
                
                case _:
                    assert False, f"MISSING CASE FOR: {data.histogram.type}"

        case _:
            assert False, f"MISSING CASE FOR: {STATS_TYPE}"

def _get_enum_arr_stats(stat_path, arr_type: type):
    type_sep = settings.stats.key_path_type_sep
    type_suffix = type_suffixes[arr_type]
    arr_info_stat_path = make_enum_array_stat_path(stat_path, arr_type)

    if arr_info_stat_path in stats:
        return stats[arr_info_stat_path]

    # We can also estimate from the ndv of each idx, but 
    # the estimation uncertainty would be massive:
    # between max(ndvs) and sum(ndvs)

@time_tracker.record_time_used
def _get_memberof_cardinality_estimate(stat_path, lookup_val):

    # Only accepts primitive values
    assert isinstance(lookup_val, Json_Primitive)

    arr_info_stat_path = make_enum_array_stat_path(stat_path, type(lookup_val))

    if arr_info_stat_path in stats:
        histogram = stats[arr_info_stat_path].histogram

        assert histogram.type in (HistogramType.SINGLETON, HistogramType.SINGLETON_PLUS), histogram.type  
        
        hist_buckets = histogram.buckets if histogram.type == HistogramType.SINGLETON else histogram.buckets[:-1]
        for bucket in hist_buckets:
            if bucket.value == lookup_val:
                return bucket.count
        else:
            if histogram.type == HistogramType.SINGLETON_PLUS:
                plus_bucket: EquiHeightBucket = hist_buckets[-1]
                # TODO: This may be a bad estimate. Look into fixing
                # return plus_bucket.count / plus_bucket.ndv
                return plus_bucket.count
    
    
    # return stats[stat_path].count * JSON_MEMBEROF_MULTIPLIER


    type_sep = settings.stats.key_path_type_sep
    type_suffix = type_suffixes[type(lookup_val)]
    # Without special structures (enum array histograms), we'll have to do a lookup on every array index
    # This is obviously very slow
    estimate, arr_idx = 0, 0
    key_path = stat_path + f".{arr_idx}{type_sep}{type_suffix}"
    while key_path in stats:
        histogram = stats[key_path].histogram
        
        # TODO: Look at handling equi-height histograms in a better way
        # Equi-height histograms can occur here for integer enum arrays
        if histogram.type == HistogramType.EQUI_HEIGHT:
            base_cardinality = stats[stat_path].valid_count if stat_path in stats else meta_stats["highest_count_skipped"]
            return base_cardinality * JSON_MEMBEROF_MULTIPLIER
        
        hist_buckets = histogram.buckets if histogram.type == HistogramType.SINGLETON else histogram.buckets[:-1]
        for bucket in hist_buckets:
            if bucket.value == lookup_val:
                estimate += bucket.count
                break
        else:
            if histogram.type == HistogramType.SINGLETON_PLUS:
                plus_bucket: EquiHeightBucket = hist_buckets[-1]
                # TODO: This may be a bad estimate. Look into fixing
                # estimate += plus_bucket.count / plus_bucket.ndv
                estimate += plus_bucket.count

        arr_idx += 1
        key_path = stat_path + f".{arr_idx}{type_sep}{type_suffix}"


    return estimate

    # TODO: If we're doing min_freq or max_no_path pruning, anywhere between a few and all array paths may disappear
    # despite holding relevant data. Is there any way to remedy this?


@_apply_common_pre_post_processing
def estimate_memberof_cardinality(stat_path, lookup_val):
    """
    MySQL docs: https://dev.mysql.com/doc/refman/8.0/en/json-search-functions.html#operator_member-of
    """

    # TODO: Look at solutions for when we we don't have histograms. 
    # Can we still do some kind of automation?

    # We can only do precise estimation with histogram data
    # TODO: Look at whether we can do something more clever here. Using NDV maybe?
    if STATS_TYPE != StatType.HISTOGRAM or stat_path not in stats:
        base_cardinality = stats[stat_path].valid_count if stat_path in stats else meta_stats["highest_count_skipped"]
        return base_cardinality * JSON_MEMBEROF_MULTIPLIER


    return _get_memberof_cardinality_estimate(stat_path, lookup_val)


@_apply_common_pre_post_processing
def estimate_contains_cardinality(stat_path, lookup_arr: list):
    """
    JSON_CONTAINS returns all objects of which the query object is a strict subset.

    Example of how MySQL's JSON_CONTAINS works for arrays (slightly changed JSON_OVERLAPS example from https://dev.mysql.com/doc/refman/8.0/en/json-search-functions.html#function_json-overlaps)

    > SELECT JSON_CONTAINS("[1,3,5,7]", "[1]");
    > 1 row: [1,3,5,7]
    
    > SELECT JSON_CONTAINS("[1,3,5,7]", "1");
    > 1 row: [1,3,5,7]
    
    > SELECT JSON_CONTAINS("[1,3,5,7]", "2");
    > 0 rows

    > SELECT JSON_CONTAINS("[1,3,5,7]", "[1, 7]");
    > 1 row: [1,3,5,7] 

    > SELECT JSON_CONTAINS("[1,3,5,7]", "[1, 2]");
    > 0 rows 

    > SELECT JSON_CONTAINS("[1,3,5,7]", "[]");
    > 1 row: [1,3,5,7]
    """
    
    # MAIN LOGIC:
    # If lookup_arr is empty: return array key-path count
    # If lookup_arr has length 1: check stats of that single value
    # If lookup_arr length > 1: ???? Either multiply as if independent, or take count of least frequently occurring val

    # While JSON_OVERLAPS support looking up a single value, this function
    # will only support lists
    assert type(lookup_arr) == list


    # If lookup array empty, everything overlaps
    if not lookup_arr:
        return stats[stat_path].valid_count if stat_path in stats else meta_stats["highest_count_skipped"]

    # We can only do precise estimation with histogram data
    if STATS_TYPE != StatType.HISTOGRAM or stat_path not in stats:
        base_cardinality = stats[stat_path].valid_count if stat_path in stats else meta_stats["highest_count_skipped"]
        return base_cardinality * JSON_CONTAINS_MULTIPLIER

    estimates = {v: 0 for v in lookup_arr}  # Estimated count for each value in the lookup array
    for lookup_value in lookup_arr:
        estimates[lookup_value] = _get_memberof_cardinality_estimate(stat_path, lookup_value)
        
        
    if len(estimates) == 1:
        return list(estimates.values())[0]

    # TODO: FIgure out what to do with multiple lookup values
    upper_bound = min(estimates.values())

    enum_arr_stats = _get_enum_arr_stats(stat_path, type(lookup_arr[0]))
    if enum_arr_stats:
        count = enum_arr_stats.count
        lower_bound = math.prod(est / count for est in estimates.values()) * count

        # print("contains bounds:", ((math.ceil(lower_bound), upper_bound)))

    # return upper_bound

    return upper_bound / len(estimates)


@_apply_common_pre_post_processing
def estimate_overlaps_cardinality(stat_path, lookup_arr):
    """
    JSON_OVERLAPS returns all objects that overlap (share values) with the query object in any way.

    Example of how MySQL's JSON_CONTAINS works for arrays (slightly changed JSON_OVERLAPS example from https://dev.mysql.com/doc/refman/8.0/en/json-search-functions.html#function_json-overlaps)

    > SELECT JSON_OVERLAPS("[1,3,5,7]", "[1]");
    > 1 row: [1,3,5,7]
    
    > SELECT JSON_OVERLAPS("[1,3,5,7]", "1");
    > 1 row: [1,3,5,7]
    
    > SELECT JSON_OVERLAPS("[1,3,5,7]", "2");
    > 0 rows

    > SELECT JSON_OVERLAPS("[1,3,5,7]", "[1, 7]");
    > 1 row: [1,3,5,7] 

    > SELECT JSON_OVERLAPS("[1,3,5,7]", "[1, 2]");
    > 1 row: [1,3,5,7] 

    > SELECT JSON_OVERLAPS("[1,3,5,7]", "[]");
    > 0 rows
    """

    # MAIN LOGIC:
    # If lookup_arr is empty: return 0
    # If lookup_arr has length 1: check stats of that single value
    # If lookup_arr length > 1: Either add up count of each, or take count of most frequently occurring val

    # While JSON_OVERLAPS support looking up a single value, this function
    # will only support lists
    assert type(lookup_arr) == list

    if not lookup_arr:
        return 0

    if STATS_TYPE != StatType.HISTOGRAM or stat_path not in stats:
        base_cardinality = stats[stat_path].valid_count if stat_path in stats else meta_stats["highest_count_skipped"]
        return base_cardinality * JSON_OVERLAPS_MULTIPLIER

    estimates = [
        _get_memberof_cardinality_estimate(stat_path, lookup_val)
        for lookup_val in lookup_arr
    ]

    return max(estimates)
    

"""
Example usage:
    data = json.load(f)
    compute_stats(data, StatType.HISTOGRAM)
    estimate_eq_cardinality("key_obj.0_num", 5)
    ...
"""


if __name__ == '__main__':
    pre_pruned_stats = {
        'a': 1,
        'a_object': 1,
        'a_object.b': 1,
        'a_object.b_array': 1,
        'a_object.b_array.0': 1,
        'a_object.b_array.0_number': 1,

        'b': 1,
        'b_object': 1,
        'b_object.b': 1,
        'b_object.b_array': 1,
        'b_object.b_array.0': 1,
        'b_object.b_array.0_number': 1,

        'c': 1,
        'c_object': 1,
        'c_object.d': 1,
        'c_object.d_array': 1,
        'c_object.d_array.0': 1,
        'c_object.d_array.0_number': 1,
        
        'c_object.d_array.0_str': 1,
    }

    pruned_stats = {
        'a': 1,
        'a_object': 1,
        'a_object.b': 1,
        'a_object.b_array': 1,
        'a_object.b_array.0': 1,
        'a_object.b_array.0_number': 1,

        'b': 1,
        'b_object': 1,
        'b_object.b': 1,
        'b_object.b_array': 1,
        'b_object.b_array.0': 1,
        'b_object.b_array.0_number': 1,
        
        'c': 1,
        'c_object': 1,
        'd': 1,
        'd_array': 1,
        'd_array.0': 1,
        'd_array.0_number': 1,

        '0_str': 1,
    }

    stats = pruned_stats
    for pre_pruned_k, pruned_k in zip(pre_pruned_stats.keys(), pruned_stats.keys()):
        assert _find_unique_reduced_key_path(pre_pruned_k) == pruned_k, (_find_unique_reduced_key_path(pre_pruned_k), pruned_k)