from copy import copy, deepcopy
import itertools
import json
import math
import os
import pickle
import random
from collections import defaultdict, namedtuple
import sys
import time
from typing import Any, Callable
import typing

from munch import munchify
from tqdm import tqdm

from compute_structures import Json_Number, Json_Primitive, Json_Primitive_No_Null, KeyStatEncoder, PruneStrat, StatType
from compute_stats import _make_base_statistics, make_key_path
from settings import lock_settings, settings, unlock_settings
from use_stats import estimate_contains_cardinality, estimate_memberof_cardinality, estimate_overlaps_cardinality, load_and_apply_stats, estimate_eq_cardinality, estimate_exists_cardinality, estimate_gt_cardinality, estimate_is_null_cardinality, estimate_lt_cardinality, estimate_not_null_cardinality
from logger import log
import data_cache
from visualize import pause_for_visuals, plot_errors, scatterplot
from trackers import time_tracker, global_mem_tracker


# Used to create ranges of real numbers used to stand in for python's builtin integer ranges
r_range = namedtuple("Real_Range", ("start", "stop"))

def get_base_path(kp):
    # Removes the type suffix from the final typed key in a key path string
    key_sep = settings.stats.key_path_key_sep 
    type_sep = settings.stats.key_path_type_sep
    return tuple(s.rsplit(type_sep, 1)[0] for s in kp.split(key_sep))

def json_path_to_key_path(json_path, ex_val):
    path = ""
    for key, next_key in zip(json_path, json_path[1:]):
        path, _ = make_key_path(path, key, [] if isinstance(next_key, int) or next_key.isdigit() else {})

    return make_key_path(path, json_path[-1], ex_val)[0]

def json_path_to_base_key_path(json_path):
    path = ""
    for key, next_key in zip(json_path, json_path[1:]):
        path, _ = make_key_path(path, key, [] if isinstance(next_key, int) or next_key.isdigit() else {})

    return make_key_path(path, json_path[-1], 0)[1]

def get_possible_key_paths(json_path: tuple[str], primitives_only=True) -> list[str]:
    ex_vals = [int(), float(), str(), bool(), None] + ([list, dict] if not primitives_only else [])

    path = ""
    for key, next_key in zip(json_path, json_path[1:]):
        path, _ = make_key_path(path, key, [] if isinstance(next_key, int) or next_key.isdigit() else {})
    

    return list(set(make_key_path(path, json_path[-1], ex)[0] for ex in ex_vals))


@time_tracker.record_time_used
def get_operation_cardinality(collection: list[dict], json_path, operation: Callable[[Any], bool]):
    def traverse(doc, path: list[str]) -> Any:
        if not path:
            return doc
        return traverse(doc[path[0]], path[1:])

    count = 0
    for doc in collection:
        try:
            count += operation(traverse(doc, json_path))
        except:
            pass

    return count

@time_tracker.record_time_used
def get_operation_cardinality_2(collection: dict, json_path, operation: Callable[[Any], bool]):
    """
    Doesn't take a normal doc collection. Instead, takes mapping of json_path -> all values from that path.
    Should be more efficient, but may lead to issues if multiple types are present in the array 
    (like if the path ever leads to null). We should be able to remedy this by including a type 
    check in the operation function though 
    (so instead of 'lambda x: x == val', do 'lambda x: type(x) == type(val) and x == val').
    """
    return sum(map(operation, collection[json_path]))

def calc_error(truth, estimate):
    # Returns the (>1) factor by which the estimate is off from the truth, minus 1

    diff = abs(truth - estimate)
    
    error = diff / (truth or 1)
    # error = diff / (min(truth, estimate) or 1)
    
    return error

    if truth <= estimate:
        return estimate / truth
    return truth / estimate 


_prev_err_json_path = None
def err_if_high_err(predicate_name: str, threshold, estimate, truth, test_val, json_path, stats, meta_stats):
    global _prev_err_json_path
    error = calc_error(truth, estimate)

    # if error >= threshold:
    if error >= threshold and json_path != _prev_err_json_path:
        _prev_err_json_path = json_path

        log()
        log(f"{predicate_name.upper()} estimate error above threshold {threshold}")
        log(f"{error=:.3f}, {truth=}, {estimate=}")
        log("val:", test_val)

        computed_key_path = json_path_to_key_path(json_path, test_val)
        log(json_path, computed_key_path)
        if computed_key_path in stats:
            log(stats[computed_key_path])
        else:
            log("*computed key path not in stats*")
        log(meta_stats)
        log()

        # sys.exit(0)



def update_stats_settings(new_stats_settings):
    unlock_settings()

    for k, v in new_stats_settings.items():
        assert k in settings.stats, f"key {k=} not already present in settings"
        if isinstance(v, dict):
            for k2, v2 in v.items():
                assert k2 in v, f"key {k2=} not already present in child settings {v}"
                settings.stats[k][k2] = v2

        else:
            settings.stats[k] = v
    
    lock_settings()


@time_tracker.record_time_used
def run_analysis():
    collection = data_cache.load_data()

    # Sets of all key paths (typed) and json paths (key names only)
    json_paths = set()
    key_paths = set()
    # Collect all values encountered for each path type. 
    key_path_values = defaultdict(list)
    json_path_values = defaultdict(list)  # Should be equal (except key representation) to  a base_key_path_values collection
    def traverse_and_record(doc, parent_path="", ancestors=tuple()):
        for key, val in (doc.items() if type(doc) == dict else enumerate(doc)): 
            key_path, base_key_path = make_key_path(parent_path, key, val)
            
            key_paths.add(key_path)
            key_paths.add(base_key_path)

            json_path = ancestors + (key, )
            json_paths.add(json_path)

            # Record all values encountered
            key_path_values[key_path].append(val)
            json_path_values[json_path].append(val)

            if type(val) == list or type(val) == dict:
                traverse_and_record(val, key_path, json_path)

    for doc in collection:
        traverse_and_record(doc)
    json_paths = list(sorted(json_paths))  # Make json_path iteration order deterministic

    
    # We could also simply store values to test against, rather than indexes to those values
    N_TEST_VALUES = 25 # per path. Does not affect the list test values
    # json_path_to_test_values = {
    #     # Not perfect. maybe try making arr into set before sampling. May require len(arr) adjustment
    #     path: random.sample(valid_arr, k=min(N_TEST_VALUES, len(valid_arr))) if valid_arr else []
    #     # path: list(set([e for e in random.sample(arr, k=min(N_TEST_VALUES, len(arr))) if isinstance(e, Json_Primitive_No_Null)]))
    #     # path: [e for e in random.sample(arr, k=min(N_TEST_VALUES, len(arr))) if e is not None]
    #     # path: random.choices(set(arr), k=min(N_TEST_VALUES, len(set(arr))))
    #     for path, arr in json_path_values.items()
    #     if (valid_arr := list(set(e for e in arr if isinstance(e, Json_Primitive_No_Null)))) is not None
    # }

    json_path_to_test_values = {}
    for path, path_vals in json_path_values.items():
        primitives_arr = [e for e in path_vals if isinstance(e, Json_Primitive_No_Null)]
        if primitives_arr:  # Primitive test vals
            val_set = set(primitives_arr)
            test_vals = random.sample(list(val_set), k=min(N_TEST_VALUES, len(val_set)))
            json_path_to_test_values[path] = test_vals

        lists_arr = [l for l in path_vals if type(l) == list]
        if lists_arr:
            # Valid arr: Array where all elements belong to the same primitive type (basically, arrs of only ints or only strs)
            valid_arrs = []  # Note that subarrays can be of different types. But all members of each subarray should be of the same type
            for arr in path_vals:
                if not arr or type(arr) != list:
                    continue

                arr_type = type(arr[0])
                if not arr_type in (int, str):
                    continue
                if not (all(type(e) == arr_type for e in arr)):
                    continue

                valid_arrs.append(arr)

            if valid_arrs:
                # NOTE: Currently does not handle duplicates like the above solution does
                
                # Random selection of complete arrays
                # + Random subsets of that random selection
                # + Random element from each arr of the random selection
                test_vals = random.sample(list(valid_arrs), k=min(N_TEST_VALUES, len(valid_arrs)))
                all_test_vals = test_vals + [
                    random_subset
                    for arr in test_vals
                    if (random_subset := [e for e in arr if random.randint(0,1)])
                ] + [
                    [random.choice(arr)]
                    for arr in test_vals
                ]

                # Remove duplicates
                # all_test_vals = list(map(list, set(map(tuple, all_test_vals))))
                all_test_vals = list(set(map(tuple, all_test_vals))) # Leave lists as tuples

                json_path_to_test_values[path] = all_test_vals

        if path not in json_path_to_test_values:                    
            json_path_to_test_values[path] = []


    # pprint(sorted(json_paths)[:5])
    # pprint(sorted(key_paths)[:5])

    # json_path_values[('contributors',)] = [1, 2, "abc", None]
    # json_path_confusions = {k: count_types(v) for k, v in json_path_values.items()}
    # top_confused_paths = sorted(json_path_confusions.items(), key=lambda t: t[1], reverse=True)[:10]
    # print(top_confused_paths)

    #
    # Problem: Currently, we can only generate new values for testing estimates when there's 
    # only a single type for the key path. If there are multiple types, we cannot find the min and max
    # (at least for string and non-string primitives. The others work due to type coercion, but it's not a good solution)
    # of the collection of values.
    # Possible solution: Separate by type, generate test values, and then combine the test values again
    # with the same key

    # key_path_ranges = {k: (min(vs), max(vs)) for k, vs in key_path_values.items() if not all(v is None for v in vs) }
    
    def check_type(x, y, number=True):
        return type(x) == type(y) or (type(x) in (int, float) and type(y) in (int, float) if number else False)

    get_exists_comparator = lambda *_: (lambda _: 1)
    get_is_null_comparator = lambda *_: (lambda x: x is None)
    get_is_not_null_comparator = lambda *_: (lambda x: x is not None)
    get_eq_comparator = lambda tval: (lambda x: check_type(x, tval) and x == tval)
    get_lt_comparator = lambda tval: (lambda x: check_type(x, tval) and x < tval)
    get_gt_comparator = lambda tval: (lambda x: check_type(x, tval) and x > tval)
    get_contains_comparator = lambda tval: (lambda arr: check_type(arr, tval) and all(v in arr for v in tval))
    get_overlaps_comparator = lambda tval: (lambda arr: check_type(arr, tval) and any(v in arr for v in tval))
    get_memberof_comparator = lambda tval: (lambda arr: check_type(arr, [tval]) and tval in arr)

    
    # Mapping of the 3-tuple (path, operator, argument?) to the
    # ground truth value
    ground_truths: dict[tuple[str, str, Any | None], int] = {}
    print("Gathering ground truths...")
    for json_path in json_paths:
        for op_name, operator_getter in [
            ("exists", get_exists_comparator),
            ("is_null", get_is_null_comparator),
            ("is_not_null", get_is_not_null_comparator)
        ]:
            truth = get_operation_cardinality_2(json_path_values, json_path, operator_getter())
            ground_truths[(json_path, op_name, None)] = truth
        test_vals = json_path_to_test_values[json_path]
        for tval in test_vals:
            if isinstance(tval, Json_Primitive_No_Null):
                truth = get_operation_cardinality_2(
                    collection=json_path_values, 
                    json_path=json_path, 
                    operation=get_eq_comparator(tval)
                )
                ground_truths[(json_path, "equal_to", tval)] = truth

            if type(tval) in (int, float):
                truth = get_operation_cardinality_2(
                    collection=json_path_values, 
                    json_path=json_path, 
                    operation=get_lt_comparator(tval)
                )
                ground_truths[(json_path, "less_than", tval)] = truth

                truth = get_operation_cardinality_2(
                    collection=json_path_values, 
                    json_path=json_path, 
                    operation=get_gt_comparator(tval)
                )
                ground_truths[(json_path, "greater_than", tval)] = truth

            assert(not type(tval) == list) # Require tuples instead of lists, to make tval hashable
            if isinstance(tval, tuple):
                if len(tval) == 1:
                    member = tval[0]
                    truth = get_operation_cardinality_2(
                        collection=json_path_values, 
                        json_path=json_path, 
                        operation=get_memberof_comparator(member)
                    )
                    ground_truths[(json_path, "member_of", tval)] = truth

                truth = get_operation_cardinality_2(
                    collection=json_path_values, 
                    json_path=json_path, 
                    operation=get_contains_comparator(tval)
                )
                ground_truths[(json_path, "contains", tval)] = truth

                truth = get_operation_cardinality_2(
                    collection=json_path_values, 
                    json_path=json_path, 
                    operation=get_overlaps_comparator(tval)
                )
                ground_truths[(json_path, "overlaps", tval)] = truth

    print(len(ground_truths))

    settings_to_try = {
        # "stats_type": [st for st in StatType if st != StatType.BASIC],
        "stats_type": [st for st in StatType],
        "prune_strats": [
            [],
            [PruneStrat.MAX_NO_PATHS],
            [PruneStrat.MIN_FREQ],
            [PruneStrat.MAX_PREFIX_LENGTH],
            [PruneStrat.NO_TYPED_INNER_NODES, PruneStrat.UNIQUE_SUFFIX],
            
            # [PruneStrat.MAX_NO_PATHS, PruneStrat.NO_TYPED_INNER_NODES, PruneStrat.UNIQUE_SUFFIX],

            # Unless max_no high and min_freq high, this last one is redundant, as max_no will take precedence
            # [PruneStrat.MIN_FREQ, PruneStrat.MAX_NO_PATHS],
        ],
        "num_histogram_buckets": [6, 16, 32, 64, 128],
        # "num_histogram_buckets": [5, 15, 30,  100],
        # "num_histogram_buckets": [10, 50],
        "sampling_rate": [0.0, 0.7, 0.9, 0.98],
        # "sampling_rate": [0.0],
        "prune_params": [
            {
                "min_freq_threshold": 0.01,
                "max_no_paths_threshold": 200,
                "max_prefix_length_threshold": 4,
            },
            {
                "min_freq_threshold": 0.003,
                "max_no_paths_threshold": 400,
                "max_prefix_length_threshold": 4,
            },
            {
                "min_freq_threshold": 0.001,
                "max_no_paths_threshold": 500,
                "max_prefix_length_threshold": 5,
            },
        ],
    }

    log("Settings to try:", quiet=True)
    log(settings_to_try, quiet=True)

    def generate_settings_combinations():
        all_combinations = itertools.product(*settings_to_try.values())
        for comb in all_combinations:
            override_setting = {k: deepcopy(v) for k, v in zip(settings_to_try, comb)}
            
            ### Skip redundant settings ###

            # Skip variations on histogram size if histograms arent being created
            if override_setting["stats_type"] != StatType.HISTOGRAM and override_setting["num_histogram_buckets"] != settings_to_try["num_histogram_buckets"][0]:
                continue
            # Skip variations of prune params when no pruning strategy is active
            if not override_setting["prune_strats"] and override_setting["prune_params"] != settings_to_try["prune_params"][0]:
                continue

            # Skip variations of prune params when only UNIQUE_SUFFIX is active (as it takes no params)
            if override_setting["prune_strats"] == [PruneStrat.UNIQUE_SUFFIX] and override_setting["prune_params"] != settings_to_try["prune_params"][0]:
                continue

            ### Clean up settings, removing things that don't affect the outcome  ###
            
            if override_setting["stats_type"] != StatType.HISTOGRAM:
                del override_setting["num_histogram_buckets"]

            if not override_setting["prune_strats"] or override_setting["prune_strats"] == [PruneStrat.UNIQUE_SUFFIX]:
                del override_setting["prune_params"]
            else:
                if override_setting["prune_params"] and PruneStrat.MIN_FREQ not in override_setting["prune_strats"]:
                    del override_setting["prune_params"]["min_freq_threshold"]

                if override_setting["prune_strats"] and PruneStrat.MAX_NO_PATHS not in override_setting["prune_strats"]:
                    del override_setting["prune_params"]["max_no_paths_threshold"]

                if override_setting["prune_strats"] and PruneStrat.MAX_PREFIX_LENGTH not in override_setting["prune_strats"]:
                    del override_setting["prune_params"]["max_prefix_length_threshold"]

            yield override_setting


    print(f"Testing {len(list(generate_settings_combinations()))} Number of unique setting combinations...")
    settings_generator = generate_settings_combinations()
    
    test_settings = [
        # {
        #     'stats_type': StatType.HISTOGRAM,
        #     # 'prune_strats': [PruneStrat.NO_TYPED_INNER_NODES],
        #     'prune_strats': [],
        #     'num_histogram_buckets': 100, 
        #     'sampling_rate': 0,
        #     "prune_params": {}
        # },
        {
            'stats_type': StatType.BASIC,
            # 'prune_strats': [PruneStrat.NO_TYPED_INNER_NODES],
            'prune_strats': [PruneStrat.NO_TYPED_INNER_NODES, PruneStrat.UNIQUE_SUFFIX],
            'num_histogram_buckets': 100, 
            'sampling_rate': 0.98,
            "prune_params": {}
        },
    ]
    use_test_settings = False
    print_high_errors = False
    if use_test_settings:
        settings_generator = test_settings
    
    all_results = []
    log("Beginning analysis...")
    for override_settings in settings_generator:
        log("Using override settings:")
        log(override_settings)
        update_stats_settings(override_settings)
        
        stats, meta_stats = load_and_apply_stats()
        stats_size = len(json.dumps([stats, meta_stats], cls=KeyStatEncoder).encode('utf-8'))
        log("Using statistics of size:", stats_size)


        # Data has two columns: ground_truth and estimate
        exists_data = []
        is_null_data = []
        is_not_null_data = []
        eq_data = []
        lt_data = []
        gt_data = []
        contains_data = []
        overlaps_data = []
        memberof_data = []


        t0 = time.time_ns()
        for json_path in tqdm(json_paths):
            err_if_bad_est = lambda pred, thresh, est, tru: err_if_high_err(pred, thresh, est, tru, None, json_path=json_path, stats=stats, meta_stats=meta_stats) if print_high_errors else None
            # test_vals = []
            # for key_path in filter(lambda kp: kp in key_paths, get_possible_key_paths(json_path)):
            #     test_vals += generate_test_vals(value_gen_stats[key_path].min_val, value_gen_stats[key_path].max_val)
            #     test_vals += random.choices(json_path_values[json_path], k=min(50, len(json_path_values[json_path])))

            test_vals = json_path_to_test_values[json_path]
            
            # path / exists
            exists_ground_truth = get_operation_cardinality_2(collection=json_path_values, json_path=json_path, operation=get_exists_comparator())
            # _exists_ground_truth = get_operation_cardinality(collection=collection, json_path=json_path, operation=get_exists_comparator())
            # assert exists_ground_truth == _exists_ground_truth, (exists_ground_truth, _exists_ground_truth)
            exists_estimate = estimate_exists_cardinality(json_path_to_base_key_path(json_path=json_path))
            exists_data.append((exists_ground_truth, exists_estimate))
            err_if_bad_est("exists", 500, exists_estimate, exists_ground_truth)

            # is null
            is_null_ground_truth = get_operation_cardinality_2(collection=json_path_values, json_path=json_path, operation=get_is_null_comparator())
            is_null_estimate = estimate_is_null_cardinality(json_path_to_base_key_path(json_path=json_path))
            is_null_data.append((is_null_ground_truth, is_null_estimate))
            err_if_bad_est("is_null", 500, is_null_estimate, is_null_ground_truth)
            
            # is not null
            is_not_null_ground_truth = get_operation_cardinality_2(collection=json_path_values, json_path=json_path, operation=get_is_not_null_comparator())
            is_not_null_estimate = estimate_not_null_cardinality(json_path_to_base_key_path(json_path=json_path))
            is_not_null_data.append((is_not_null_ground_truth, is_not_null_estimate))
            err_if_bad_est("is_not_null", 500, is_not_null_estimate, is_not_null_ground_truth)

            for tval in test_vals:
                get_eq_truth = lambda: ground_truths[(json_path, "equal_to", tval)]
                get_lt_truth = lambda: ground_truths[(json_path, "less_than", tval)]
                get_gt_truth = lambda: ground_truths[(json_path, "greater_than", tval)]
                get_memberof_truth = lambda: ground_truths[(json_path, "member_of", tval)]
                get_contains_truth = lambda: ground_truths[(json_path, "contains", tval)]
                get_overlaps_truth = lambda: ground_truths[(json_path, "overlaps", tval)]

                err_if_bad_est = lambda pred, thresh, est, tru: err_if_high_err(pred, thresh, est, tru, tval, json_path=json_path, stats=stats, meta_stats=meta_stats) if print_high_errors else None

                # When a query is made, we find the type of the constant that's being compared against
                # We do the same thing here to find the correct key-path

                # equality operator
                if isinstance(tval, Json_Primitive_No_Null):
                    # eq_ground_truth = get_operation_cardinality_2(collection=json_path_values, json_path=json_path, operation=get_eq_comparator(tval))
                    # eq_ground_truth_2 = ground_truths[(json_path, "equal_to", tval)]
                    # assert(eq_ground_truth == eq_ground_truth_2)
                    
                    eq_ground_truth = get_eq_truth()
                    eq_estimate = estimate_eq_cardinality(json_path_to_key_path(json_path, tval), tval)
                    eq_data.append((eq_ground_truth, eq_estimate))
                    if type(tval) in (int, float):
                        err_if_bad_est("eq", 500, eq_estimate, eq_ground_truth)
                

                # Operators below only work with numeric values
                # if isinstance(tval, Json_Number): 
                if type(tval) in (int, float):
                    # less than operator
                    # lt_ground_truth = get_operation_cardinality_2(collection=json_path_values, json_path=json_path, operation=get_lt_comparator(tval))
                    # lt_ground_truth_2 = ground_truths[(json_path, "less_than", tval)]
                    lt_ground_truth = get_lt_truth()                    
                    lt_estimate = estimate_lt_cardinality(json_path_to_key_path(json_path, tval), tval)
                    lt_data.append((lt_ground_truth, lt_estimate))
                    err_if_bad_est("lt", 500, lt_estimate, lt_ground_truth)

                    # greater than operator
                    # gt_ground_truth = get_operation_cardinality_2(collection=json_path_values, json_path=json_path, operation=get_gt_comparator(tval))
                    gt_ground_truth = get_gt_truth()
                    gt_estimate = estimate_gt_cardinality(json_path_to_key_path(json_path, tval), tval)
                    gt_data.append((gt_ground_truth, gt_estimate))
                    err_if_bad_est("gt", 500, gt_estimate, gt_ground_truth)

                    # range
                    # But: how do we combine two vals to get a range?


                # LIKE? 
                if isinstance(tval, list):
                    ...
                    # member of
                    if len(tval) == 1:
                        member = tval[0]
                        # memberof_ground_truth = get_operation_cardinality_2(collection=json_path_values, json_path=json_path, operation=get_memberof_comparator(member))
                        memberof_ground_truth = get_memberof_truth()
                        memberof_estimate = estimate_memberof_cardinality(json_path_to_key_path(json_path, tval), member)
                        memberof_data.append((memberof_ground_truth, memberof_estimate))
                        err_if_bad_est("memberof", 500, memberof_estimate, memberof_ground_truth)

                    # contains operator
                    # contains_ground_truth = get_operation_cardinality_2(collection=json_path_values, json_path=json_path, operation=get_contains_comparator(tval))
                    contains_ground_truth = get_contains_truth()
                    contains_estimate = estimate_contains_cardinality(json_path_to_key_path(json_path, tval), tval)
                    contains_data.append((contains_ground_truth, contains_estimate))
                    err_if_bad_est("contains", 500, contains_estimate, contains_ground_truth)

                    # overlaps operator
                    # overlaps_ground_truth = get_operation_cardinality_2(collection=json_path_values, json_path=json_path, operation=get_overlaps_comparator(tval))
                    overlaps_ground_truth = get_overlaps_truth()
                    overlaps_estimate = estimate_overlaps_cardinality(json_path_to_key_path(json_path, tval), tval)
                    overlaps_data.append((overlaps_ground_truth, overlaps_estimate))
                    err_if_bad_est("overlaps", 500, overlaps_estimate, overlaps_ground_truth)

                    # print()
                    # print(len(tval))
                    # print(tval)
                    # if len(tval) == 1:
                    #     print((memberof_ground_truth, memberof_estimate))
                    # print((contains_ground_truth, contains_estimate))
                    # print((overlaps_ground_truth, overlaps_estimate))

        t1 = time.time_ns()
        analysis_time_taken = t1 - t0


        log()
        log(f"{settings.stats.stats_type.name} error statistics:")
        log("exists data:", end="\t\t")
        exists_results = analyze_data(exists_data)
        log("is_null data:", end="\t\t")
        is_null_results = analyze_data(is_null_data)
        log("is_not_null data:", end="\t")
        is_not_null_results = analyze_data(is_not_null_data)
        log("eq data:", end="\t\t")
        eq_results = analyze_data(eq_data)
        log("lt data:", end="\t\t")
        lt_results = analyze_data(lt_data)
        log("gt data:", end="\t\t")
        gt_results = analyze_data(gt_data)
        log("memberof data:", end="\t\t")
        memberof_results = analyze_data(memberof_data)
        log("contains data:", end="\t\t")
        contains_results = analyze_data(contains_data)
        log("overlaps data:", end="\t\t")
        overlaps_results = analyze_data(overlaps_data)
        log("\n")

        actual_data = {
            "exists": exists_data,
            "is_null": is_null_data,
            "is_not_null": is_not_null_data,
            "eq": eq_data,
            "lt": lt_data,
            "gt": gt_data,
            "memberof": memberof_data,
            "contains": contains_data,
            "overlaps": overlaps_data,
        }

        error_data = {
                "exists": exists_results,
                "is_null": is_null_results,
                "is_not_null": is_not_null_results,
                "eq": eq_results,
                "lt": lt_results,
                "gt": gt_results,
                "memberof": memberof_results,
                "contains": contains_results,
                "overlaps": overlaps_results,
        }
        
        if use_test_settings:
            ...
            # plot_errors(error_data, override_settings)


        meta_data = {
            "stats_size": stats_size,
            "time_taken": analysis_time_taken
        }
        all_results.append((override_settings, error_data, actual_data, meta_data))

    global_mem_tracker.record_global_memory()
    if not use_test_settings:
        with open(os.path.join(settings.stats.out_dir, settings.stats.filename + "_analysis.pickle"), "wb") as f:
            log(len(all_results))
            pickle.dump(all_results, f)


@time_tracker.record_time_used
def analyze_data(arr: list[tuple[int, int]]):
    if not arr:
        log("No values to analyze")
        return []

    error_percent = [calc_error(tru, est) for tru, est in arr]

    sorted_errors = sorted(error_percent)
    mean_err = sum(error_percent) / len(error_percent)
    median_err = sorted_errors[len(error_percent)//2]
    max_err = max(error_percent)
    _90th_percentile_err = sorted_errors[math.floor(len(error_percent)*0.9)]
    log(f"{mean_err=:6.1f},   {median_err=:6.1f},   {_90th_percentile_err=:6.1f},   {max_err=:6.1f}")

    return error_percent


Query = namedtuple("Query", ("stats", "err_keys", "split_key"))
def _examine_query(query: Query):
    with open(os.path.join(settings.stats.out_dir, settings.stats.filename + "_analysis.pickle"), "rb") as f:
        data = pickle.load(f)

    def is_match(st_query, setting):
        for k, v in st_query.items():
            if v is None or \
                (k in setting and v == setting[k]) or \
                (isinstance(v, tuple) and any(_v == setting[k] for _v in v)):
                continue

            return False

        return True

    
    def get_matches(query):
        matches = [t for t in data if is_match(query.stats, t[0])]
        pruned_matches = [
            (
                t[0],
                {ek: t[1][ek] for ek in query.err_keys},
                t[2],
            )
            for t in matches
        ]
        return pruned_matches


    def plot_data(tups, split_key=None):
        """
        Plots the data in a single group.
        If the a split key is given, the data will be grouped by 
        the values the split_key leads to, and each group will be plotted.
        """
        if split_key:
            groups = defaultdict(list)
            for t in tups:
                group_key = f"{split_key}={t[0][split_key]}"
                groups[group_key].append(t)

        else:
            groups = {"": tups}

        plot_dicts = {}  
        for group_key, group in groups.items():
            # identify by which settings the group members vary
            grouped_settings = defaultdict(set)
            for member in group:
                for k, v in member[0].items():
                    grouped_settings[k].add(v if not isinstance(v, list) else tuple(v))

            assert all(len(v) in (1, len(group)) for v in grouped_settings.values()), (grouped_settings, len(group))
            varieds = [k for k, v in grouped_settings.items() if len(v) > 1]

            # Fix group member names
            group_plot_dict = {}
            for member in group:
                for pred, err_data in member[1].items():
                    relevant_settings = {var: member[0][var] for var in varieds}
                    data_name = f"{pred}_{str(relevant_settings)}"
                    group_plot_dict[data_name] = err_data

            plot_dicts[group_key] = group_plot_dict


        for title, data_dict in plot_dicts.items():
            plot_errors(data_dict, title)


    query_results = get_matches(query)
    plot_data(query_results, split_key=query.split_key)


def examine_analysis_results():
    def plot_stat_size_v_err():
        # data is a list of tuples: (override_settings, error_data, meta_data)
        with open(os.path.join(settings.stats.out_dir, settings.stats.filename + "_analysis.pickle"), "rb") as f:
            data = pickle.load(f)

        errors, stats_sizes, stats_infos = [], [], []
        for tup in data:

            all_mean_errs = [
                sum(err_arr) / (len(err_arr) or 1)
                for err_arr in tup[1].values()
            ]
            num_empty_arrs = sum(not err_arr for err_arr in tup[1].values())
            mean_err = sum(all_mean_errs) / (len(all_mean_errs) - num_empty_arrs)
                
            # eq_err_data = tup[1]["eq"]
            # eq_err = sum(eq_err_data) / len(eq_err_data)

            # lt_err_data = tup[1]["lt"]
            # lt_err = sum(lt_err_data) / len(lt_err_data)

            # gt_err_data = tup[1]["gt"]
            # gt_err = sum(gt_err_data) / len(gt_err_data)

            # err = (eq_err + lt_err + gt_err) / 3

            # err = eq_err
            err = mean_err


            errors.append(err)
            stats_sizes.append(tup[3]["stats_size"])
            # stats_sizes.append(tup[3]["time_taken"])
            stats_infos.append(tup[0])

        scatterplot(errors, stats_sizes, stats_infos)


    # The query stats is a override_settings dict matching the settings you're looking for.
    # If the key leads to None, all values will be accepted.
    # If multiple values should accepted, wrap those values in a *tuple*
    query_1 = Query(
        stats={
            "stats_type": StatType.BASIC,
            "prune_strats": [],
            "num_histogram_buckets": None,
            "sampling_rate": (0.0, 0.3, 0.9, 0.98),
            "prune_params": {
                "min_freq_threshold": 0.01,
                "max_no_paths_threshold": 200,
            },
        },
        err_keys=[
            "eq"
        ],
        split_key="sampling_rate"
    )

    query_2 = Query(
        stats={
            "stats_type": None,
            "prune_strats": [PruneStrat.MIN_FREQ],
            "num_histogram_buckets": 3,
            "sampling_rate": None,
            "prune_params": {
                "min_freq_threshold": 0.01,
                "max_no_paths_threshold": 200,
            },
        },
        err_keys = [
            "exists"
        ],
        split_key=None
    )

    query_3 = Query(
        stats={
            "stats_type": StatType.HISTOGRAM,
            "prune_strats": [],
            "num_histogram_buckets": None,
            "sampling_rate": 0.0,
            "prune_params": None,
        },
        err_keys=[
            "gt",
            "lt",
        ],
        split_key=None
    )

    show_scatter = True

    if show_scatter:
        plot_stat_size_v_err()
    else:
        _examine_query(query_3)



def specific_queries():
    f_name = "mini.json"
    json_path = ('retweet_count', )
    compare_value = 3  # Count = 1 or 2 in "mini". 32(?) in "test"

    with open(os.path.join(settings.stats.data_dir, f_name)) as f:
        collection = json.load(f)

    for stats_type in StatType:
        stats, meta_stats = load_and_apply_stats(collection, stats_type)

        eq_ground_truth = get_operation_cardinality(collection=collection, json_path=json_path, operation=lambda x: x == compare_value)
        eq_estimate = estimate_eq_cardinality(json_path_to_key_path(json_path, compare_value), compare_value)
        
        log()
        log(eq_ground_truth, eq_estimate)
        log(stats[json_path_to_key_path(json_path, compare_value)])
        log()



def generate_test_vals(min_val, max_val, n_rand=10):
    # Should None always be included?
    # TODO: Figure out something better for all of these. Especially strings.
    # IDEA: Just sample from the values recorded for each key. Then do some 
        # value generation as well. 
        # We can either sample from the array, so that the likelihood follows the frequency of values
        # or from the set of the values, so that each value is just as likely to be chosen
    
    def gen_str_intermediary_vals(s1, s2):
        return []

    def gen_str_edge_vals(s):
        return [
            "a" + s,
            "Z" + s,
            s + "a",
            s + "Z",
            s + s,
        ] + [
            chr(ord(s[0])-1) + s,
            chr(ord(s[0])-1) + s[1:],
            chr(ord(s[0])+1) + s,
            chr(ord(s[0])+1) + s[1:],

            s + chr(ord(s[-1])-1),
            s[:-1] + chr(ord(s[-1])-1),
            s + chr(ord(s[-1])+1),
            s[:-1] + chr(ord(s[+1])-1),
        ] if s else []


    def gen_num_edge_vals(v):
        return [
            v,
            v - 0.0001,
            v + 0.0001,
            v - 1,
            v + 1,
            v - 10000,
            v + 10000,
            v / 1.001,
            v * 1.001,
            v / 2,
            v * 2,
            v / 1000000,
            v * 1000000,
        ]

    
    assert type(min_val) == type(max_val)
    val_type = type(min_val)

    if val_type in (int, float):
        return gen_num_edge_vals(min_val)\
            + gen_num_edge_vals(max_val)\
            + [random.randint(math.floor(min_val), math.ceil(max_val)) for _ in range(n_rand)]

    if val_type == bool:
        return [True, False]

    if val_type == str:
        return ["", min_val, max_val, min_val + max_val, max_val + min_val] \
            + gen_str_edge_vals(min_val)\
            + gen_str_edge_vals(max_val)\
            + gen_str_intermediary_vals(min_val, max_val)

    return []


def count_types(arr: list[Any], exclude_null=True, one_number_type=True):
    types = set(map(type, arr))
    return len(types) - (type(None) in types and exclude_null) - (float in types and int in types and one_number_type)

if __name__ == '__main__':
    assert count_types([1]) == 1
    assert count_types([1,1,1]) == 1
    assert count_types([1,2,3,4,None]) == 1
    assert count_types([1,2,3,4,None], exclude_null=False) == 2
    assert count_types([1,"2",3,4]) == 2
    assert count_types([1,[1],3,4]) == 2
    assert count_types([1,True,3,4]) == 2
    assert count_types([True,"1",2,[],{},None]) == 5
    assert count_types([True,"1",2,[],{},None], exclude_null=False) == 6

    # assert get_base_path("a_obj.up_down_arr.0") == ("a", "up_down", "0")

    # print(set(get_possible_key_paths(("a", "up_down", "0"))))
    # assert set(get_possible_key_paths(("a", "up_down", "0"))) == set((
    #     'a_obj.up_down_arr.0_str', 'a_obj.up_down_arr.0_bool', 'a_obj.up_down_arr.0_num', 'a_obj.up_down_arr.0'
    # ))

    # Test update_stats_settings function
    stats_settings_copy = copy(settings.stats)
    t1 = {"prune_strats": None}
    assert stats_settings_copy == settings.stats
    update_stats_settings(t1)
    assert stats_settings_copy != settings.stats
    unlock_settings()
    settings.stats = stats_settings_copy

    prune_params_copy = copy(settings.stats.prune_params)
    t2 = {"prune_params": {"min_freq_threshold": 0.000001}}
    assert prune_params_copy == settings.stats.prune_params
    update_stats_settings(t2)
    assert prune_params_copy != settings.stats.prune_params
    unlock_settings()
    settings.stats.prune_params = prune_params_copy


    # assert calc_error(1, 1) == 0, calc_error(1, 1)
    # assert calc_error(1, 2) == calc_error(2, 4) == calc_error(4, 8) == 1
    # assert calc_error(2, 1) == calc_error(4, 2) == calc_error(8, 4) == 1
    # assert calc_error(1, 100) == 99, calc_error(1, 100)
    # assert calc_error(100, 1) == 99, calc_error(100, 1)
    # assert calc_error(0.1, 0.01) == 9, calc_error(0.1, 0.01)
    # assert calc_error(0, 1) == 1, calc_error(0, 1)
    # assert calc_error(1, 0) == 1, calc_error(1, 0)


    run_analysis()

    # specific_queries()

    # examine_analysis_results()
