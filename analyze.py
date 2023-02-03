import json
import math
import os
import random
from collections import defaultdict, namedtuple
from typing import Any, Callable
import typing

from compute_structures import StatType
from compute_stats import make_base_statistics, make_key_path
from settings import settings
from use_stats import compute_and_set_stats, estimate_eq_cardinality, estimate_exists_cardinality, estimate_gt_cardinality, estimate_is_null_cardinality, estimate_lt_cardinality, estimate_not_null_cardinality
from logger import log
import data_cache

Json_Number = int | float
Json_Null = type(None)
Json_Primitive = Json_Number | bool | str | Json_Null
Json_value = Json_Primitive | list | dict

# Used to create ranges of real numbers used to stand in for python's builtin integer ranges
r_range = namedtuple("Real_Range", ("start", "stop"))

def get_base_path(kp, key_sep=".", type_sep="_"):
    # Removes the type suffix from the final typed key in a key path string
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


def get_operation_cardinality_2(collection: dict, json_path, operation: Callable[[Any], bool]):
    """
    Doesn't take a normal doc collection. Instead, takes mapping of json_path -> all values from that path.
    Should be more efficient, but may lead to issues if multiple types are present in the array 
    (like if the path ever leads to null). We should be able to remedy this by including a type 
    check in the operation function though 
    (so instead of 'lambda x: x == val', do 'lambda x: type(x) == type(val) and x == val').
    """
    return sum(map(operation, collection[json_path]))
    

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

    # pprint(sorted(json_paths)[:5])
    # pprint(sorted(key_paths)[:5])

    # json_path_values[('contributors',)] = [1, 2, "abc", None]
    # json_path_confusions = {k: count_types(v) for k, v in json_path_values.items()}
    # top_confused_paths = sorted(json_path_confusions.items(), key=lambda t: t[1], reverse=True)[:10]
    # print(top_confused_paths)

    def get_range(arr):
        return min(arr), max(arr)

    #
    # Problem: Currently, we can only generate new values for testing estimates when there's 
    # only a single type for the key path. If there are multiple types, we cannot find the min and max
    # (at least for string and non-string primitives. The others work due to type coercion, but it's not a good solution)
    # of the collection of values.
    # Possible solution: Separate by type, generate test values, and then combine the test values again
    # with the same key

    # key_path_ranges = {k: (min(vs), max(vs)) for k, vs in key_path_values.items() if not all(v is None for v in vs) }
    
    # TODO: This won't do if we enable sampling before this point
    value_gen_stats = make_base_statistics(collection, StatType.BASIC)

    get_exists_comparator = lambda *_: (lambda _: 1)
    get_is_null_comparator = lambda *_: (lambda x: x is None)
    get_is_not_null_comparator = lambda *_: (lambda x: x is not None)
    get_eq_comparator = lambda val: (lambda x: x == val and type(x) == type(val))
    get_lt_comparator = lambda val: (lambda x: type(x) == type(val) and x < val)
    get_gt_comparator = lambda val: (lambda x: type(x) == type(val) and x > val)

    # Data has two columns: ground_truth and estimate
    exists_data = []
    is_null_data = []
    is_not_null_data = []
    eq_data = []
    lt_data = []
    gt_data = []

    # TODO: We're currently retrieving (likely) different random values to test against for each stat type.
    # This is obviously bad if we want a valid comparison. 
    # How about pre-computing which indices to sample, or just trying every single value
    
    log("Gathering data for analysis...")
    for stats_type in StatType:
        settings.stats.stats_type = stats_type
        stats, meta_stats = compute_and_set_stats()

        for json_path in json_paths:
            test_vals = []
            for key_path in filter(lambda kp: kp in key_paths, get_possible_key_paths(json_path)):
                test_vals += generate_test_vals(value_gen_stats[key_path].min_val, value_gen_stats[key_path].max_val)
                test_vals += random.choices(json_path_values[json_path], k=min(50, len(json_path_values[json_path])))
            
            # path / exists
            exists_ground_truth = get_operation_cardinality_2(collection=json_path_values, json_path=json_path, operation=get_exists_comparator())
            exists_estimate = estimate_exists_cardinality(json_path_to_base_key_path(json_path=json_path))
            exists_data.append((exists_ground_truth, exists_estimate))

            # is null
            is_null_ground_truth = get_operation_cardinality_2(collection=json_path_values, json_path=json_path, operation=get_is_null_comparator())
            is_null_estimate = estimate_is_null_cardinality(json_path_to_base_key_path(json_path=json_path))
            is_null_data.append((is_null_ground_truth, is_null_estimate))
            
            # is not null
            is_not_null_ground_truth = get_operation_cardinality_2(collection=json_path_values, json_path=json_path, operation=get_is_not_null_comparator())
            is_not_null_estimate = estimate_not_null_cardinality(json_path_to_base_key_path(json_path=json_path))
            is_not_null_data.append((is_not_null_ground_truth, is_not_null_estimate))

            for val in test_vals:
                if not isinstance(val, typing.get_args(Json_Primitive)) or val is None:
                    continue

                # When a query is made, we find the type of the constant that's being compared against
                # We do the same thing here to find the correct key-path

                # equality operator
                eq_ground_truth = get_operation_cardinality_2(collection=json_path_values, json_path=json_path, operation=get_eq_comparator(val))
                eq_estimate = estimate_eq_cardinality(json_path_to_key_path(json_path, val), val)
                eq_data.append((eq_ground_truth, eq_estimate))
                
                # if (error := abs(eq_ground_truth - eq_estimate) / (eq_ground_truth or 1)) > 500 :
                #     print()
                #     print(f"bad estimate: {error=}, {eq_ground_truth=}, {eq_estimate=}")
                #     print("val:", val)
                #     print(json_path, json_path_to_key_path(json_path, val))
                #     print(stats[json_path_to_key_path(json_path, val)])
                #     print(meta_stats)
                #     print()
                #     return
                

                # Operators below only work with numeric values
                if type(val) in (float, int): 
                    # less than operator
                    try:
                        lt_ground_truth = get_operation_cardinality_2(collection=json_path_values, json_path=json_path, operation=get_lt_comparator(val))
                        lt_estimate = estimate_lt_cardinality(json_path_to_key_path(json_path, val), val)
                        lt_data.append((lt_ground_truth, lt_estimate))
                    except:
                        log(json_path)
                        log(val)
                        log(json_path_values[json_path])

                    # greater than operator
                    gt_ground_truth = get_operation_cardinality_2(collection=json_path_values, json_path=json_path, operation=get_gt_comparator(val))
                    gt_estimate = estimate_gt_cardinality(json_path_to_key_path(json_path, val), val)
                    gt_data.append((gt_ground_truth, gt_estimate))

                    # range
                    # But: how do we combine two vals to get a range?


                # LIKE? IN ARRAY?


        log()
        log(f"{stats_type.name} error statistics:")
        log("exists data:", end="\t\t")
        analyze_data(exists_data)
        log("is_null data:", end="\t\t")
        analyze_data(is_null_data)
        log("is_not_null data:", end="\t")
        analyze_data(is_not_null_data)
        log("eq_data:", end="\t\t")
        analyze_data(eq_data)
        log("lt_data:", end="\t\t")
        analyze_data(lt_data)
        log("gt_data:", end="\t\t")
        analyze_data(gt_data)

def analyze_data(arr: list[tuple[int, int]]):
    error_percent = [
        abs(tru - est) / (tru or 1)
        for tru, est in arr
    ]

    mean_error = sum(error_percent) / len(error_percent)
    median_error = sorted(error_percent)[len(error_percent)//2]
    max_error = max(error_percent)
    log(f"{mean_error=:.2f},\t{median_error=:.2f},\t{max_error=:.2f}")


def specific_queries():
    f_name = "mini.json"
    json_path = ('retweet_count', )
    compare_value = 3  # Count = 1 or 2 in "mini". 32(?) in "test"

    with open(os.path.join(settings.stats.data_dir, f_name)) as f:
        collection = json.load(f)

    for stats_type in StatType:
        stats, meta_stats = compute_and_set_stats(collection, stats_type)

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

    assert get_base_path("a_object.up_down_array.0") == ("a", "up_down", "0")

    assert set(get_possible_key_paths(("a", "up_down", "0"))) == set((
        'a_object.up_down_array.0_str', 'a_object.up_down_array.0_bool', 'a_object.up_down_array.0_number', 'a_object.up_down_array.0'
    ))

    # run_analysis()

    specific_queries()
