import json
import random
from collections import defaultdict, namedtuple
from typing import Any, Callable

from compute_stats import make_base_statistics, make_key_path
from compute_structures import StatType
from settings import settings

Json_Number = int | float
Json_Null = type(None)
Json_Primitive = Json_Number | bool | str | Json_Null
Json_value = Json_Primitive | list | dict

# Used to create ranges of real numbers used to stand in for python's builtin integer ranges
r_range = namedtuple("Real_Range", ("start", "stop"))

def get_base_path(kp, key_sep=".", type_sep="_"):
    return tuple(s.rsplit(type_sep, 1)[0] for s in kp.split(key_sep))

def get_possible_key_paths(json_path: tuple[str]) -> list[str]:
    ex_vals = [int(), float(), str(), bool(), None]

    out = ""
    for key, next_key in zip(json_path, json_path[1:]):
        out, _ = make_key_path(out, key, [] if next_key.isdigit() else {})
    

    return list(set(make_key_path(out, json_path[-1], ex)[0] for ex in ex_vals))


def get_operation_cardinality(collection, json_path, operation: Callable[[Any], bool]):
    def traverse(doc, path: list[str]):
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
    

def collect_path_data():
    with open(settings.stats.data_path) as f:
        data = json.load(f)

    # Sets of all key paths (typed) and json paths (key names only)
    json_paths = set()
    key_paths = set()
    # Collect all values encountered for each path type. 
    key_path_values = defaultdict(list)
    json_path_values = defaultdict(list)  # Should be equal (except key representation) to  a base_key_path_values collection
    def traverse(doc, parent_path="", ancestors=tuple()):
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
                traverse(val, key_path, json_path)

    for doc in data:
        traverse(doc)

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
    stats = make_base_statistics(data, StatType.BASIC)


    for json_path in json_paths:
        test_vals = []
        for key_path in filter(lambda kp: kp in key_paths, get_possible_key_paths(json_path)):
            test_vals += generate_test_vals(stats[key_path].min_val, stats[key_path].max_val)

        if test_vals:
            for val in test_vals:
                ...
                # eq
                # lt, gt, gte, lte
                # is null
                # is not null
                # range
                # path / exists
                # type eq / matching type? UNLIKELY

                # TODO: How do we combine two vals to get a range?


    



def generate_test_vals(min_val, max_val, type, n_rand=10):
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

    if type in (int, float):
        return gen_num_edge_vals(min_val)\
            + gen_num_edge_vals(max_val)\
            + [random.randint(min_val, max_val) for _ in range(n_rand)]

    if type == bool:
        return [True, False]

    if type == str:
        return ["", min_val, max_val, min_val + max_val, max_val + min_val] \
            + gen_str_edge_vals(min_val)\
            + gen_str_edge_vals(max_val)\
            + gen_str_intermediary_vals(min_val, max_val)


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

    print(get_possible_key_paths(("a", "up_down", "0")))
    assert set(get_possible_key_paths(("a", "up_down", "0"))) == set((
        'a_object.up_down_array.0_str', 'a_object.up_down_array.0_bool', 'a_object.up_down_array.0_number', 'a_object.up_down_array.0'
    ))

    collect_path_data()
