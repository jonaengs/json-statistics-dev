from dataclasses import dataclass, asdict
import json
from collections import defaultdict
from pprint import pprint
import math



# TODO: Check that the case of repeat keys ({a: 1, a: 2}) is covered. Should be fine as long a json library is used
# TODO: Figure out a key path format that is unlikely to collide with existing keys
    # Problem: {"a": {"a": []}} and {"a_dict.a": []} gives "a_dict.a_list" = 2



# Statistics design
# Q: Keep only most frequent type for all key paths (like JSON Tiles)? What if two types are split 50/50 in freq?
# Idea: Calculate some mean cardinality of all pruned (due to infrequency) key-paths, and use that as a stand-in




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

@dataclass
class KeyStat:
    count: int = 0
    null_count: int = 0
    min_val: (None | int) = None
    max_val: (None | int) = None

    def __repr__(self) -> str:
        return str({k:v for k, v in asdict(self).items() if v is not None})


class KeyStatEncoder(json.JSONEncoder):
    def default(self, o):
        if type(o) == KeyStat:
            return asdict(o)
        return super().default(o)


def make_base_statistics(collection):
    def make_key_path(ex_path, key, val)-> tuple[str, str]:
        type_str = {
            list: "array",
            dict: "object",
            None.__class__: "", 
        }.get(type(val), type(val).__name__)

        parent_path = ex_path + ("." if ex_path else "")

        return parent_path + str(key) + ("_" + type_str if type_str else ""), parent_path + str(key)

    
    stats = defaultdict(KeyStat)

    def traverse(doc, parent_path=""):
        if type(doc) == list:
            for key, val in enumerate(doc):  # use index as key
                # record keypath both with and without type information
                key_str, base_key_str = make_key_path(parent_path, key, val)
                stats[key_str].count += 1
                stats[base_key_str].count += val is not None  # Change this if null values get a type suffix
                stats[key_str].null_count += val is None

                if type(val) == int or type(val) == float:
                    stats[key_str].min_val = min(val, stats[key_str].min_val or math.inf)
                    stats[key_str].max_val = max(val, stats[key_str].max_val or -math.inf)
                elif type(val) == list or type(val) == dict:
                    traverse(val, key_str)

        if type(doc) == dict:
            for key, val in doc.items():
                key_str, base_key_str = make_key_path(parent_path, key, val)
                stats[key_str].count += 1
                stats[base_key_str].count += val is not None  # Change this if null values get a type suffix
                stats[key_str].null_count += val is None

                if type(val) == int or type(val) == float:
                    stats[key_str].min_val = min(val, stats[key_str].min_val or math.inf)
                    stats[key_str].max_val = max(val, stats[key_str].max_val or -math.inf)
                elif type(val) == list or type(val) == dict:
                    traverse(val, key_str)

    print(f"creating statistics for a collection of {len(collection)} documents...")
    for doc in collection:
        traverse(doc)

    return dict(stats)


# Removes uncommon paths. Returns some summary statistics as well
def make_statistics(collection) -> list[dict, dict]:
    # Tunable vars:
    min_freq_threshold = 0.001

    base_stats = make_base_statistics(collection)
    pruned_path_stats = {}


    min_count_threshold = int(min_freq_threshold * len(collection))
    min_count_included = None

    _pruned_count = 0
    for key_path, path_stats in base_stats.items():
        if path_stats.count >= min_count_threshold:
            min_count_included = min(min_count_included or math.inf, path_stats.count)
            pruned_path_stats[key_path] = path_stats
        else:
            _pruned_count += 1

    print("num_pruned", _pruned_count, f"({len(base_stats)} unique paths total)")

    summary_stats = {"min_count_included": min_count_included, "min_count_threshold": min_count_threshold}
    return [pruned_path_stats, summary_stats]


# Maybe there should be one of these for each approach
def get_estimate(path, accessor):
    with open(path) as f:
        path_stats, summary_stats = json.load(f)

    try:
        return accessor(path_stats)
    except:
        return summary_stats["min_count_threshold"]


if __name__ == '__main__':
    data_path = "data/recsys/test.json"
    stats_path = "test_stats.json"

    with open(data_path) as f:
        stats = make_statistics(json.load(f))
        # pprint(stats[0])
        print("-"*50)
        pprint(stats[1])
        print("-"*50)

    with open("test_stats.json", mode="w+") as f:
        print(len(json.dumps(stats, cls=KeyStatEncoder)))
        json.dump(stats, f, cls=KeyStatEncoder)

    print(compute_cardinality(data_path, lambda d: d["retweeted_status"]))
    print(get_estimate(stats_path, lambda d: d["retweeted_status"]["count"]))