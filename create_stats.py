import json
import sys
from compute_stats import make_statistics
from compute_structures import KeyStatEncoder

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: file.py input_path output_path", file=sys.stderr)

    _, in_path, out_path = sys.argv

    with open(in_path, mode="r") as f:
        collection = json.load(f)

    statistics = make_statistics(collection)
    print(statistics[0])
    print(statistics[1])
    with open(out_path, mode="w") as f:
        json.dump(statistics, f, cls=KeyStatEncoder)