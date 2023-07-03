import json
import os
import tarfile
import sys
import gc

import psutil
from compute_stats import make_statistics
from compute_structures import KeyStatEncoder, StatType
from settings import settings

st_map = {
    "h": StatType.HISTOGRAM,
    "hist": StatType.HISTOGRAM,
    "histogram": StatType.HISTOGRAM,
    "ndv": StatType.BASIC_NDV,
}

if __name__ == '__main__':
    if len(sys.argv) not in (3, 4):
        print("Usage: file.py input_path output_path [stats_type]", file=sys.stderr)

    _, in_path, out_path, *_stats_type = sys.argv
    stats_type = st_map[_stats_type[0]] if _stats_type else settings.stats.stats_type
    settings.stats.stats_type = stats_type
    settings.stats.force_new = True

    # Simple case: input path is a json file 
    if in_path.endswith(".json"):
        with open(in_path, mode="r") as f:
            collection = json.load(f)

        statistics = make_statistics(collection)
        with open(out_path, mode="w") as f:
            json.dump(statistics, f, cls=KeyStatEncoder, indent=1)

    # More complex case: input path is a tar archive containing several json collections. 
    elif in_path.endswith(".tar"):
        with tarfile.open(in_path, mode='r') as archive:
            while (info := archive.next()):
                if not (info.isfile() and info.name.endswith(".json")):
                    continue
                
                print(info.name)
                gc.collect()  # Try to get ahead of OOM issues
                
                collection_name = info.name.split("_")[-1][:-5]
                json_file = archive.extractfile(info)
                
                
                # collection = (json.loads(line) for line in json_file)
                def yield_n_fields(generator, n): return (next(generator) for _ in range(n))
                collection = (json.loads(line) for line in yield_n_fields(json_file, 100_000))


                statistics = make_statistics(collection)
                # Treat output path as a directory, not a file
                stats_out_path = os.path.join(out_path, collection_name + ".json")
                with open(stats_out_path, mode="w") as f:
                    json.dump(statistics, f, cls=KeyStatEncoder, indent=1)
                
                # Try to prevent OOM issues
                json_file.close()
                del statistics
                del collection
                

    else:
        raise Exception("input_path has invalid file type identifier")

                


