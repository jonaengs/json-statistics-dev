import json
import os
import tarfile
import sys
import gc

import psutil
from compute_stats import make_statistics
from compute_structures import KeyStatEncoder

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: file.py input_path output_path", file=sys.stderr)

    _, in_path, out_path = sys.argv

    # Simple case: input path is a json file 
    if in_path.endswith(".json"):
        with open(in_path, mode="r") as f:
            collection = json.load(f)

        statistics = make_statistics(collection)
        with open(out_path, mode="w") as f:
            json.dump(statistics, f, cls=KeyStatEncoder, indent=1)

    # More complex case: input path is a tar archive containing several json collections. 
    elif in_path.endswith(".tar"):
        print("mem before open tar:", psutil.Process().memory_info().rss)
        with tarfile.open(in_path, mode='r') as archive:
            print("mem after open tar:", psutil.Process().memory_info().rss)  
            while (info := archive.next()):
                print()
                print(info.name)
                print("mem while start before gc:", psutil.Process().memory_info().rss)
                gc.collect()
                print("mem while start after gc:", psutil.Process().memory_info().rss)

                if not (info.isfile() and info.name.endswith(".json")):
                    continue
                
                collection_name = info.name.split("_")[-1][:-5]
                json_file = archive.extractfile(info)
                print("mem file open:", psutil.Process().memory_info().rss)
                collection = (json.loads(line) for line in json_file)
                print("mem collection created:", psutil.Process().memory_info().rss)

                statistics = make_statistics(collection)
                print("mem statistics created:", psutil.Process().memory_info().rss)

                # Treat output path as a directory, not a file
                stats_out_path = os.path.join(out_path, collection_name + ".json")
                with open(stats_out_path, mode="w") as f:
                    json.dump(statistics, f, cls=KeyStatEncoder, indent=1)
                
                # Try to prevent OOM issues
                json_file.close()
                del statistics
                del collection
                
                print("")

    else:
        raise Exception("input_path has invalid file type identifier")

                


