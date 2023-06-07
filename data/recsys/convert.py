import json
import sys


def store_as_json(path):
    with open(path) as f:
        lines = f.readlines()[1:]  # skip header
        json_data = (line.split(",", 4)[4:][0] for line in lines)

    with open(path[:-4] + ".json", mode="w+") as out:
        out.write("[\n",)
        out.write(",".join(json_data))
        out.write("\n]")

def create_custom_size():
    name = "tiiiiny"
    size = 100

    with open("test.json") as f:
        data = json.load(f)

    with open(name + ".json", "w+") as f:
        json.dump(data[:size], f)

    
if __name__ == '__main__':
    file_path = sys.argv[1] if len(sys.argv) > 1 else "test.dat"
    
    store_as_json(file_path)
    # create_custom_size()


