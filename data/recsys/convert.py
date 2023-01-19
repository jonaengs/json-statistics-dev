import csv
import json
from multiprocessing import parent_process
import sys

def fix_file(path):
    d = {'{': 1, '}': -1}
    with open(path) as f_in, open(path + "_fixed", "w+") as f_out:
        paren_count = 0
        for c in f_in.read():
            paren_count += d.get(c, 0)

            if c == '{' and paren_count == 1:
                f_out.write('"{')
            elif c == '}' and paren_count == 0: 
                f_out.write('}"')
            elif c == '"':
                f_out.write("'")
            else:
                f_out.write(c)



def store_json(path):
    with open(path + "_fixed") as f:
        reader = csv.DictReader(f)

        with open(path[:-4] + ".json", mode="w+") as out:
            out.write("[")
            # print(next(reader)["tweet_in_json_format"])
            out.write(",\n".join(row["tweet_in_json_format"] for row in reader))
            out.write("]")


def store_json_better(path):
    with open(path) as f:
        lines = f.readlines()[1:]  # skip header
        json_data = (line.split(",", 4)[4:][0] for line in lines)

    with open(path[:-4] + ".json", mode="w+") as out:
        out.write("[\n",)
        out.write(",".join(json_data))
        out.write("\n]")
    
if __name__ == '__main__':
    file_path = "" or sys.argv[1]

    # fix_file(file_path)
    # store_json(file_path)
    
    store_json_better(file_path)


