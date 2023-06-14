import csv
import gzip
import json
import os
import sys
import math

def str_to_num(collection: list[dict]) -> list[dict]:
    # Try to convert string representations of numeric values to numbers
    for doc in collection:
        for key, val in doc.items():
            # Don't convert ids to integers as they can be too large for what is allowed of a JSON "integer"
            if type(val) != str or ("_id" in key or key in ("id", "license")):
                continue
            try:
                as_float = float(val)
                if not (math.isnan(as_float) or math.isinf(as_float)):
                    doc[key] = as_float
                doc[key] = int(val)
            except:
                pass

    return collection

def str_to_lists(collection: list[dict]) -> list[dict]:
    # Convert string-representations of lists into lists
    for doc in collection:
        if "amenities" in doc:
            doc["amenities"] = json.loads(doc["amenities"])
        if doc.get("host_verifications") is not None:
            doc["host_verifications"] = eval(doc["host_verifications"])

    return collection

def convert(name: str, processing_funcs=[]):
    print(name)

    gz_file = f"{name}.csv.gz"
    csv_file = f"{name}.csv"

    # Open file into f
    if os.path.exists(gz_file):
        f = gzip.open(gz_file, mode='rt', encoding='utf-8')
    elif os.path.exists(csv_file):
        f = open(csv_file, mode='r', encoding='utf-8')
    else:
        raise Exception(f"Cannot find {gz_file} or {csv_file}")
    
    # Convert csv into a list of dicts
    listings = csv.DictReader(f)
    collection = list(listings)

    # Apply processing functions if supplied
    for func in processing_funcs:
        collection = func(collection)


    # Write data as json into a subdirectory that is created now if need be
    if not os.path.exists("./converted"):
        os.mkdir("./converted")
    with open(f"./converted/{name}.json", "w") as f_out:
        json.dump(collection, f_out, indent=1)


if __name__ == '__main__':
    # Call with argument to change base directory
    if len(sys.argv) > 1:
        os.chdir(sys.argv[1])

    convert("listings", [str_to_num, str_to_lists])
    convert("reviews", [str_to_num])
    convert("neighbourhoods")
    convert("calendar", [str_to_num])