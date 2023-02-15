import csv
import gzip
import json


with gzip.open("listings.csv.gz", mode='rt', encoding='utf-8') as f:
    listings = csv.DictReader(f)
    collection = list(listings)
    for doc in collection:
        # Convert string-representations of lists into lists
        if "amenities" in doc:
            doc["amenities"] = json.loads(doc["amenities"])
        if "host_verifications" in doc:
            val = doc["host_verifications"]
            val = val.replace("'", '\"')
            doc["host_verifications"] = json.loads(val)

        # Try to convert string-representations of numeric values to numbers
        for key, val in doc.items():
            try: 
                doc[key] = float(val)
                doc[key] = int(val)
            except:
                pass

    with open("listings.json", "w") as f_out:
        json.dump(collection, f_out, indent=1)