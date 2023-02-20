import json 


with open("_dislikes_1.json",encoding="utf-8") as f_in:
    json_str = "[" + ",".join(f_in.readlines()) + "]"
    
json_data = json.loads(json_str)

json_data = json_data[:20_000]

with open("dislikes_small.json", mode="w", encoding="utf-8") as f_out:
    json.dump(json_data, f_out, indent=1)