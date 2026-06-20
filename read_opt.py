import json

with open("opt.json", "r") as f:
    opt = json.load(f)

print(json.dumps(opt, indent=2))