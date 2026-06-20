import pickle
import numpy as np

with open("prototypes.pkl", "rb") as f:
    proto = pickle.load(f)

print("Type:", type(proto))

if isinstance(proto, dict):
    print("Keys:", proto.keys())
    for key, value in proto.items():
        print(f"  Class '{key}': shape = {np.array(value).shape}")

elif isinstance(proto, list):
    print("Length:", len(proto))
    print("First item type:", type(proto[0]))

else:
    print(proto)