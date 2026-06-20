import pickle
import torch

with open("prototypes.pkl", "rb") as f:
    proto = pickle.load(f)

print("Class names:", proto["class_names"])
print("Defect index:", proto["defect_idx"])
print("Prototypes shape:", proto["prototypes"].shape)