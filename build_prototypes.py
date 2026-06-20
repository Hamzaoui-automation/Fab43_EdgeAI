import cv2
import numpy as np
import onnxruntime as ort
import pickle
import os

# ── Load model ───────────────────────────────────────────
session = ort.InferenceSession("model.onnx")

# ── Preprocessing ────────────────────────────────────────
def preprocess(frame):
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img.astype(np.float32)

# ── Get embedding for one image ──────────────────────────
def get_embedding(frame):
    output = session.run(["output"], {"input": preprocess(frame)})
    return output[0][0]

# ── Compute prototype for a folder of images ─────────────
def compute_prototype(folder_path):
    embeddings = []
    files = [f for f in os.listdir(folder_path)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if len(files) == 0:
        print(f"ERROR: No images found in {folder_path}")
        return None

    print(f"Processing {len(files)} images from {folder_path}")

    for filename in files:
        path = os.path.join(folder_path, filename)
        frame = cv2.imread(path)

        if frame is None:
            print(f"  Skipping {filename} — could not read")
            continue

        embedding = get_embedding(frame)
        embeddings.append(embedding)
        print(f"  ✓ {filename}")

    prototype = np.mean(embeddings, axis=0)
    print(f"  → Prototype shape: {prototype.shape}")
    return prototype

# ── Build both prototypes ─────────────────────────────────
print("=== Building FAILURE prototype ===")
failure_proto = compute_prototype("prototype_images/failure")

print("\n=== Building SUCCESS prototype ===")
success_proto = compute_prototype("prototype_images/normal")

# ── Save new prototypes.pkl ───────────────────────────────
import torch

new_proto = {
    "prototypes" : torch.tensor(np.stack([failure_proto, success_proto])),
    "class_names": ["failure", "success"],
    "defect_idx" : torch.tensor(0)
}

with open("prototypes.pkl", "wb") as f:
    pickle.dump(new_proto, f)

print("\n=== Done ===")
print("prototypes.pkl has been rebuilt from your images")
print("failure prototype: computed from", "prototype_images/failure")
print("success prototype: computed from", "prototype_images/normal")