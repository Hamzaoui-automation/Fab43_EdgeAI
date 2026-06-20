import cv2
import numpy as np
import onnxruntime as ort
import pickle

# ── Load model ───────────────────────────────────────────
session = ort.InferenceSession("model.onnx")

# ── Load prototypes ──────────────────────────────────────
with open("prototypes.pkl", "rb") as f:
    proto = pickle.load(f)

prototypes  = proto["prototypes"].numpy()
class_names = proto["class_names"]
defect_idx  = int(proto["defect_idx"])

# ── Preprocessing ────────────────────────────────────────
def preprocess(frame):
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img.astype(np.float32)

# ── Inference ────────────────────────────────────────────
def get_embedding(frame):
    output = session.run(["output"], {"input": preprocess(frame)})
    return output[0][0]

# ── Classification (Euclidean distance — ProtoNet standard) ──
def classify(embedding):
    distances = [np.linalg.norm(embedding - prototypes[i])
                 for i in range(len(class_names))]

    best_idx   = int(np.argmin(distances))          # closest prototype
    confidence = 1 / (1 + distances[best_idx])      # convert to 0-1 score

    print(f"  failure distance : {distances[0]:.4f}")
    print(f"  success distance : {distances[1]:.4f}")

    return class_names[best_idx], best_idx, confidence

# ── Run on single image ──────────────────────────────────
frame = cv2.imread("testing2.png")

if frame is None:
    print("ERROR: testing2.png not found")
else:
    embedding        = get_embedding(frame)
    label, idx, conf = classify(embedding)
    is_failure       = (idx == defect_idx)

    print("\n=== RESULT ===")
    print(f"Label      : {label}")
    print(f"Status     : {'FAILURE' if is_failure else 'NORMAL'}")
    print(f"Confidence : {conf*100:.1f}%")