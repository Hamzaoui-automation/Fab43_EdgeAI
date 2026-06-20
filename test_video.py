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

# ── Classification ───────────────────────────────────────
def classify(embedding):
    distances  = [np.linalg.norm(embedding - prototypes[i])
                  for i in range(len(class_names))]
    best_idx   = int(np.argmin(distances))
    confidence = 1 / (1 + distances[best_idx])
    return class_names[best_idx], best_idx, confidence

# ── Video loop ───────────────────────────────────────────
cap = cv2.VideoCapture("test_video.mp4")

if not cap.isOpened():
    print("ERROR: Could not open test_video.mp4")
    exit()

frame_count = 0

# Initialize before loop so first frames don't crash
label, idx, conf = "unknown", -1, 0.0   # ← ADD THIS LINE

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video finished.")
        break

    frame_count += 1

    if frame_count % 5 == 0:
        embedding        = get_embedding(frame)
        label, idx, conf = classify(embedding)

    is_failure = (idx == defect_idx)
    color      = (0, 0, 255) if is_failure else (0, 255, 0)
    status     = "FAILURE" if is_failure else "NORMAL"

    cv2.putText(frame, f"Status     : {status}",
                (20, 50),  cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(frame, f"Confidence : {conf*100:.1f}%",
                (20, 90),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"Frame      : {frame_count}",
                (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("PrintGuard — Video Test", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()