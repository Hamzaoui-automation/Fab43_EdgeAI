import cv2
import numpy as np
import onnxruntime as ort
import pickle
from ultralytics import YOLO

# ── Load PrintGuard ──────────────────────────────────────
pg_session = ort.InferenceSession("model.onnx")

with open("prototypes.pkl", "rb") as f:
    proto = pickle.load(f)

prototypes  = proto["prototypes"].numpy()
class_names = proto["class_names"]
defect_idx  = int(proto["defect_idx"])

# ── Load YOLOv8 ──────────────────────────────────────────
yolo = YOLO("best.onnx", task="detect")

SEVERITY = {
    "spaghetti" : "CRITICAL",
    "stringing"  : "WARNING",
    "zits"       : "MINOR"
}

# ── PrintGuard functions ─────────────────────────────────
def preprocess_pg(frame):
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img.astype(np.float32)

def get_embedding(frame):
    output = pg_session.run(["output"], {"input": preprocess_pg(frame)})
    return output[0][0]

def classify_pg(embedding):
    distances  = [np.linalg.norm(embedding - prototypes[i])
                  for i in range(len(class_names))]
    best_idx   = int(np.argmin(distances))
    confidence = 1 / (1 + distances[best_idx])
    return class_names[best_idx], best_idx, confidence

# ── Printer action ───────────────────────────────────────
def handle_defect(defect_name):
    severity = SEVERITY.get(defect_name, "UNKNOWN")
    if severity == "CRITICAL":
        print(f"🔴 CRITICAL — {defect_name} detected → Pausing printer")
        pause_printer()
    elif severity == "WARNING":
        print(f"🟡 WARNING  — {defect_name} detected → Alert sent")
    elif severity == "MINOR":
        print(f"🟢 MINOR    — {defect_name} detected → Logged")

def pause_printer():
    print("🛑 PRINTER PAUSED")

# ── Webcam ───────────────────────────────────────────────
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam")
    exit()

# Optional: set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

pg_label, pg_idx, pg_conf = "unknown", -1, 0.0
yolo_result  = None
frame_count  = 0

print("Pipeline running — press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Lost webcam feed")
        break

    frame_count += 1

    # ── PrintGuard every 5 frames ────────────────────────
    if frame_count % 5 == 0:
        embedding                  = get_embedding(frame)
        pg_label, pg_idx, pg_conf = classify_pg(embedding)

        is_failure = (pg_idx == defect_idx)

        # ── YOLO fires only on failure ────────────────────
        if is_failure:
            yolo_result = yolo.predict(frame, conf=0.25,
                                       device="cpu", verbose=False)[0]
            for box in yolo_result.boxes:
                defect_name = yolo.names[int(box.cls)]
                handle_defect(defect_name)
        else:
            yolo_result = None

    # ── Display ──────────────────────────────────────────
    is_failure = (pg_idx == defect_idx)
    pg_color   = (0, 0, 255) if is_failure else (0, 255, 0)
    pg_status  = "FAILURE" if is_failure else "NORMAL"

    display = yolo_result.plot() if yolo_result and len(yolo_result.boxes) > 0 \
              else frame.copy()

    overlay = display.copy()
    cv2.rectangle(overlay, (10, 10), (460, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, display, 0.6, 0, display)

    cv2.putText(display, f"PrintGuard : {pg_status} ({pg_conf*100:.1f}%)",
                (20, 50),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, pg_color, 2)

    if yolo_result and len(yolo_result.boxes) > 0:
        n = len(yolo_result.boxes)
        cv2.putText(display, f"YOLO       : {n} defect(s) found",
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    elif is_failure:
        cv2.putText(display, "YOLO       : no defects localized",
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    cv2.imshow("PrintGuard + YOLOv8 — Live", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()