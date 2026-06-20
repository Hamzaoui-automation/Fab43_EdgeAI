from ultralytics import YOLO
import cv2

model = YOLO("best.onnx", task="detect")

frame = cv2.imread("VID_IMG.png")

results = model.predict(frame, conf=0.25, device="cpu")

annotated = results[0].plot()

print(f"Detections: {len(results[0].boxes)}")
for box in results[0].boxes:
    cls   = int(box.cls)
    conf  = float(box.conf)
    label = model.names[cls]
    print(f"  {label:15s} {conf*100:.1f}%")

cv2.imshow("YOLOv8 - Single Image", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()