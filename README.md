# Fab43 EdgeAI — 3D Print Failure Detection

> A real-time edge AI pipeline for detecting and localizing 3D printing failures using **PrintGuard** and **YOLOv8**, developed at [Fab43](https://fab43.com) — FabLab & Innovation Hub, Tunisia.

**Author:** Mohamed Hamzaoui  
**Status:** 🚧 Prototyping  
**Target hardware:** Raspberry Pi 5 + Pi Camera Module 3 Wide

---

## Overview

This project implements a two-stage computer vision pipeline that monitors a 3D printer in real time and detects failures before they waste material:

```
Camera Frame
      ↓
┌─────────────────────┐
│  Stage 1            │
│  PrintGuard         │  ← runs every frame (lightweight)
│  ShuffleNetV2       │
│  Anomaly Detection  │
└─────────────────────┘
      ↓ FAILURE only
┌─────────────────────┐
│  Stage 2            │
│  YOLOv8n            │  ← fires only when anomaly detected
│  Defect Localization│
└─────────────────────┘
      ↓
  Pause Printer (G-code M0)
```

### Why two stages?

- **PrintGuard** is a few-shot prototype network — fast, lightweight, runs on every frame
- **YOLOv8** only activates when PrintGuard flags a failure — saves compute
- This hybrid approach is practical for edge deployment on a Raspberry Pi 5

---

## Detectable Failure Types

| Class | Severity | Action |
|---|---|---|
| `spaghetti` | 🔴 Critical | Pause printer |
| `stringing` | 🟡 Warning | Alert |
| `zits` | 🟢 Minor | Log |

---

## Project Structure

```
Fab43_EdgeAI/
│
├── model.onnx                  # PrintGuard encoder (ShuffleNetV2)
├── prototypes.pkl              # Prototype vectors (failure / success)
├── opt.json                    # PrintGuard training config
├── best.onnx                   # YOLOv8n defect localizer
│
├── build_prototypes.py         # Rebuild prototypes from your own images
│
├── combined_image.py           # Full pipeline — single image
├── combined.py                 # Full pipeline — video file
├── Realtime_Pipeline.py        # Full pipeline — live webcam
│
├── test_image.py               # PrintGuard only — single image
├── test_video.py               # PrintGuard only — video
├── test_yolo_IMG.py            # YOLOv8 only — single image
├── test_yolo_VID               # YOLOv8 only — video
│
├── inspect_prototypes.py       # Debug: inspect prototypes.pkl
├── model_caracteristics.py     # Debug: inspect ONNX model I/O
├── read_opt.py                 # Debug: read opt.json
│
├── prototype_images/
│   ├── normal/                 # Images used to build success prototype
│   └── failure/               # Images used to build failure prototype
│
└── requirements.txt
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Hamzaoui-automation/Fab43_EdgeAI.git
cd Fab43_EdgeAI
```

### 2. Download model files

The PrintGuard model files are not included in this repo due to size. Download them from the official PrintGuard Hugging Face repository and place them in the root folder:

```
model.onnx
prototypes.pkl
opt.json
```

### 3. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux / Pi
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Run on a single image

```bash
python combined_image.py
```

### Run on a video file

```bash
python combined.py
```

### Run real-time (webcam)

```bash
python Realtime_Pipeline.py
```

---

## Rebuild Prototypes

The default `prototypes.pkl` was built from generic online images. For accurate detection on your specific printer and camera, rebuild the prototypes using your own images:

1. Add 10–20 images of **normal prints** to `prototype_images/normal/`
2. Add 10–20 images of **failed prints** to `prototype_images/failure/`
3. Run:

```bash
python build_prototypes.py
```

This overwrites `prototypes.pkl` with vectors computed from your images.

---

## Model Details

### PrintGuard (Stage 1)

| Property | Value |
|---|---|
| Architecture | ShuffleNetV2 (protonet_conv) |
| Input | `[1, 3, 224, 224]` |
| Output | `[1, 1024]` feature embedding |
| Classification | Euclidean distance to prototypes |
| Classes | `failure`, `success` |

### YOLOv8n (Stage 2)

| Property | Value |
|---|---|
| Architecture | YOLOv8 nano |
| Input | `[1, 3, 640, 640]` |
| Output | `[1, 7, 8400]` |
| Classes | `spaghetti`, `stringing`, `zits` |

---

## Target Deployment

| Component | Choice |
|---|---|
| SBC | Raspberry Pi 5 — 8GB |
| Camera | Pi Camera Module 3 Wide |
| Printer interface | USB serial — G-code `M0` |
| Runtime | ONNX Runtime (CPU) |

Printer control via `pyserial` — to be integrated in hardware phase.

---

## Roadmap

- [x] PrintGuard ONNX inference pipeline
- [x] YOLOv8n ONNX inference pipeline
- [x] Two-stage combined pipeline (image / video / realtime)
- [x] Severity-based defect handling
- [ ] Rebuild prototypes from real printer images
- [ ] Deploy on Raspberry Pi 5
- [ ] Serial printer control (G-code M0)
- [ ] Logging system

---

## Acknowledgements

- [PrintGuard](https://huggingface.co/harrymcconell/printguard) by Harry McConnell
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Fab43 FabLab](https://fab43.com) — Tunisia
