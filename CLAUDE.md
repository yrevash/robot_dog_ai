# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

REVO is a Python system that controls a robot dog via face recognition and hand gesture commands. It runs on a desktop/laptop connected to a webcam and sends HTTP commands to the robot.

## Dependencies

```bash
pip install opencv-contrib-python numpy pillow mediapipe
# Optional: pyinstaller (for building executables)
```

`opencv-contrib-python` is required (not plain `opencv-python`) — it provides `cv2.FaceDetectorYN` and `cv2.FaceRecognizerSF`.

ONNX models are auto-downloaded from GitHub on first run to `models/`. A `bark.wav` file (optional) can be placed at the project root for the bark sound.

## Running the Scripts

**GUI control center (main application):**
```bash
python face_control_center.py
```

**CLI face enrollment pipeline:**
```bash
# Step 1: Capture face images for a person
python face_embedding.py capture --name Alice --samples 25

# Step 2: Build the embedding database
python face_embedding.py build

# Step 3: Run real-time recognition
python face_embedding.py recognize

# Or do capture + build in one step:
python face_embedding.py enroll --name Alice --samples 25
```

**Simple face capture (legacy tool):**
```bash
python capture_faces.py
```

**Build standalone executables:**
```bash
pyinstaller RevoFaceControl.spec       # with mediapipe bundled
pyinstaller RevoFaceControl_v2.spec    # minimal spec
```

## Architecture

### Core Files

- **`face_embedding.py`** — CLI pipeline with four subcommands (`capture`, `build`, `recognize`, `enroll`). Handles model downloading, CLAHE/gamma lighting normalization, face detection (YuNet ONNX), embedding extraction (SFace ONNX), and identity matching with a two-gate system: cosine similarity threshold + centroid gate.

- **`face_control_center.py`** — Tkinter GUI that combines everything: face enrollment UI, live camera feed, gesture recognition via MediaPipe hand landmarks, and robot HTTP command dispatch. Runs detection/recognition in background threads. The `robot_action()` stub maps recognized person names to robot behavior profiles.

- **`capture_faces.py`** — Simple standalone capture script (no face detection quality checks). Used for collecting raw training images.

### Data Flow

1. `known_faces/<PersonName>/` → face images on disk
2. `face_embedding.py build` → `face_db.npz` (embeddings + per-person centroids)
3. At runtime: camera frame → YuNet detector → SFace recognizer → cosine similarity against `face_db.npz` → temporal vote smoothing (6-frame history, 4-vote threshold) → identity authorized
4. Authorized identity → MediaPipe hand gesture → HTTP POST to robot dog

### Identity Matching Logic (`match_identity` in `face_embedding.py`)

Two-gate approach:
1. Best sample cosine similarity must exceed `--threshold` (default 0.42) AND beat second-best by `--margin` (default 0.06)
2. Centroid similarity must exceed `--centroid-threshold` (default 0.40) AND match the same identity

### Gesture Commands

Defined in `GESTURE_COMMANDS` dict in `face_control_center.py`. See `GESTURE_CHEAT_SHEET.md` for the hand-sign-to-command mapping. Gestures only fire when a face is authorized, palm faces camera, and the sign is held stable (~0.4–0.8s).

### Robot Communication

Commands are sent as HTTP requests. The `robot_action()` function in `face_embedding.py` is a stub — replace with actual robot API calls. The GUI version sends JSON payloads with `person`, `command`, `source`, `timestamp` fields.

### Models Directory

| File | Purpose |
|------|---------|
| `models/face_detection_yunet_2023mar.onnx` | YuNet face detector |
| `models/face_recognition_sface_2021dec.onnx` | SFace face recognizer |
| `models/hand_landmarker.task` | MediaPipe hand landmark model |
