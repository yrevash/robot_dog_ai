# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

REVO is a Python system that controls a robot dog via face recognition and hand gesture commands. It runs on a desktop/laptop connected to a webcam and sends HTTP commands to the robot. The system is also designed to run headlessly on a Raspberry Pi.

## Dependencies

```bash
pip install opencv-contrib-python numpy pillow mediapipe
# Optional: pyinstaller (for building executables)
# Optional: psutil (for RPi benchmarks)
# Optional: scikit-learn (for gesture ML comparison experiments)
```

`opencv-contrib-python` is required (not plain `opencv-python`) — it provides `cv2.FaceDetectorYN` and `cv2.FaceRecognizerSF`.

ONNX models are auto-downloaded from GitHub on first run to `models/`. A `bark.wav` file (optional) can be placed at the project root for the bark sound.

---

## Folder Structure

```
Revo_Robot_AI/
│
├── src/                            # Core Python source files
│   ├── face_embedding.py           # CLI pipeline: capture / build / recognize / enroll
│   ├── face_control_center.py      # Tkinter GUI desktop app (full system)
│   ├── revo_pi.py                  # Optimized headless runtime for Raspberry Pi
│   └── capture_faces.py            # Legacy simple capture tool (no quality checks)
│
├── experiments/                    # IEEE paper experiment scripts
│   ├── utils.py                    # Shared: logging, paths, metrics, plot style
│   ├── eval_face_recognition.py    # Phase 2 — gate configs, lighting ablation, LBPH
│   ├── sweep_threshold.py          # Phase 2.4 — ROC curve, EER, margin sensitivity
│   ├── sweep_voting.py             # Phase 3 — voting window heatmaps
│   ├── eval_gesture.py             # Phase 4 — rule-based + ML gesture comparison
│   ├── bench_rpi.py                # Phase 5 — incremental RPi optimization benchmarks
│   ├── latency_measure.py          # Phase 6 — per-component latency breakdown
│   ├── mock_robot_server.py        # Phase 6 helper — stdlib HTTP server
│   ├── collect_gesture_dataset.py  # Phase 1 helper — guided camera dataset collection
│   └── run_all.py                  # Master runner → generates results/SUMMARY.md
│
├── models/                         # ONNX model files (auto-downloaded on first run)
│   ├── face_detection_yunet_2023mar.onnx
│   ├── face_recognition_sface_2021dec.onnx
│   └── hand_landmarker.task
│
├── data/                           # All runtime data
│   ├── known_faces/                # Enrolled person face images
│   │   └── Harshhini/              # 13 enrolled images
│   └── face_db.npz                 # Built embedding database (embeddings + centroids)
│
├── results/                        # Experiment outputs (CSVs, PNGs, logs)
│   ├── logs/                       # Per-run timestamped log files
│   ├── phase2/                     # Face recognition accuracy results
│   ├── phase3/                     # Voting window analysis results
│   ├── phase4/                     # Gesture classification results
│   ├── phase5/                     # RPi benchmark results
│   ├── phase6/                     # Latency measurement results
│   └── SUMMARY.md                  # Auto-generated paper summary (after run_all.py)
│
├── docs/                           # All documentation
│   ├── README.md                   # Project overview
│   ├── USAGE.md                    # Quick-start guide
│   ├── GESTURE_CHEAT_SHEET.md      # Hand sign → command reference
│   └── RESEARCH.md                 # Full IEEE paper research plan (all 8 phases)
│
├── build/                          # PyInstaller specs
│   ├── RevoFaceControl.spec        # Full build with mediapipe bundled
│   └── RevoFaceControl_v2.spec     # Minimal build spec
│
└── CLAUDE.md                       # This file (project instructions for Claude Code)
```

---

## Running the Scripts

All scripts run from the **project root** (`Revo_Robot_AI/`).

**GUI control center (main application):**
```bash
python src/face_control_center.py
```

**CLI face enrollment pipeline:**
```bash
# Step 1: Capture face images for a person
python src/face_embedding.py capture --name Alice --samples 25

# Step 2: Build the embedding database
python src/face_embedding.py build

# Step 3: Run real-time recognition
python src/face_embedding.py recognize

# Or do capture + build in one step:
python src/face_embedding.py enroll --name Alice --samples 25
```

**Raspberry Pi headless runtime:**
```bash
python src/revo_pi.py
python src/revo_pi.py --iot-url http://192.168.x.x:8080/cmd
python src/revo_pi.py --enroll Alice --samples 25
python src/revo_pi.py --save-config        # write revo_config.json
```

**Simple face capture (legacy tool):**
```bash
python src/capture_faces.py
```

**Build standalone executables (run from project root):**
```bash
pyinstaller build/RevoFaceControl.spec       # with mediapipe bundled
pyinstaller build/RevoFaceControl_v2.spec    # minimal spec
```

---

## Running Experiments (IEEE Paper)

```bash
# Run all experiments sequentially, produce results/SUMMARY.md
python experiments/run_all.py

# Run individual experiments
python experiments/eval_face_recognition.py     # Phase 2 (auto-selects demo/full mode)
python experiments/sweep_threshold.py           # Phase 2.4 ROC curve
python experiments/sweep_voting.py              # Phase 3 voting heatmaps
python experiments/eval_gesture.py              # Phase 4 gesture accuracy
python experiments/bench_rpi.py --camera -1     # Phase 5 RPi benchmarks (synthetic)
python experiments/latency_measure.py           # Phase 6 latency breakdown

# Phase 1: Collect gesture dataset (needs camera)
python experiments/collect_gesture_dataset.py --name Alice --samples-per-gesture 50

# Phase 6: Start mock server, then run latency_measure
python experiments/mock_robot_server.py &
python experiments/latency_measure.py --server-url http://localhost:8080/cmd
```

All results (CSVs + PNGs) go to `results/phase<N>/`. Logs go to `results/logs/`.

---

## Architecture

### Core Pipeline

1. `data/known_faces/<PersonName>/` → face images on disk
2. `src/face_embedding.py build` → `data/face_db.npz` (embeddings + per-person centroids)
3. At runtime: camera frame → YuNet detector → SFace recognizer → cosine similarity against `face_db.npz` → temporal vote smoothing (6-frame history, 4-vote threshold) → identity authorized
4. Authorized identity → MediaPipe hand gesture → HTTP POST to robot dog

### Identity Matching Logic (`match_identity` in `src/face_embedding.py`)

Two-gate approach:
1. Best sample cosine similarity must exceed `--threshold` (default 0.42) AND beat second-best by `--margin` (default 0.06)
2. Centroid similarity must exceed `--centroid-threshold` (default 0.40) AND match the same identity

### Gesture Commands

Defined in `GESTURE_COMMANDS` dict in `src/face_control_center.py` and `src/revo_pi.py`. See `docs/GESTURE_CHEAT_SHEET.md` for the hand-sign-to-command mapping. Gestures only fire when a face is authorized, palm faces camera, and the sign is held stable (~0.4–0.8s).

### Robot Communication

Commands are sent as HTTP requests with JSON payload: `{person, command, source, timestamp}`. The `robot_action()` function in `src/face_embedding.py` is a stub — replace with actual robot API calls.

### RPi Optimizations (`src/revo_pi.py`)

| Optimization | Description |
|---|---|
| Dual-resolution | Detect at 320×240, recognize on full-res crop |
| Face-position cache | Skip SFace if IOU ≥ 0.72 and within 1.5s |
| Adaptive frame skip | IDLE=5 / DETECTING=2 / AUTHORIZED=1 |
| Motion gate | Skip inference if <0.8% pixels changed |
| Producer-consumer threads | Capture thread never blocks inference |
| Gesture ROI | MediaPipe runs only on face-adjacent region |

### Key Path Constants

| Variable | Value | File |
|---|---|---|
| `KNOWN_FACES_DIR` | `data/known_faces/` | `src/face_embedding.py`, `src/revo_pi.py` |
| `DB_FILE` | `data/face_db.npz` | `src/face_embedding.py`, `src/revo_pi.py` |
| `MODELS_DIR` | `models/` | all src files |
| `PROJECT_ROOT` | resolved from `__file__` | `experiments/utils.py` |

---

## What Was Built (Session Log)

### Folder Reorganization
- Moved all Python source files → `src/`
- Moved `known_faces/` and `face_db.npz` → `data/`
- Moved all `.md` docs (except CLAUDE.md) → `docs/`
- Moved PyInstaller `.spec` files → `build/`
- Updated all path constants in `src/face_embedding.py`, `src/revo_pi.py`, `src/face_control_center.py`, `src/capture_faces.py`, `experiments/utils.py`, and both `.spec` files to reflect new layout

### IEEE Paper Research Plan
- `docs/RESEARCH.md` — full 8-phase research plan with all comparison tables, metric definitions, execution order, parameter reference, and paper structure

### Experiment Scripts (`experiments/`)
- `utils.py` — shared utilities: logging, paths, metric computation, matplotlib style
- `eval_face_recognition.py` — Phase 2: gate config comparison, lighting ablation, LBPH baseline
- `sweep_threshold.py` — Phase 2.4: ROC curve, EER detection, margin sensitivity sweep
- `sweep_voting.py` — Phase 3: (H, V) heatmaps for latency and impostor security
- `eval_gesture.py` — Phase 4: rule-based + SVM/RF/KNN, confusion matrix, cross-subject test
- `bench_rpi.py` — Phase 5: 6-config cumulative benchmark (FPS / CPU / RAM / latency)
- `latency_measure.py` — Phase 6: per-component timing + gesture vote sweep
- `mock_robot_server.py` — stdlib HTTP server for latency testing (no Flask)
- `collect_gesture_dataset.py` — Phase 1: guided camera UI to build gesture dataset
- `run_all.py` — master runner with isolated subprocess execution + auto-generates `results/SUMMARY.md`
