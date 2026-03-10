# REVO Robot Dog — IEEE Paper Research Plan
**End-to-End Implementation Guide (Self-Executable)**

---

## Codebase Context (Read This First)

**Project root:** `/Users/yrevash/Revo_Robot_AI/`

**Key files:**
- `face_embedding.py` — CLI pipeline: `capture / build / recognize / enroll` subcommands
- `face_control_center.py` — Tkinter GUI desktop app (full system)
- `revo_pi.py` — Raspberry Pi optimized headless runtime
- `capture_faces.py` — legacy simple capture (no quality checks)
- `known_faces/Harshhini/` — 13 enrolled face images
- `face_db.npz` — built embedding database
- `models/` — YuNet ONNX (detector) + SFace ONNX (recognizer) + MediaPipe hand model

**Dependencies:** `opencv-contrib-python numpy pillow mediapipe`

**Core pipeline:**
1. Frame → CLAHE/gamma normalization (optional, toggle with `--no-light-normalize`)
2. YuNet ONNX face detector → face bounding box + landmarks
3. SFace ONNX → alignCrop → 128-dim L2-normalized embedding
4. Two-gate matching: cosine similarity threshold (0.42) + margin (0.06) vs second-best + centroid gate (0.40)
5. Temporal voting: 6-frame history, 4-vote threshold → authorized identity
6. MediaPipe Hands Lite → 21 landmarks → rule-based gesture classifier → HTTP POST to robot

**Two-gate match logic** (`face_embedding.py:match_identity` lines 451–490):
- Gate 1: `best_score >= threshold AND (best_score - second_best) >= margin`
- Gate 2: `centroid_score >= centroid_threshold AND centroid_name == candidate_name`
- Returns `"Unknown"` if either gate fails

**RPi optimizations in `revo_pi.py`:**
- Dual-resolution: detect at 320×240, recognize on full-res crop
- Face-position cache: skip SFace when IOU of face bbox >= 0.72 and within 1.5s
- Adaptive frame skip: IDLE=5, DETECTING=2, AUTHORIZED=1
- Motion gate: 160×120 frame diff, skip if fewer than 0.8% pixels changed
- Producer-consumer threads: capture thread never blocks on inference
- Gesture ROI: MediaPipe runs only on face-adjacent region

**Gesture commands** (11 classes):
`FORWARD, BACKWARD, LEFT, RIGHT, BARK, STAND, TAIL_WAG, WALK, SIT, STOP, GREET`

---

## Paper Title (Draft)
**"REVO: A Real-Time Face-Authorized Gesture Control System for Robot Dogs on Embedded Hardware"**

---

## Paper Structure

```
1. Abstract
2. Introduction
3. Related Work
4. System Architecture
5. Methodology
   5.1 Two-Gate Face Identity Matching
   5.2 Temporal Voting for Stable Authorization
   5.3 Lighting Normalization
   5.4 Gesture Classification
   5.5 Raspberry Pi Optimizations
6. Experiments (Phases 1–5 below)
7. Results & Discussion
8. Limitations
9. Conclusion
10. References
```

---

## PHASE 1 — Dataset & Ground Truth Setup
**Goal:** Build a controlled test dataset for all downstream experiments.
**Prerequisite for:** All other phases.

### 1.1 Face Dataset Collection

**What to collect:**
- Minimum 3 subjects (enrolled persons)
- Minimum 2 impostors (never enrolled)
- Per enrolled subject: 25 training images + 100 test images
- Per impostor: 100 test images

**Lighting conditions (must capture separately):**
| Condition | Label | Description |
|---|---|---|
| Normal | `L0` | Room light, face-on |
| Dim | `L1` | Reduce room light / evening |
| Backlit | `L2` | Light source behind subject |
| Side-lit | `L3` | Single side lamp |
| Overhead | `L4` | Overhead only, shadows under eyes |

**How to collect:**
```bash
# For each subject and each lighting condition:
python face_embedding.py capture --name Alice_L0 --samples 25
python face_embedding.py capture --name Alice_L1 --samples 25
# ... repeat for all conditions

# Build DB from only the Normal (L0) training set:
# Move all L1-L4 captures to a separate test_faces/ folder first
python face_embedding.py build
```

**Directory layout to create:**
```
test_faces/
  enrolled/
    Alice/
      L0/ (100 images)
      L1/ (100 images)
      L2/ (100 images)
      L3/ (100 images)
      L4/ (100 images)
  impostors/
    Impostor1/
      L0/ (100 images)
    Impostor2/
      L0/ (100 images)
```

**Ground truth CSV format:**
```
image_path, true_identity, lighting_condition, is_enrolled
test_faces/enrolled/Alice/L0/001.jpg, Alice, L0, True
test_faces/impostors/Impostor1/L0/001.jpg, Unknown, L0, False
```
Save as: `test_faces/ground_truth.csv`

### 1.2 Gesture Dataset Collection

**What to collect:**
- Minimum 3 subjects performing all 11 gestures
- 50 samples per gesture per subject = 550 samples per subject
- Capture as short video clips or individual frames with labels

**How to capture gesture frames:**
```bash
# Write a simple script (see Phase 1 task) that:
# - Opens webcam
# - Displays gesture name on screen
# - Records 50 frames when user presses SPACE
# - Saves frames to gesture_dataset/<subject>/<gesture>/001.jpg etc.
```

**Gesture dataset layout:**
```
gesture_dataset/
  Alice/
    FORWARD/ (50 images)
    BACKWARD/ (50 images)
    LEFT/ (50 images)
    ... (all 11 classes)
  Bob/
    ...
```

**Ground truth CSV:**
```
image_path, subject, gesture_label
gesture_dataset/Alice/FORWARD/001.jpg, Alice, FORWARD
```
Save as: `gesture_dataset/ground_truth.csv`

---

## PHASE 2 — Face Recognition Accuracy Experiments
**Goal:** Prove the two-gate matching approach outperforms single-threshold and alternative methods.
**Depends on:** Phase 1 face dataset.

### 2.1 Evaluate Your System (Baseline)

**Script to write:** `experiments/eval_face_recognition.py`

**What it does:**
1. Load `face_db.npz` (built from L0 training images only)
2. For each test image in `test_faces/`:
   - Run YuNet detection
   - Run SFace embedding extraction
   - Run `match_identity()` with default params
   - Record: predicted name, true name, best_score, centroid_score
3. Compute metrics:
   - True Accept Rate (TAR) = correctly identified enrolled
   - False Accept Rate (FAR) = impostors accepted
   - False Reject Rate (FRR) = enrolled rejected
   - Equal Error Rate (EER) = threshold where FAR == FRR
   - ROC curve data

**Key function to call** (already exists in `face_embedding.py`):
```python
from face_embedding import load_db, match_identity, create_detector, create_recognizer
from face_embedding import detect_faces, embedding_from_face, normalize_lighting

db_embeddings, db_names, centroids, centroid_names = load_db(Path("face_db.npz"))
# For each test image:
name, sample_score, centroid_score = match_identity(
    emb, db_embeddings, db_names, centroids, centroid_names,
    threshold=0.42, margin=0.06, centroid_threshold=0.40
)
```

**Output CSV:** `results/phase2_recognition_results.csv`
```
image_path, true_name, predicted_name, sample_score, centroid_score, lighting, correct
```

### 2.2 Ablation: Single-Gate vs Two-Gate

**Variants to test:**
| Variant | threshold | margin | centroid_threshold | Description |
|---|---|---|---|---|
| A | 0.42 | 0.00 | 0.00 | Sample score only, no margin, no centroid |
| B | 0.42 | 0.06 | 0.00 | Sample + margin, no centroid |
| C | 0.42 | 0.06 | 0.40 | Full two-gate (your system) |
| D | 0.35 | 0.06 | 0.40 | Lower threshold |
| E | 0.50 | 0.06 | 0.40 | Higher threshold |

**Implement by:** modifying `match_identity()` call parameters in the eval script.
For Variant A: pass `margin=0.0, centroid_threshold=0.0`

**Table to produce:**
| Variant | TAR | FAR | FRR | EER |
|---|---|---|---|---|
| A (no margin, no centroid) | | | | |
| B (margin only) | | | | |
| C (full two-gate, yours) | | | | |

### 2.3 Lighting Robustness Ablation

**Test:** Run eval on each lighting condition separately, with and without `normalize_lighting()`.

**Variants:**
- Without normalization: pass raw frame to detector
- With normalization: apply `normalize_lighting(frame)` before detection

**Table to produce:**
| Lighting | Without Norm (Acc%) | With Norm (Acc%) | Delta |
|---|---|---|---|
| L0 Normal | | | |
| L1 Dim | | | |
| L2 Backlit | | | |
| L3 Side-lit | | | |
| L4 Overhead | | | |

**`normalize_lighting` function** is at `face_embedding.py:138`.
Toggle by setting `args.light_normalize = True/False` or calling directly.

### 2.4 Threshold Sensitivity Analysis

**Sweep:** threshold from 0.25 to 0.65 in steps of 0.05
For each threshold value: compute FAR and FRR → plot ROC curve.

**Script:** Loop over threshold values, call `match_identity()` for each test sample, accumulate TP/FP/TN/FN.

**Output:** ROC curve plot saved as `results/phase2_roc_curve.png`

### 2.5 Baseline Comparisons

**Compare against LBPH (already in OpenCV):**
```python
# Train
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(face_images, labels)
# Test
label, confidence = recognizer.predict(face_roi)
```

**Compare against dlib (if installed: `pip install dlib`):**
```python
import dlib
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
# 128-dim descriptor, compare with euclidean distance threshold 0.6
```

**Comparison table:**
| Method | TAR | FAR | EER | Avg Latency (ms/frame) |
|---|---|---|---|---|
| LBPH (OpenCV classic) | | | | |
| dlib HOG + ResNet | | | | |
| SFace single-gate (no centroid) | | | | |
| **SFace two-gate (REVO, ours)** | | | | |

---

## PHASE 3 — Temporal Voting Analysis
**Goal:** Show the tradeoff between security (FAR) and responsiveness (latency) as voting parameters change.
**Depends on:** Phase 1 face dataset, Phase 2 eval script.

### 3.1 Voting Window Sweep

**Parameters to sweep:**
- `history_len` (H): 3, 4, 5, 6, 8, 10 (frames kept in deque)
- `stable_count` (V): 2, 3, 4, 5 (votes needed to authorize)

**At 30 FPS:** authorization latency = `history_len / 30` seconds minimum

**For each (H, V) combination:**
- Simulate frame sequence: feed recognition results frame-by-frame
- Record: frames until authorization (TAR cases) and whether impostors got authorized (FAR cases)

**How to simulate** (no camera needed):
```python
from collections import deque, Counter

def simulate_voting(frame_results, history_len, stable_count):
    # frame_results = list of sets of recognized names per frame
    history = deque(maxlen=history_len)
    authorized_at_frame = None
    for i, names_in_frame in enumerate(frame_results):
        history.append(names_in_frame)
        if len(history) == history_len:
            vote = Counter()
            for s in history:
                vote.update(s)
            if any(v >= stable_count for v in vote.values()):
                authorized_at_frame = i
                break
    return authorized_at_frame
```

**Tables to produce:**
- Heatmap: (H, V) → mean authorization latency (frames)
- Heatmap: (H, V) → FAR on impostor test set

**Output:** `results/phase3_voting_heatmap.png`

### 3.2 Frame Skip Effect on Latency

**In `revo_pi.py`**, adaptive frame skip: IDLE=5, DETECTING=2, AUTHORIZED=1.

**Test:** With fixed (H=6, V=4), vary frame skip from 1 to 6.
Record: wall-clock time from face appearing to authorization.

**Table:**
| Frame Skip | Wall-clock auth latency (s) | CPU % saved vs skip=1 |
|---|---|---|
| 1 | | |
| 2 | | |
| 3 | | |
| 4 | | |
| 5 | | |
| 6 | | |

---

## PHASE 4 — Gesture Classification Accuracy
**Goal:** Prove rule-based landmark classifier is sufficient vs ML alternatives.
**Depends on:** Phase 1 gesture dataset.

### 4.1 Evaluate Rule-Based Classifier (Current System)

**Script to write:** `experiments/eval_gesture.py`

**What it does:**
1. For each image in `gesture_dataset/`:
   - Run MediaPipe Hands on image
   - If hand detected, run `_classify(lmk, hand_label)` (same logic as `revo_pi.py:457`)
   - Record predicted label vs ground truth label
2. Compute per-class precision, recall, F1
3. Build 11×11 confusion matrix

**Key: copy the `_classify` method from `revo_pi.py` (lines 457–497) into the eval script** — it is purely a function of landmarks, no camera needed.

**Output:**
- `results/phase4_gesture_confusion_matrix.png`
- `results/phase4_gesture_per_class_metrics.csv`

**Expected confusion matrix format:**
```
             FORWARD BACKWARD LEFT RIGHT BARK STAND TAIL_WAG WALK SIT STOP GREET
FORWARD      48      0        0    0     0    0     0        2    0   0    0
BACKWARD     0       47       0    0     0    0     0        0    0   3    0
...
```

### 4.2 ML Baseline Comparisons

**Feature vector:** flatten 21 landmarks × 2 (x,y) = 42 features per sample (use normalized coords).

**Classifiers to compare:**
```python
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# Build feature matrix X (N, 42) and labels y (N,)
# For each image: run MediaPipe, extract lmk x,y coords, flatten

svm = SVC(kernel='rbf', C=10, gamma='scale')
rf  = RandomForestClassifier(n_estimators=100)
knn = KNeighborsClassifier(n_neighbors=5)
```

**Comparison table:**
| Method | Accuracy | F1 (macro) | Latency (ms/frame) |
|---|---|---|---|
| Rule-based (REVO, ours) | | | |
| KNN (k=5) on landmarks | | | |
| SVM (RBF) on landmarks | | | |
| Random Forest (100 trees) | | | |

**Why rule-based wins here:**
- Zero training data required
- Generalizes across users without retraining
- Interpretable (can debug wrong gestures)
- Faster (no sklearn inference overhead)

### 4.3 Cross-Subject Generalization

**Test:** Train SVM/RF on Subject A, test on Subject B.
Compare: rule-based (no training) vs ML (requires per-subject training).

This is a key result: rule-based classifier is **training-free** and still competitive.

| Method | Same-subject accuracy | Cross-subject accuracy |
|---|---|---|
| Rule-based | | |
| SVM (trained on A, tested on B) | | |
| RF (trained on A, tested on B) | | |

---

## PHASE 5 — Raspberry Pi Performance Benchmarks
**Goal:** Quantify the cumulative speedup from each optimization in `revo_pi.py`.
**This is the strongest systems/embedded contribution.**

### 5.1 Incremental Ablation Setup

**Run on actual Raspberry Pi 4 (or record on desktop as surrogate).**

**Measure for each configuration:**
- FPS (frames processed per second)
- CPU utilization (%)
- RAM usage (MB) — `psutil.Process().memory_info().rss`
- Wall-clock authorization latency (ms, from face appearing to `AUTHORIZED` log)
- Inference time per frame (ms) — `time.perf_counter()` around `_process()`

**Script to write:** `experiments/bench_revo_pi.py`

Modify `revo_pi.py` temporarily to expose each optimization as a toggle:

```python
# Config flags to toggle for benchmarking:
USE_DUAL_RES    = True   # detect at 320x240 vs full-res
USE_FACE_CACHE  = True   # IOU-based skip
USE_MOTION_GATE = True   # frame diff gate
USE_ADAPTIVE_SKIP = True # IDLE=5/DETECTING=2/AUTHORIZED=1
USE_GESTURE_ROI = True   # gesture on face-adjacent ROI only
```

### 5.2 Benchmark Configurations

| Config | Dual-Res | Face Cache | Motion Gate | Adaptive Skip | Gesture ROI |
|---|---|---|---|---|---|
| 0 — Naive baseline | ✗ | ✗ | ✗ | skip=1 fixed | ✗ |
| 1 — + Dual-Res | ✓ | ✗ | ✗ | skip=1 fixed | ✗ |
| 2 — + Face Cache | ✓ | ✓ | ✗ | skip=1 fixed | ✗ |
| 3 — + Motion Gate | ✓ | ✓ | ✓ | skip=1 fixed | ✗ |
| 4 — + Adaptive Skip | ✓ | ✓ | ✓ | adaptive | ✗ |
| 5 — + Gesture ROI (full REVO) | ✓ | ✓ | ✓ | adaptive | ✓ |

**Expected results table:**
| Config | FPS | CPU% | RAM MB | Auth Latency ms |
|---|---|---|---|---|
| 0 Naive | ~3–5 | ~95 | ~320 | ~2000 |
| 1 +Dual-Res | ~8–10 | ~75 | ~310 | ~1800 |
| 2 +Face Cache | ~12–15 | ~60 | ~310 | ~1600 |
| 3 +Motion Gate | ~15–18 | ~50 | ~310 | ~1600 |
| 4 +Adaptive Skip | ~20–24 | ~45 | ~310 | ~1400 |
| 5 Full REVO | ~25–30 | ~40 | ~300 | ~1200 |

*(Fill in real numbers when run on RPi)*

### 5.3 Threading Benefit Measurement

**revo_pi.py uses producer-consumer thread model.**
Compare:
- Single-thread: capture + inference in one loop
- Dual-thread: `CameraCapture` thread + inference thread (current)

Measure: dropped frames, inference lag, effective FPS.

### 5.4 Memory Profile

```python
import tracemalloc
tracemalloc.start()
# ... run inference for 100 frames ...
snapshot = tracemalloc.take_snapshot()
# Report top memory consumers
```

---

## PHASE 6 — End-to-End System Latency
**Goal:** Measure total time from gesture shown to robot command fired.

### 6.1 Latency Components

```
t0 = gesture onset (hand sign held)
t1 = gesture detected by MediaPipe
t2 = gesture votes reach threshold (gesture_votes=2)
t3 = CommandDispatcher.send() called
t4 = HTTP POST received by robot endpoint (if testing with mock server)

Report: t1-t0 (detection), t2-t1 (voting), t3-t2 (dispatch), t4-t3 (network)
```

**Mock HTTP server** to measure t4:
```python
# Simple Flask server to timestamp received POSTs
from flask import Flask, request
import time
app = Flask(__name__)
@app.route("/cmd", methods=["POST"])
def cmd():
    print(f"Received at {time.time()}: {request.json}")
    return "OK"
```

### 6.2 Gesture Voting Latency Tradeoff

| gesture_votes | gesture_history | Mean latency (ms) | Gesture accuracy |
|---|---|---|---|
| 1 | 2 | fastest | lowest |
| 2 | 4 | — | — |
| 3 | 6 | — | — |
| 4 | 8 | slowest | highest |

**`gesture_every_n`** in Config (default=2) also affects this: gesture runs every N inference frames. Sweep N=1,2,3.

---

## PHASE 7 — Security / Anti-Spoofing Analysis
**Goal:** Demonstrate known security limitations honestly (for the Limitations section) and show what the two-gate adds.

### 7.1 Attack Scenarios to Test

| Attack | Method | Expected result |
|---|---|---|
| Photo attack | Print enrolled person's photo, show to camera | Should accept (no liveness — state as limitation) |
| Similar-face impostor | Person resembling enrolled subject | Two-gate reduces FAR vs single-gate |
| Unknown person | Completely different person | Should reject |
| Partial occlusion | Enrolled person with mask/sunglasses | Should reject (detection fails) |

**This section is honest: your system does NOT have liveness detection.**
But you show: two-gate significantly reduces FAR vs single-threshold (Phase 2 data proves this).

### 7.2 FAR Comparison (Security)

Use Phase 2 results to extract FAR per gate configuration.
Show: full two-gate has lowest FAR = highest security without sacrificing TAR.

---

## PHASE 8 — Paper Writing Plan

### 8.1 Figures to Generate

| Figure | Phase | File |
|---|---|---|
| System architecture diagram | — | draw manually or use matplotlib |
| ROC curve (single-gate vs two-gate) | Phase 2 | `results/phase2_roc_curve.png` |
| Lighting robustness bar chart | Phase 2 | `results/phase2_lighting_accuracy.png` |
| Voting window heatmap | Phase 3 | `results/phase3_voting_heatmap.png` |
| Gesture confusion matrix | Phase 4 | `results/phase4_gesture_confusion_matrix.png` |
| RPi incremental speedup bar chart | Phase 5 | `results/phase5_rpi_benchmarks.png` |
| End-to-end latency breakdown | Phase 6 | `results/phase6_latency_breakdown.png` |

### 8.2 Key Claims to Support with Data

1. **Two-gate matching reduces FAR by X% vs single threshold** → Phase 2.2 table
2. **CLAHE+gamma normalization improves accuracy by X% under poor lighting** → Phase 2.3 table
3. **Incremental optimizations achieve Xx speedup on RPi vs naive** → Phase 5.2 table
4. **Rule-based gesture classifier achieves X% accuracy, competitive with trained SVM** → Phase 4.2 table
5. **End-to-end gesture latency under Xms at 30FPS** → Phase 6.2

### 8.3 Related Work to Cite

- **YuNet:** Shiqi Yu et al., "YuNet: A Tiny Millisecond-level Face Detector" (2022)
- **SFace:** Yaoyao Zhong et al., "SFace: Sigmoid-Constrained Hypersphere Loss for Face Recognition" (2021)
- **MediaPipe Hands:** Zhang et al., "MediaPipe Hands: On-device Real-time Hand Tracking" (2020)
- **ArcFace:** Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition" (CVPR 2019)
- **FaceNet:** Schroff et al., "FaceNet: A Unified Embedding for Face Recognition and Clustering" (CVPR 2015)
- **LBPH:** Ahonen et al., "Face Description with Local Binary Patterns" (TPAMI 2006)
- **Robot HRI gesture:** Rautaray & Agrawal, "Vision based hand gesture recognition for human computer interaction" (2015)
- **Embedded face recognition:** Kortli et al., "Face Recognition Systems: A Survey" (Sensors 2020)

---

## Experiment Scripts — File Plan

```
experiments/
  eval_face_recognition.py     # Phase 2 — runs match_identity on test_faces/
  eval_gesture.py              # Phase 4 — runs classifier on gesture_dataset/
  bench_revo_pi.py             # Phase 5 — incremental optimization benchmarks
  sweep_voting.py              # Phase 3 — voting window heatmap
  sweep_threshold.py           # Phase 2.4 — ROC curve generation
  latency_measure.py           # Phase 6 — end-to-end latency
  plot_results.py              # All phases — generate paper figures
  mock_robot_server.py         # Phase 6 — Flask HTTP receiver for latency test

results/
  phase2_recognition_results.csv
  phase2_roc_curve.png
  phase2_lighting_accuracy.png
  phase3_voting_heatmap.png
  phase4_gesture_confusion_matrix.png
  phase4_gesture_per_class_metrics.csv
  phase5_rpi_benchmarks.csv
  phase5_rpi_benchmarks.png
  phase6_latency_breakdown.png

test_faces/
  ground_truth.csv
  enrolled/
  impostors/

gesture_dataset/
  ground_truth.csv
```

---

## Execution Order

```
Phase 1  →  Phase 2  →  Phase 3
                     ↘  Phase 4
                     ↘  Phase 5
                     ↘  Phase 6
                     ↘  Phase 7
All phases → Phase 8 (paper writing)
```

**Critical path:** Phase 1 (data collection) must complete before anything else.
Phases 2–7 are parallelizable after Phase 1.

---

## Quick Reference: Parameters & Defaults

| Parameter | Default | File | Where Used |
|---|---|---|---|
| `threshold` | 0.42 | `face_embedding.py` | `match_identity()` Gate 1 |
| `margin` | 0.06 | `face_embedding.py` | `match_identity()` Gate 1 |
| `centroid_threshold` | 0.40 | `face_embedding.py` | `match_identity()` Gate 2 |
| `history` | 6 | `face_embedding.py` | temporal voting deque |
| `stable_count` | 4 | `face_embedding.py` | votes needed to authorize |
| `cooldown` | 3.0s | `face_embedding.py` | re-trigger gap |
| `track_timeout` | 2.0s | `face_embedding.py` | authorized state timeout |
| `face_cache_iou` | 0.72 | `revo_pi.py` | skip SFace if face barely moved |
| `face_cache_timeout` | 1.5s | `revo_pi.py` | force re-recognition after |
| `frame_skip_idle` | 5 | `revo_pi.py` | process every 5th frame in IDLE |
| `frame_skip_detecting` | 2 | `revo_pi.py` | every 2nd in DETECTING |
| `frame_skip_authorized` | 1 | `revo_pi.py` | every frame in AUTHORIZED |
| `gesture_every_n` | 2 | `revo_pi.py` | gesture inference every N frames |
| `gesture_votes` | 2 | `revo_pi.py` | votes to confirm gesture |
| `gesture_cooldown` | 1.2s | `revo_pi.py` | same gesture re-trigger gap |
| `motion_gate_fraction` | 0.008 | `revo_pi.py` | 0.8% pixels must change |
| `det_score` | 0.88–0.9 | both | YuNet face detection confidence |
| `min_face_area_ratio` | 0.04–0.08 | both | ignore very small faces |
| `hand_max_dim` | 320 | `revo_pi.py` | max edge for MediaPipe gesture ROI |

---

## Limitations to State Honestly in Paper

1. **No liveness detection** — photo/video replay attacks possible
2. **Single camera** — occlusion or extreme angles cause false rejects
3. **Lighting normalization not always helpful** — very dark scenes still fail detection
4. **Gesture set is fixed** — no mechanism to add custom gestures without code changes
5. **HTTP-only robot interface** — no acknowledgment/retry on robot command failure
6. **Single enrolled person demo** — FAR tested with limited impostor pool
7. **No encryption** — HTTP commands are plaintext (not relevant for research but note for deployment)

---

## Status Tracker

| Phase | Status | Notes |
|---|---|---|
| Phase 1 — Data Collection | ⬜ Not started | Need 3+ subjects + 2 impostors |
| Phase 2 — Face Recognition Eval | ⬜ Not started | Depends on Phase 1 |
| Phase 3 — Voting Analysis | ⬜ Not started | Can simulate with Phase 2 results |
| Phase 4 — Gesture Eval | ⬜ Not started | Depends on Phase 1 gesture dataset |
| Phase 5 — RPi Benchmarks | ⬜ Not started | Needs RPi hardware |
| Phase 6 — Latency Measurement | ⬜ Not started | Needs mock server setup |
| Phase 7 — Security Analysis | ⬜ Not started | Depends on Phase 2 |
| Phase 8 — Paper Writing | ⬜ Not started | Depends on all phases |
