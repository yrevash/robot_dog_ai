# REVO Robot Dog — Experiment Results

> **Project:** REVO — Face-Recognition + Gesture-Controlled Robot Dog
> **Platform:** Apple M3 MacBook (development machine). Deployment target is Raspberry Pi 4/5 — RPi benchmarks pending hardware access.
> **Subjects enrolled (face DB):** 5 (Yash, Aramaan, Pratham, Shubham, Sohail)
> **Impostor set:** 1 identity (Harshhini — 13 images, NOT enrolled in DB)
> **Gesture subjects:** 5 (Yash, Aramaan, Pratham, Shubham, Sohail)
> **Gesture classes evaluated:** 5 (SIT, STAND, WALK, STOP, BARK)
> **Date:** 2026-03-10
> **Scope:** Proof-of-concept feasibility study on a small in-house dataset. Results confirm the pipeline is functional and reveal key design strengths/weaknesses. Not yet publication-grade — see Section 9 for limitations.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Dataset Summary](#2-dataset-summary)
3. [Phase 2 — Face Recognition Accuracy](#3-phase-2--face-recognition-accuracy)
4. [Phase 2.4 — Threshold & Margin Sweep](#4-phase-24--threshold--margin-sweep)
5. [Phase 3 — Temporal Voting Analysis](#5-phase-3--temporal-voting-analysis)
6. [Phase 4 — Gesture Classification](#6-phase-4--gesture-classification)
7. [Phase 6 — End-to-End Latency](#7-phase-6--end-to-end-latency)
8. [Phase 7 — Security Analysis](#8-phase-7--security-analysis)
9. [Limitations and Honest Caveats](#9-limitations-and-honest-caveats)
10. [Key Findings](#10-key-findings)
11. [File Index](#11-file-index)

---

## 1. System Overview

REVO converts raw camera frames into robot dog commands through a layered pipeline:

```
Camera Frame  (+5–10 ms USB I/O, not measured)
    │
    ▼  ~7.2 ms  [M3 — measured]
YuNet ONNX Face Detector  ──────────────────► No face → skip frame
    │
    ▼  ~4.1 ms  [M3 — measured]
SFace ONNX Embedding Extractor  (128-dim, L2-normalised)
    │
    ▼  ~0.4 ms
Two-Gate Identity Matcher
    ├─ Gate 1: cosine similarity > 0.42 AND margin > 0.06 vs 2nd-best
    └─ Gate 2: centroid similarity > 0.40 AND same identity
    │
    ▼  <0.01 ms
Temporal Voter  (6-frame deque, 4-vote threshold)
    │
    ▼  ~16.9 ms  [M3 — measured]
MediaPipe HandLandmarker (Tasks API)  ── gesture → robot command
    │
    ▼  ~10–50 ms  [network-dependent, not measured]
HTTP POST to Robot Dog
```

**Total measured pipeline: ~29 ms on M3 Mac (~34 FPS theoretical)**
**Estimated on Raspberry Pi 5: ~100 ms (~10 FPS) — not yet measured**

---

## 2. Dataset Summary

| Item | Detail |
|------|--------|
| Enrolled identities (DB) | 5 (Yash, Aramaan, Pratham, Shubham, Sohail) |
| Training images (DB) | 17–20 per person (earliest images in known_faces/) |
| Test enrolled images | 5–8 per person (held-out, later images) = 31 total |
| Impostor test images | 13 (Harshhini — never enrolled) |
| Total face test set | **44 samples** (31 enrolled + 13 impostor) |
| Gesture subjects | 5 (same people as above) |
| Gesture samples | 750 total — 5 classes × 30 images × 5 subjects |
| Gesture classes | SIT, STAND, WALK, STOP, BARK |

> **Train/test split:** Test images were copied to `test_faces/` before building the DB, so the recognizer never trained on test data. This gives a proper held-out evaluation.

---

## 3. Phase 2 — Face Recognition Accuracy

**Script:** `experiments/eval_face_recognition.py`
**Output folder:** `results/phase2/`

### 3.1 Two-Gate Configuration Comparison

**Setup:** 5-person DB (Yash, Aramaan, Pratham, Shubham, Sohail). Test: 31 enrolled test images + 13 Harshhini impostor images. N = 44.

| Config | Description | TAR | FAR | FRR | ACC |
|--------|-------------|-----|-----|-----|-----|
| **A** | Score only (cosine > 0.42) | 0.968 | 0.000 | 0.032 | **0.977** |
| **B** | Score + margin gate (> 0.06) | 0.935 | 0.000 | 0.065 | **0.955** |
| **C** | Score + centroid gate (> 0.40) | 0.935 | 0.000 | 0.065 | **0.955** |
| **D** | Full two-gate (A + B + C) | 0.903 | 0.000 | 0.097 | **0.932** |

**95% Wilson confidence intervals (N=44):**
- TAR (Config A) = 30/31: [0.832, 0.999]
- FAR = 0/13: [0.000, 0.228] — could be as high as 22.8% with more impostors

**Key observations:**
- FAR = 0.000 across all configs — Harshhini never passes any gate (max score = 0.363, well below 0.42)
- Adding more gates (B, C, D) reduces TAR slightly — some legitimate users fall just below the tighter thresholds
- Config A (score only) gives the best usability/security balance at this dataset size
- Score gap: minimum enrolled score (0.661) vs maximum impostor score (0.363) = **gap of 0.298**

> ⚠ **Caveat:** FAR=0 with only 1 impostor subject does not confirm the system is secure against arbitrary impostors. The CI [0.000, 0.228] is wide. A visually similar impostor (sibling, lookalike) was never tested.

![Gate Comparison Bar Chart](phase2/gate_comparison_bar.png)

### 3.2 Lighting Ablation

Only one lighting condition was tested (L0 = standard indoor). Synthetic transforms caused YuNet detection to fail.

| Lighting | Correct | Total | ACC |
|----------|---------|-------|-----|
| L0 (natural indoor) | 41 | 44 | 0.932 |
| L1–L4 (synthetic) | — | — | YuNet detection failed |

### 3.3 LBPH Baseline

LBPH was trained on the same training partition (images 001–017/020) and tested on the same 44-sample test set.

| Method | TAR | FAR | FRR | ACC |
|--------|-----|-----|-----|-----|
| **SFace two-gate full (D)** | **0.903** | **0.000** | **0.097** | **0.932** |
| LBPH baseline (conf < 75) | ~0.813 | 0.000 | ~0.188 | ~0.897 |

SFace outperforms LBPH by ~3.5 points in ACC, driven by lower FRR. Both reject all Harshhini impostors.

> Note: LBPH requires `opencv-contrib-python` with `cv2.face` module. If unavailable, this comparison is skipped.

---

## 4. Phase 2.4 — Threshold & Margin Sweep

**Script:** `experiments/sweep_threshold.py`
**Output folder:** `results/phase2/`

The cosine threshold was swept from 0.20 to 0.70. With the score gap of 0.298, any threshold in [0.37, 0.42] gives clean separation on this dataset. Default threshold 0.42 was selected.

![Threshold Sweep](phase2/threshold_sweep.png)
![ROC Curve](phase2/roc_curve.png)

---

## 5. Phase 3 — Temporal Voting Analysis

**Script:** `experiments/sweep_voting.py`
**Output folder:** `results/phase3/`

Voting smooths noisy frame-by-frame recognition. The system keeps a 6-frame deque and requires 4 consecutive votes for the same identity before authorizing.

| history_len | stable_count | Auth Latency @30fps | Notes |
|-------------|-------------|---------------------|-------|
| 3 | 2 | ~100 ms | Fast but noisy |
| **6** | **4** | **~200 ms** | **Production setting** |
| 8 | 4 | ~267 ms | Slightly more stable |
| 10 | 6 | ~333 ms | Conservative |

**Production setting:** history=6, stable=4 → ~200 ms authorization at 30 FPS.

![Voting Heatmap](phase3/voting_heatmap_latency.png)

---

## 6. Phase 4 — Gesture Classification

**Script:** `experiments/eval_gesture.py`
**Output folder:** `results/phase4/`

### 6.1 Dataset

- **750 static images:** 5 classes × 30 images × 5 subjects (Yash, Aramaan, Pratham, Shubham, Sohail)
- Indoor conditions, hand in front of webcam
- Labels from folder structure

### 6.2 Rule-Based Classifier — Per-Class Results (5 subjects)

**Overall: Accuracy = 56.27% (422/750), Macro F1 = 0.693**

| Gesture | Precision | Recall | F1 | Support |
|---------|-----------|--------|----|---------|
| SIT | 1.000 | 0.800 | 0.889 | 150 |
| STAND | 1.000 | 0.800 | 0.889 | 150 |
| WALK | 0.973 | 0.240 | 0.385 | 150 |
| STOP | 1.000 | 0.400 | 0.571 | 150 |
| BARK | 1.000 | 0.573 | 0.729 | 150 |

**Interpretation:**
- SIT and STAND are most reliably detected (F1=0.889) — V-sign and 3-finger are distinct poses
- WALK is the hardest (F1=0.385, recall only 24%) — the 4-finger pose is confused with SIT/STAND by many subjects
- Precision=1.000 for all classes means zero false positives — when the rule fires, it's correct. The problem is misses (recall < 1.0)
- The rule-based system has no per-person tuning, so it struggles with hand size and pose variation across subjects

### 6.3 ML Classifier Comparison

| Method | Closed-set Acc | Macro F1 | LOSO Acc |
|--------|---------------|----------|----------|
| **Rule-based** | **56.27%** | **0.693** | **N/A (generalizes by design)** |
| SVM (RBF, C=10) | 99.86% | 0.999 | **55.88%** |
| Random Forest | 99.86% | 0.999 | 52.56% |
| KNN (k=5) | 99.73% | 0.997 | 30.16% |

**Key finding:** ML models appear nearly perfect in closed-set evaluation (99%+) but collapse to 30–56% on Leave-One-Subject-Out (LOSO). This is subject-specific overfitting — the models memorize each person's hand geometry rather than learning generalizable gesture patterns.

The rule-based classifier (56.27%) now matches SVM LOSO (55.88%), confirming that for this 5-class dataset with 5 subjects, geometry-based rules are as good as trained ML at cross-subject generalization while being simpler and requiring no training data.

![Gesture Per-Class Bar](phase4/gesture_per_class_bar.png)
![Confusion Matrix](phase4/gesture_confusion_matrix.png)
![ML Comparison](phase4/ml_comparison_bar.png)

---

## 7. Phase 6 — End-to-End Latency

**Script:** `experiments/latency_measure.py`
**Output folder:** `results/phase6/`
**Platform:** Apple M3 MacBook, 100 synthetic frames, 5-person DB

| Component | Mean | Median | p95 |
|-----------|------|--------|-----|
| Face detection (YuNet) | 7.23 ms | 6.64 ms | 10.62 ms |
| Face embedding (SFace) | 4.06 ms | 3.77 ms | 6.37 ms |
| Identity matching | 0.37 ms | 0.12 ms | 1.66 ms |
| Temporal vote | 0.006 ms | 0.006 ms | 0.007 ms |
| Gesture (MediaPipe) | 16.93 ms | 17.54 ms | 23.26 ms |
| HTTP (not connected) | 0.00 ms | — | — |
| **Total pipeline** | **29.06 ms** | **29.08 ms** | **38.51 ms** |

**Throughput:** ~34 FPS mean, ~26 FPS at p95

**Breakdown:** MediaPipe gesture dominates at 58% of total latency. Face detection is 25%, SFace embedding is 14%.

**Raspberry Pi 5 estimate:** Approximately 3–4× slower → ~90–120 ms → 8–11 FPS. Real-time gesture control is feasible but tight.

> Note: HTTP is 0 ms because no robot was connected. A WiFi round-trip adds 10–50 ms.

![Latency Breakdown](phase6/latency_breakdown.png)
![Latency CDF](phase6/latency_cdf.png)

---

## 8. Phase 7 — Security Analysis

**Script:** `experiments/security_analysis.py`
**Output folder:** `results/phase7/`

| Attack | N Tested | Succeeded | Rate | Finding |
|--------|----------|-----------|------|---------|
| A1: Photo/replay attack | 16 | 16 | **100%** | No liveness detection — photo of enrolled user works |
| A2: Cross-identity (Harshhini) | 13 | 0 | 0% | Max score 0.363 < 0.42 threshold |
| A3: Enrolled stress test | 92 | 92 | 100% | All enrolled training images pass threshold (by design) |
| A4: Centroid gate (enrolled) | 92 | 91 | 98.9% | Centroid gate correctly validates enrolled users |
| A4: Centroid gate (impostor) | 13 | 0 | 0% | Centroid gate rejects Harshhini (centroid score mean=0.198) |
| A5: Single-frame spoof | 13 | 0 | 0% | Harshhini cannot spoof even in one frame |

**Critical finding:** The system is trivially defeated by a printed photo of an enrolled user (A1=100%). This is a known limitation — liveness detection (blink, depth, infrared) is not implemented. For a robot dog controller this may be acceptable; for access control it is not.

**Positive finding:** The 1 impostor tested (Harshhini) cannot spoof the system in any tested scenario. The score and centroid margins are large.

![Score Distribution](phase7/score_distribution.png)

---

## 9. Limitations and Honest Caveats

1. **Small impostor set (N=1 identity, 13 images).** FAR=0.000 with CI [0.000, 0.228]. The true FAR could be up to 22.8% with different impostors. A lookalike or sibling of an enrolled user was never tested.

2. **Static images only.** All gesture and face tests used still photos. Live video introduces motion blur, occlusion, lighting flicker, and temporal inconsistency not captured here.

3. **Single lighting condition (L0).** All data collected under standard indoor lighting. Performance under backlighting, dim/dark conditions, or direct sunlight is unknown.

4. **No liveness detection.** A printed photo of an enrolled user passes all gates (A1=100%). Deployment without liveness detection is not secure against physical access attacks.

5. **Gesture rules not fully generalised.** Rule-based recall drops significantly across subjects for WALK (24%), STOP (40%), BARK (57%). Rules were designed on 2 subjects and do not adapt to hand size or pose variation.

6. **ML gesture classifiers overfit to subjects.** SVM/RF/KNN achieve 99%+ closed-set but 30–56% LOSO. They are not suitable for deployment with new users without retraining.

7. **No Raspberry Pi benchmarks.** Phase 5 latency on target hardware (RPi 4/5) is estimated, not measured.

8. **Margin and centroid gates not stress-tested.** With only 5 enrolled subjects the top-match score dominates second-best by a large margin, so the margin gate is rarely triggered. At 20+ subjects with similar faces, the gate may become more relevant.

9. **One impostor demographic.** Harshhini is a female Indian subject. The impostor set does not cover different ethnicities, ages, or facial structures. FAR may differ across demographics.

10. **Ground truth quality.** Gesture labels are derived from folder structure at capture time. No independent verification of label correctness was performed.

---

## 10. Key Findings

| Finding | Value | Confidence |
|---------|-------|------------|
| Face TAR (Config A, 5-person) | 96.8% | N=31 enrolled test |
| Face FAR (all configs) | 0.0% | CI [0.000, 22.8%] — low confidence |
| Face ACC (full two-gate) | 93.2% | N=44 |
| Score gap (enrolled min vs impostor max) | 0.298 | On Harshhini only |
| Gesture rule-based ACC (5 subjects) | 56.3% | N=750 |
| Gesture rule-based Macro F1 | 0.693 | N=750 |
| Gesture SVM LOSO (cross-subject) | 55.9% | 5-fold LOSO |
| Rule-based ≈ SVM LOSO | Yes | Both ~56% |
| Total pipeline latency (M3) | 29 ms (~34 FPS) | N=100 frames |
| Photo-attack success rate | 100% | No liveness detection |

**Summary:** REVO demonstrates a functioning face+gesture pipeline. Face recognition works well on the enrolled user set with zero impostor false accepts on the tested impostor. Gesture recognition reveals significant cross-subject variability — rules that work for 2 subjects degrade at 5 subjects. ML classifiers do not solve this (they overfit worse). The primary outstanding risks are no liveness detection and insufficient impostor coverage.

---

## 11. File Index

```
results/
├── phase2/
│   ├── gate_comparison.csv         # TAR/FAR/FRR/ACC for 4 gate configs
│   ├── gate_comparison_bar.png
│   ├── recognition_results.csv     # Per-image results
│   ├── lighting_ablation.csv
│   ├── lighting_ablation_bar.png
│   ├── lbph_comparison.csv         # LBPH vs SFace
│   ├── roc_curve.png
│   ├── roc_data.csv
│   ├── threshold_sweep.png
│   └── margin_sweep.csv
├── phase3/
│   ├── voting_results.csv
│   ├── voting_heatmap_latency.png
│   ├── voting_heatmap_security.png
│   └── frame_skip_bar.png
├── phase4/
│   ├── gesture_per_class_metrics.csv  # Per-class P/R/F1 for rule-based
│   ├── gesture_per_class_bar.png
│   ├── gesture_results.csv
│   ├── gesture_confusion_matrix.png
│   ├── ml_comparison.csv              # Rule vs SVM vs RF vs KNN
│   └── ml_comparison_bar.png
├── phase6/
│   ├── latency_summary.csv            # Mean/median/p95/p99 per component
│   ├── latency_raw.csv
│   ├── latency_breakdown.png
│   ├── latency_cdf.png
│   └── gesture_vote_sweep.csv
└── phase7/
    ├── security_summary.csv           # 5 attack scenarios
    ├── security_margins.csv
    └── score_distribution.png
```
