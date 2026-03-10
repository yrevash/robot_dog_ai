#!/usr/bin/env python3
"""
experiments/eval_gesture.py
============================
Phase 4 — Gesture Classifier Evaluation for the REVO Robot AI project.

Evaluates the rule-based GestureClassifier from revo_pi.py against ground-truth
gesture labels in three modes:

  MODE 1  live     — capture frames from webcam interactively, one class at a time
  MODE 2  dataset  — batch-evaluate images in gesture_dataset/<subject>/<gesture>/
  MODE 3  compare  — benchmark rule-based vs SVM / Random Forest / KNN (requires sklearn)

Usage
-----
  python experiments/eval_gesture.py                    # auto-selects mode
  python experiments/eval_gesture.py --mode live
  python experiments/eval_gesture.py --mode dataset
  python experiments/eval_gesture.py --mode compare

Outputs (all written to results/phase4/)
-----------------------------------------
  gesture_results.csv             per-sample predictions
  gesture_per_class_metrics.csv   precision / recall / F1 / support per class
  ml_comparison.csv               ML method accuracy & F1 (compare mode only)
  gesture_confusion_matrix.png    10×10 confusion-matrix heat-map
  gesture_per_class_bar.png       per-class F1 bar chart
  ml_comparison_bar.png           accuracy bar chart across methods (compare mode)

Dependencies
------------
  opencv-contrib-python, numpy, mediapipe        (always required)
  sklearn                                        (required for --mode compare only)
"""

from __future__ import annotations

# ── stdlib ────────────────────────────────────────────────────────────────────
import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Path bootstrap (must happen before any project imports) ───────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS  = Path(__file__).resolve().parent
for _p in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT), str(EXPERIMENTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Shared utilities (provides setup_logging, get_results_dir, save_csv …) ───
from utils import (
    GESTURE_DATA,
    MODELS_DIR,
    apply_paper_style,
    check_gesture_data_exist,
    get_results_dir,
    save_csv,
    setup_logging,
)

# ── Third-party: always required ──────────────────────────────────────────────
try:
    import cv2
except ImportError:
    sys.exit("ERROR: opencv-contrib-python is not installed.  Run: pip install opencv-contrib-python")

try:
    import numpy as np
except ImportError:
    sys.exit("ERROR: numpy is not installed.  Run: pip install numpy")

try:
    import mediapipe as mp  # noqa: F401 — checked here so GestureClassifier can load
    _MP_OK = True
except ImportError:
    _MP_OK = False

# ── GestureClassifier from revo_pi ───────────────────────────────────────────
try:
    from revo_pi import GestureClassifier
    _CLASSIFIER_OK = True
except ImportError as exc:
    _CLASSIFIER_OK = False
    _CLASSIFIER_ERR = str(exc)


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

#: The 10 testable gesture classes in UI display order.
GESTURE_CLASSES: List[str] = [
    "SIT",
    "STAND",
    "WALK",
    "STOP",
    "BARK",
]

#: How many frames to capture per class in live mode.
LIVE_SAMPLES_PER_CLASS: int = 20

#: MediaPipe landmark count (21 points × {x, y} = 42 features for ML).
N_LANDMARKS: int = 21
N_FEATURES:  int = N_LANDMARKS * 2  # 42


# ══════════════════════════════════════════════════════════════════════════════
# Helpers: metrics
# ══════════════════════════════════════════════════════════════════════════════

def _safe_div(num: float, den: float) -> float:
    """Return num/den or 0.0 when den == 0."""
    return num / den if den else 0.0


def compute_per_class_metrics(
    y_true: List[str],
    y_pred: List[str],
    classes: List[str],
) -> List[Dict]:
    """
    Compute per-class precision, recall, F1, and support.

    Predictions of None / empty string are treated as "NO_GESTURE" so they
    can still be tracked as misclassifications.
    """
    NONE_LABEL = "NO_GESTURE"
    y_pred_clean = [p if p else NONE_LABEL for p in y_pred]

    all_classes = list(classes) + ([NONE_LABEL] if NONE_LABEL not in classes else [])

    # Build per-class TP / FP / FN counts
    tp: Dict[str, int] = defaultdict(int)
    fp: Dict[str, int] = defaultdict(int)
    fn: Dict[str, int] = defaultdict(int)

    for true, pred in zip(y_true, y_pred_clean):
        if true == pred:
            tp[true] += 1
        else:
            fn[true] += 1
            fp[pred] += 1

    rows = []
    for cls in classes:
        support = tp[cls] + fn[cls]
        precision = _safe_div(tp[cls], tp[cls] + fp[cls])
        recall    = _safe_div(tp[cls], support)
        f1        = _safe_div(2 * precision * recall, precision + recall)
        rows.append(dict(
            class_name=cls,
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            support=support,
            tp=tp[cls],
            fp=fp[cls],
            fn=fn[cls],
        ))
    return rows


def compute_accuracy(y_true: List[str], y_pred: List[str]) -> float:
    if not y_true:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def macro_f1(per_class_rows: List[Dict]) -> float:
    if not per_class_rows:
        return 0.0
    return sum(r["f1"] for r in per_class_rows) / len(per_class_rows)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers: landmark extraction for ML comparison
# ══════════════════════════════════════════════════════════════════════════════

def extract_landmarks_mediapipe(
    frame_bgr: np.ndarray,
    hands_solution,
) -> Optional[List[float]]:
    """
    Run MediaPipe Hands on *frame_bgr* and return a 42-float feature vector
    [lmk[0].x, lmk[0].y, lmk[1].x, lmk[1].y, … lmk[20].x, lmk[20].y]
    normalised to [0, 1] by MediaPipe internally.

    Supports both Tasks API (mediapipe 0.10+) landmarker objects
    and legacy mp.solutions.hands.Hands objects.

    Returns None when no hand is detected.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Tasks API path (HandLandmarker returned by _make_ml_landmarker)
    if hasattr(hands_solution, "_is_tasks_api"):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = hands_solution.detect(mp_image)
        if not result.hand_landmarks:
            return None
        lmk = result.hand_landmarks[0]
    else:
        # Legacy solutions API
        result = hands_solution.process(rgb)
        if not result.multi_hand_landmarks:
            return None
        lmk = result.multi_hand_landmarks[0].landmark

    features: List[float] = []
    for i in range(N_LANDMARKS):
        features.append(float(lmk[i].x))
        features.append(float(lmk[i].y))
    return features


class _TasksAPIHandWrapper:
    """Thin wrapper around HandLandmarker to signal Tasks API usage."""
    _is_tasks_api = True

    def __init__(self, landmarker) -> None:
        self._lm = landmarker

    def detect(self, mp_image):
        return self._lm.detect(mp_image)

    def close(self) -> None:
        self._lm.close()


def classify_frame_rule_based(
    classifier: "GestureClassifier",
    frame_bgr: np.ndarray,
) -> Optional[str]:
    """
    Run the rule-based classifier on a full frame (no face_bbox restriction),
    returning the gesture label or None.
    """
    return classifier.detect(frame_bgr, face_bbox=None)


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def _make_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    classes: List[str],
) -> np.ndarray:
    """Return a len(classes) × len(classes) count matrix (rows=true, cols=pred)."""
    idx = {c: i for i, c in enumerate(classes)}
    n   = len(classes)
    mat = np.zeros((n, n), dtype=int)
    for true, pred in zip(y_true, y_pred):
        if pred is None or pred not in idx:
            continue                      # off-class predictions excluded from square matrix
        if true not in idx:
            continue
        mat[idx[true], idx[pred]] += 1
    return mat


def plot_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    classes: List[str],
    save_path: Path,
    title: str = "Gesture Confusion Matrix",
) -> None:
    """
    Save a confusion-matrix heat-map as a PNG using matplotlib imshow.
    Each cell is annotated with the integer count.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_paper_style()
    mat = _make_confusion_matrix(y_true, y_pred, classes)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mat, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    # Annotate cells with integer counts
    thresh = mat.max() / 2.0 if mat.max() > 0 else 1
    for row in range(len(classes)):
        for col in range(len(classes)):
            count = mat[row, col]
            color = "white" if count > thresh else "black"
            ax.text(col, row, str(count), ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_per_class_bar(
    per_class_rows: List[Dict],
    save_path: Path,
    title: str = "Per-Class F1 Score",
) -> None:
    """Save a bar chart of per-class F1 scores."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_paper_style()
    names = [r["class_name"] for r in per_class_rows]
    f1s   = [r["f1"] for r in per_class_rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, f1s, color="steelblue", edgecolor="white", linewidth=0.5)

    # Annotate bar tops
    for bar, val in zip(bars, f1s):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_ylim(0, 1.12)
    ax.set_xlabel("Gesture class")
    ax.set_ylabel("F1 score")
    ax.set_title(title)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_ml_comparison_bar(
    methods: List[str],
    accuracies: List[float],
    f1s: List[float],
    save_path: Path,
) -> None:
    """Save an accuracy + F1 grouped bar chart across classification methods."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_paper_style()
    x  = np.arange(len(methods))
    w  = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w / 2, accuracies, w, label="Accuracy",  color="steelblue",  edgecolor="white")
    b2 = ax.bar(x + w / 2, f1s,        w, label="F1 (macro)", color="darkorange", edgecolor="white")

    for bar in list(b1) + list(b2):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.2f}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Gesture Classifier: Rule-Based vs ML Methods")
    ax.legend()

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# MODE 1 — Live camera capture
# ══════════════════════════════════════════════════════════════════════════════

def run_live_mode(
    classifier: "GestureClassifier",
    results_dir: Path,
    log,
    camera_index: int = 0,
    samples_per_class: int = LIVE_SAMPLES_PER_CLASS,
) -> Tuple[List[str], List[str], List[Dict]]:
    """
    Interactively capture webcam frames for each gesture class.

    For each gesture class:
      - Shows a live camera preview with on-screen instructions.
      - Press SPACE to capture *samples_per_class* frames (with real-time feedback).
      - Press Q to skip the current class.

    Returns (y_true, y_pred, per_sample_rows).
    """
    log.info("=" * 60)
    log.info("MODE: Live camera capture")
    log.info(f"Camera index : {camera_index}")
    log.info(f"Samples/class: {samples_per_class}")
    log.info("=" * 60)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        log.error(f"Cannot open camera index {camera_index}.")
        log.error("Try --camera <index> or check that no other app is using the camera.")
        return [], [], []

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    y_true: List[str]  = []
    y_pred: List[str]  = []
    per_sample_rows: List[Dict] = []

    for gesture_idx, gesture in enumerate(GESTURE_CLASSES):
        log.info(f"[{gesture_idx + 1}/{len(GESTURE_CLASSES)}] Ready for gesture: {gesture}")

        # ── Wait-for-SPACE phase ──────────────────────────────────────────────
        waiting = True
        while waiting:
            ret, frame = cap.read()
            if not ret:
                log.warning("Camera read failed — retrying…")
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            display = frame.copy()

            # Live gesture preview
            live_pred = classify_frame_rule_based(classifier, frame)
            pred_text = live_pred if live_pred else "—"

            cv2.putText(display,
                        f"Show: {gesture}  ({gesture_idx + 1}/{len(GESTURE_CLASSES)})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display,
                        f"Live prediction: {pred_text}",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
            cv2.putText(display,
                        "SPACE = capture  |  Q = skip",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("eval_gesture — live capture", display)
            key = cv2.waitKey(30) & 0xFF

            if key == ord("q") or key == ord("Q"):
                log.info(f"  Skipped: {gesture}")
                waiting = False
                break
            elif key == ord(" "):
                waiting = False  # fall through to capture loop

        else:
            continue   # gesture was skipped via Q in the outer loop

        # ── Capture phase ─────────────────────────────────────────────────────
        captured = 0
        log.info(f"  Capturing {samples_per_class} samples for {gesture}…")

        while captured < samples_per_class:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.03)
                continue

            frame = cv2.flip(frame, 1)
            pred  = classify_frame_rule_based(classifier, frame)
            pred_str = pred if pred else "NO_GESTURE"

            y_true.append(gesture)
            y_pred.append(pred_str)
            per_sample_rows.append(dict(
                source="live",
                frame=f"live_{gesture}_{captured:04d}",
                true_label=gesture,
                predicted_label=pred_str,
                correct=(pred_str == gesture),
            ))
            captured += 1

            # Show capture feedback
            display = frame.copy()
            color   = (0, 255, 0) if pred_str == gesture else (0, 0, 255)
            cv2.putText(display,
                        f"CAPTURING {gesture}: {captured}/{samples_per_class}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display,
                        f"Predicted: {pred_str}",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow("eval_gesture — live capture", display)
            cv2.waitKey(1)

        n_correct = sum(1 for r in per_sample_rows[-samples_per_class:] if r["correct"])
        log.info(f"  {gesture}: {n_correct}/{samples_per_class} correct "
                 f"({100.0 * n_correct / samples_per_class:.1f}%)")

    cap.release()
    cv2.destroyAllWindows()
    return y_true, y_pred, per_sample_rows


# ══════════════════════════════════════════════════════════════════════════════
# MODE 2 — Dataset evaluation
# ══════════════════════════════════════════════════════════════════════════════

def _discover_dataset_images(
    gesture_data: Path,
    log,
) -> Tuple[List[Path], List[str], List[str]]:
    """
    Walk gesture_dataset/<subject>/<gesture>/<img>.jpg  (or .png).

    Falls back to gesture_dataset/<gesture>/<img>.jpg if no subject layer is present.

    Returns (image_paths, true_labels, subject_names).
    """
    image_paths:  List[Path] = []
    true_labels:  List[str]  = []
    subject_names: List[str] = []

    if not gesture_data.exists():
        log.error(f"gesture_dataset/ directory not found: {gesture_data}")
        return [], [], []

    # Detect layout: does the first subdirectory contain gesture-named folders?
    subdirs = [d for d in sorted(gesture_data.iterdir()) if d.is_dir()]
    if not subdirs:
        log.warning("gesture_dataset/ is empty.")
        return [], [], []

    gesture_names_lower = {g.lower() for g in GESTURE_CLASSES}
    first_sub_children  = {d.name.upper() for d in subdirs[0].iterdir() if d.is_dir()}
    has_subject_layer   = bool(first_sub_children & set(GESTURE_CLASSES))

    if has_subject_layer:
        # Layout: gesture_dataset/<subject>/<GESTURE>/
        for subject_dir in sorted(gesture_data.iterdir()):
            if not subject_dir.is_dir():
                continue
            for gesture_dir in sorted(subject_dir.iterdir()):
                if not gesture_dir.is_dir():
                    continue
                gesture_label = gesture_dir.name.upper()
                if gesture_label not in GESTURE_CLASSES:
                    log.debug(f"Skipping unknown gesture folder: {gesture_dir}")
                    continue
                for img_path in sorted(gesture_dir.iterdir()):
                    if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                        image_paths.append(img_path)
                        true_labels.append(gesture_label)
                        subject_names.append(subject_dir.name)
    else:
        # Layout: gesture_dataset/<GESTURE>/
        for gesture_dir in sorted(gesture_data.iterdir()):
            if not gesture_dir.is_dir():
                continue
            gesture_label = gesture_dir.name.upper()
            if gesture_label not in GESTURE_CLASSES:
                log.debug(f"Skipping unknown gesture folder: {gesture_dir}")
                continue
            for img_path in sorted(gesture_dir.iterdir()):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    image_paths.append(img_path)
                    true_labels.append(gesture_label)
                    subject_names.append("default")

    log.info(f"Dataset: {len(image_paths)} images, "
             f"{len(set(true_labels))} classes, "
             f"{len(set(subject_names))} subject(s)")
    return image_paths, true_labels, subject_names


def _load_ground_truth_csv(gt_csv: Path, log) -> Dict[str, str]:
    """
    Load gesture_dataset/ground_truth.csv if present.
    Expected columns: filename (or path), label.
    Returns dict mapping filename stem → label.
    """
    if not gt_csv.exists():
        return {}
    mapping: Dict[str, str] = {}
    try:
        with open(gt_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fn  = (row.get("filename") or row.get("path") or row.get("image")
                       or row.get("image_path") or "")
                lbl = (row.get("label") or row.get("gesture") or row.get("class_name") or "").upper()
                if fn and lbl:
                    mapping[Path(fn).stem] = lbl
        log.info(f"Loaded ground_truth.csv: {len(mapping)} entries")
    except Exception as exc:
        log.warning(f"Could not read ground_truth.csv: {exc}")
    return mapping


def run_dataset_mode(
    classifier: "GestureClassifier",
    results_dir: Path,
    log,
) -> Tuple[List[str], List[str], List[Dict], List[str]]:
    """
    Batch-evaluate all images in gesture_dataset/.

    Returns (y_true, y_pred, per_sample_rows, subject_names_per_sample).
    """
    log.info("=" * 60)
    log.info("MODE: Dataset evaluation")
    log.info(f"Dataset path: {GESTURE_DATA}")
    log.info("=" * 60)

    image_paths, true_labels, subjects = _discover_dataset_images(GESTURE_DATA, log)
    if not image_paths:
        log.error("No images found.  Check gesture_dataset/ layout.")
        return [], [], [], []

    # Optional ground_truth.csv override
    gt_map = _load_ground_truth_csv(GESTURE_DATA / "ground_truth.csv", log)

    y_true:  List[str]  = []
    y_pred:  List[str]  = []
    per_sample_rows: List[Dict] = []
    subject_per_sample: List[str] = []

    total = len(image_paths)
    t0    = time.time()

    for i, (img_path, true_label, subject) in enumerate(
        zip(image_paths, true_labels, subjects)
    ):
        # Ground truth: CSV override takes priority over directory name
        true_label = gt_map.get(img_path.stem, true_label)

        frame = cv2.imread(str(img_path))
        if frame is None:
            log.warning(f"Cannot read image: {img_path}")
            continue

        pred     = classify_frame_rule_based(classifier, frame)
        pred_str = pred if pred else "NO_GESTURE"

        y_true.append(true_label)
        y_pred.append(pred_str)
        subject_per_sample.append(subject)
        per_sample_rows.append(dict(
            source=subject,
            frame=str(img_path.relative_to(PROJECT_ROOT)),
            true_label=true_label,
            predicted_label=pred_str,
            correct=(pred_str == true_label),
        ))

        if (i + 1) % 50 == 0 or (i + 1) == total:
            elapsed  = time.time() - t0
            acc_so_far = compute_accuracy(y_true, y_pred)
            log.info(f"  [{i + 1:>5}/{total}] running accuracy={acc_so_far:.3f}  "
                     f"elapsed={elapsed:.1f}s")

    log.info(f"Dataset evaluation complete: {len(y_true)} samples processed.")
    return y_true, y_pred, per_sample_rows, subject_per_sample


# ══════════════════════════════════════════════════════════════════════════════
# MODE 3 — ML comparison
# ══════════════════════════════════════════════════════════════════════════════

def _build_feature_matrix(
    image_paths: List[Path],
    true_labels: List[str],
    subjects: List[str],
    hands_solution,
    log,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Run MediaPipe Hands on every image and collect 42-float feature vectors.
    Samples where no hand is detected are silently dropped.

    Returns (X, y_labels, y_subjects).
    """
    X_rows: List[List[float]] = []
    y_out:  List[str]         = []
    s_out:  List[str]         = []
    n_no_detect = 0

    for img_path, label, subject in zip(image_paths, true_labels, subjects):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        feats = extract_landmarks_mediapipe(frame, hands_solution)
        if feats is None:
            n_no_detect += 1
            continue
        X_rows.append(feats)
        y_out.append(label)
        s_out.append(subject)

    log.info(f"Feature extraction: {len(X_rows)} usable / "
             f"{len(image_paths)} total  ({n_no_detect} no-hand detections dropped)")
    return np.array(X_rows, dtype=np.float32), y_out, s_out


def _evaluate_rule_based_on_dataset(
    image_paths: List[Path],
    true_labels: List[str],
    classifier: "GestureClassifier",
    log,
) -> Tuple[float, float]:
    """
    Re-evaluate the rule-based classifier on the same image list used for ML,
    so numbers are directly comparable (same sample set, excluding no-detection drops).
    Returns (accuracy, macro_f1).
    """
    y_true, y_pred = [], []
    for img_path, label in zip(image_paths, true_labels):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        pred     = classify_frame_rule_based(classifier, frame)
        pred_str = pred if pred else "NO_GESTURE"
        y_true.append(label)
        y_pred.append(pred_str)

    acc  = compute_accuracy(y_true, y_pred)
    rows = compute_per_class_metrics(y_true, y_pred, GESTURE_CLASSES)
    mf1  = macro_f1(rows)
    log.info(f"Rule-based  acc={acc:.4f}  macro-F1={mf1:.4f}")
    return acc, mf1


def _run_sklearn_method(
    name: str,
    clf,
    X: np.ndarray,
    y: List[str],
    n_splits: int,
    log,
) -> Tuple[float, float, List[float]]:
    """
    5-fold cross-validation for a sklearn estimator.
    Returns (mean_accuracy, mean_macro_f1, per_class_f1_list).
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import LabelEncoder

    le  = LabelEncoder()
    y_enc = le.fit_transform(y)

    skf    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs:  List[float] = []
    mf1s:  List[float] = []
    per_class_f1_accum: Dict[str, List[float]] = defaultdict(list)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y_enc)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y_enc[train_idx], y_enc[test_idx]

        clf.fit(X_tr, y_tr)
        y_hat = clf.predict(X_te)

        accs.append(accuracy_score(y_te, y_hat))
        mf1s.append(f1_score(y_te, y_hat, average="macro", zero_division=0))

        # Per-class F1 for this fold
        f1_per = f1_score(y_te, y_hat, average=None, labels=range(len(le.classes_)),
                          zero_division=0)
        for cls_idx, cls_name in enumerate(le.classes_):
            per_class_f1_accum[cls_name].append(f1_per[cls_idx])

        log.debug(f"  {name} fold {fold_idx + 1}: acc={accs[-1]:.4f}  "
                  f"macro-F1={mf1s[-1]:.4f}")

    mean_acc  = float(np.mean(accs))
    mean_mf1  = float(np.mean(mf1s))
    mean_pcf1 = [float(np.mean(per_class_f1_accum[c])) for c in GESTURE_CLASSES
                 if c in per_class_f1_accum]
    log.info(f"{name:25s}  acc={mean_acc:.4f}  macro-F1={mean_mf1:.4f}")
    return mean_acc, mean_mf1, mean_pcf1


def _run_cross_subject(
    name: str,
    clf,
    X: np.ndarray,
    y: List[str],
    subjects: List[str],
    log,
) -> Optional[float]:
    """
    Leave-one-subject-out cross-validation.
    Returns mean accuracy, or None if only one subject exists.
    """
    unique_subjects = list(set(subjects))
    if len(unique_subjects) < 2:
        log.info(f"{name}: cross-subject evaluation skipped (only 1 subject).")
        return None

    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import LabelEncoder

    le    = LabelEncoder()
    y_enc = le.fit_transform(y)
    s_arr = np.array(subjects)
    accs  = []

    for test_subj in unique_subjects:
        mask_test  = s_arr == test_subj
        mask_train = ~mask_test
        if mask_train.sum() == 0 or mask_test.sum() == 0:
            continue
        clf.fit(X[mask_train], y_enc[mask_train])
        y_hat = clf.predict(X[mask_test])
        accs.append(accuracy_score(y_enc[mask_test], y_hat))
        log.debug(f"  {name} LOSO on '{test_subj}': acc={accs[-1]:.4f}")

    mean = float(np.mean(accs)) if accs else None
    log.info(f"{name:25s}  LOSO cross-subject acc={mean:.4f}" if mean is not None
             else f"{name}: LOSO skipped")
    return mean


def run_compare_mode(
    classifier: "GestureClassifier",
    results_dir: Path,
    log,
) -> List[Dict]:
    """
    ML comparison mode.  Requires gesture_dataset/ and sklearn.

    Returns a list of dicts suitable for writing to ml_comparison.csv.
    """
    log.info("=" * 60)
    log.info("MODE: ML comparison (rule-based vs SVM / RF / KNN)")
    log.info("=" * 60)

    # ── Import sklearn (optional dependency) ──────────────────────────────────
    try:
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neighbors import KNeighborsClassifier
    except ImportError:
        log.error("scikit-learn is not installed.")
        log.error("Run: pip install scikit-learn")
        log.error("Falling back to dataset mode.")
        y_true, y_pred, rows, subjects = run_dataset_mode(classifier, results_dir, log)
        return []

    if not GESTURE_DATA.exists():
        log.error(f"gesture_dataset/ not found at {GESTURE_DATA}")
        log.error("Cannot run compare mode without a dataset.")
        return []

    # ── Discover images ───────────────────────────────────────────────────────
    image_paths, true_labels, subjects = _discover_dataset_images(GESTURE_DATA, log)
    if len(image_paths) < 30:
        log.error(f"Only {len(image_paths)} images found — need ≥30 for meaningful ML comparison.")
        return []

    # ── Build feature matrix (Tasks API for mediapipe 0.10+) ─────────────────
    import mediapipe as mp
    try:
        from mediapipe.tasks import python as _mp_tasks
        from mediapipe.tasks.python import vision as _mp_vision
        _model_path = str(MODELS_DIR / "hand_landmarker.task")
        _base = _mp_tasks.BaseOptions(model_asset_path=_model_path)
        _opts = _mp_vision.HandLandmarkerOptions(
            base_options=_base,
            running_mode=_mp_vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.45,
            min_tracking_confidence=0.5,
        )
        _lm = _mp_vision.HandLandmarker.create_from_options(_opts)
        hands = _TasksAPIHandWrapper(_lm)
        log.info("ML feature extraction: using MediaPipe Tasks API (0.10+)")
    except Exception as _exc:
        log.debug("Tasks API unavailable (%s), trying mp.solutions", _exc)
        try:
            hands = mp.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            log.info("ML feature extraction: using MediaPipe solutions API (0.9.x)")
        except Exception as _exc2:
            log.error(f"Could not initialise MediaPipe for feature extraction: {_exc2}")
            return []
    try:
        X, y_ml, subjects_ml = _build_feature_matrix(
            image_paths, true_labels, subjects, hands, log
        )
    finally:
        hands.close()

    if len(X) < 30:
        log.error(f"Only {len(X)} samples with detected hands — need ≥30.")
        return []

    N_SPLITS = min(5, min(
        sum(1 for lbl in y_ml if lbl == cls)
        for cls in set(y_ml)
    ))
    if N_SPLITS < 2:
        log.error("Not enough per-class samples for cross-validation (need ≥2 per class).")
        return []
    log.info(f"Cross-validation: {N_SPLITS}-fold  ({len(X)} usable samples)")

    # ── Rule-based baseline (matched sample set) ──────────────────────────────
    # Re-evaluate on the subset where hands were detected
    rb_y_true, rb_y_pred = [], []
    for img_path, label in zip(image_paths, true_labels):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        pred     = classify_frame_rule_based(classifier, frame)
        pred_str = pred if pred else "NO_GESTURE"
        rb_y_true.append(label)
        rb_y_pred.append(pred_str)

    rb_acc  = compute_accuracy(rb_y_true, rb_y_pred)
    rb_rows = compute_per_class_metrics(rb_y_true, rb_y_pred, GESTURE_CLASSES)
    rb_mf1  = macro_f1(rb_rows)
    rb_pcf1 = [r["f1"] for r in rb_rows]
    log.info(f"{'Rule-based':25s}  acc={rb_acc:.4f}  macro-F1={rb_mf1:.4f}")

    # ── ML methods ────────────────────────────────────────────────────────────
    methods_config = [
        ("SVM (RBF, C=10)",    SVC(kernel="rbf", C=10, gamma="scale", random_state=42)),
        ("Random Forest",      RandomForestClassifier(n_estimators=100, random_state=42)),
        ("KNN (k=5)",          KNeighborsClassifier(n_neighbors=5, metric="euclidean")),
    ]

    comparison_rows: List[Dict] = [
        dict(
            method="Rule-based",
            accuracy=round(rb_acc, 4),
            f1_macro=round(rb_mf1, 4),
            f1_per_class=";".join(f"{v:.4f}" for v in rb_pcf1),
            cross_subject_acc="N/A",
        )
    ]

    for mname, clf in methods_config:
        acc, mf1_val, pcf1 = _run_sklearn_method(
            mname, clf, X, y_ml, N_SPLITS, log
        )
        # Cross-subject generalization
        cs_acc = _run_cross_subject(mname, clf, X, y_ml, subjects_ml, log)

        comparison_rows.append(dict(
            method=mname,
            accuracy=round(acc, 4),
            f1_macro=round(mf1_val, 4),
            f1_per_class=";".join(f"{v:.4f}" for v in pcf1),
            cross_subject_acc=round(cs_acc, 4) if cs_acc is not None else "N/A",
        ))

    # ── Summary table ─────────────────────────────────────────────────────────
    log.info("")
    log.info("-" * 60)
    log.info(f"{'Method':<25} {'Accuracy':>9} {'F1 macro':>9}")
    log.info("-" * 60)
    for row in comparison_rows:
        log.info(f"{row['method']:<25} {row['accuracy']:>9.4f} {row['f1_macro']:>9.4f}")
    log.info("-" * 60)

    return comparison_rows


# ══════════════════════════════════════════════════════════════════════════════
# Output: save results
# ══════════════════════════════════════════════════════════════════════════════

def save_all_outputs(
    y_true: List[str],
    y_pred: List[str],
    per_sample_rows: List[Dict],
    results_dir: Path,
    log,
    compare_rows: Optional[List[Dict]] = None,
) -> None:
    """Save all CSV and PNG outputs to results_dir."""

    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Per-sample CSV ────────────────────────────────────────────────────────
    if per_sample_rows:
        csv_path = results_dir / "gesture_results.csv"
        save_csv(
            csv_path,
            per_sample_rows,
            fieldnames=["source", "frame", "true_label", "predicted_label", "correct"],
        )
        log.info(f"Saved: {csv_path}")

    # ── Per-class metrics CSV ─────────────────────────────────────────────────
    if y_true:
        per_class = compute_per_class_metrics(y_true, y_pred, GESTURE_CLASSES)
        pc_path   = results_dir / "gesture_per_class_metrics.csv"
        save_csv(
            pc_path,
            per_class,
            fieldnames=["class_name", "precision", "recall", "f1", "support", "tp", "fp", "fn"],
        )
        log.info(f"Saved: {pc_path}")

        # ── Confusion matrix PNG ──────────────────────────────────────────────
        cm_path = results_dir / "gesture_confusion_matrix.png"
        plot_confusion_matrix(y_true, y_pred, GESTURE_CLASSES, cm_path)
        log.info(f"Saved: {cm_path}")

        # ── Per-class F1 bar PNG ──────────────────────────────────────────────
        bar_path = results_dir / "gesture_per_class_bar.png"
        plot_per_class_bar(per_class, bar_path)
        log.info(f"Saved: {bar_path}")

    # ── ML comparison CSV + bar PNG ───────────────────────────────────────────
    if compare_rows:
        ml_path = results_dir / "ml_comparison.csv"
        save_csv(
            ml_path,
            compare_rows,
            fieldnames=["method", "accuracy", "f1_macro", "f1_per_class", "cross_subject_acc"],
        )
        log.info(f"Saved: {ml_path}")

        methods    = [r["method"]   for r in compare_rows]
        accuracies = [float(r["accuracy"]) for r in compare_rows]
        f1s        = [float(r["f1_macro"]) for r in compare_rows]

        ml_bar_path = results_dir / "ml_comparison_bar.png"
        plot_ml_comparison_bar(methods, accuracies, f1s, ml_bar_path)
        log.info(f"Saved: {ml_bar_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Summary logging
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(
    y_true: List[str],
    y_pred: List[str],
    log,
) -> None:
    """Print a final summary table to the log."""
    if not y_true:
        log.warning("No samples to summarise.")
        return

    acc        = compute_accuracy(y_true, y_pred)
    per_class  = compute_per_class_metrics(y_true, y_pred, GESTURE_CLASSES)
    mf1        = macro_f1(per_class)

    log.info("")
    log.info("=" * 60)
    log.info("FINAL SUMMARY")
    log.info("=" * 60)
    log.info(f"  Total samples : {len(y_true)}")
    log.info(f"  Accuracy      : {acc:.4f}  ({100 * acc:.2f}%)")
    log.info(f"  Macro F1      : {mf1:.4f}")
    log.info("")
    log.info(f"  {'Class':<12} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Supp':>6}")
    log.info("  " + "-" * 46)
    for row in per_class:
        log.info(
            f"  {row['class_name']:<12} "
            f"{row['precision']:>6.3f} "
            f"{row['recall']:>6.3f} "
            f"{row['f1']:>6.3f} "
            f"{row['support']:>6}"
        )
    log.info("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 4 — Gesture Classifier Evaluation.\n"
            "Evaluates the rule-based GestureClassifier from revo_pi.py in three modes."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python experiments/eval_gesture.py\n"
            "  python experiments/eval_gesture.py --mode live\n"
            "  python experiments/eval_gesture.py --mode dataset\n"
            "  python experiments/eval_gesture.py --mode compare\n"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["live", "dataset", "compare", "auto"],
        default="auto",
        help=(
            "Evaluation mode: 'live' (webcam), 'dataset' (gesture_dataset/), "
            "'compare' (ML methods), or 'auto' (choose based on what's available). "
            "Default: auto"
        ),
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        metavar="INDEX",
        help="Camera device index for live mode (default: 0).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=LIVE_SAMPLES_PER_CLASS,
        metavar="N",
        help=f"Frames to capture per gesture class in live mode (default: {LIVE_SAMPLES_PER_CLASS}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Override output directory (default: results/phase4/).",
    )
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    args = parse_result = _parse_args()
    log  = setup_logging("phase4_gesture")

    # ── Pre-flight checks ─────────────────────────────────────────────────────
    # mediapipe is required for live and compare modes; dataset mode does not
    # use it for the rule-based classifier path, only for compare-mode ML.
    if not _MP_OK and args.mode in ("live", "compare"):
        log.error("mediapipe is not installed.  Run: pip install mediapipe")
        return 1
    if not _MP_OK and args.mode == "auto":
        log.warning("mediapipe not installed — live mode unavailable; will use dataset mode if data exists.")

    if not _CLASSIFIER_OK:
        log.error(f"Could not import GestureClassifier from revo_pi: {_CLASSIFIER_ERR}")
        log.error(f"Expected revo_pi.py at: {PROJECT_ROOT / 'src' / 'revo_pi.py'}")
        return 1

    # ── Resolve output directory ──────────────────────────────────────────────
    if args.output_dir is not None:
        results_dir = args.output_dir
        results_dir.mkdir(parents=True, exist_ok=True)
    else:
        results_dir = get_results_dir("phase4")

    log.info(f"Output directory: {results_dir}")

    # ── Auto-select mode ──────────────────────────────────────────────────────
    mode = args.mode
    if mode == "auto":
        if GESTURE_DATA.exists() and any(GESTURE_DATA.iterdir()):
            mode = "dataset"
            log.info("Auto-selected mode: dataset (gesture_dataset/ found)")
        elif _MP_OK:
            mode = "live"
            log.info("Auto-selected mode: live (gesture_dataset/ not found)")
        else:
            log.error("gesture_dataset/ not found and mediapipe not installed — cannot run.")
            log.error("Either collect a dataset or install mediapipe for live mode.")
            return 1

    # ── Instantiate classifier ────────────────────────────────────────────────
    try:
        classifier = GestureClassifier(hand_max_dim=320)
        log.info("GestureClassifier initialised successfully.")
    except RuntimeError as exc:
        log.error(f"Failed to initialise GestureClassifier: {exc}")
        return 1

    try:
        # ── Dispatch mode ─────────────────────────────────────────────────────
        y_true:   List[str]  = []
        y_pred:   List[str]  = []
        per_rows: List[Dict] = []
        compare_rows: Optional[List[Dict]] = None

        if mode == "live":
            y_true, y_pred, per_rows = run_live_mode(
                classifier,
                results_dir,
                log,
                camera_index=args.camera,
                samples_per_class=args.samples,
            )

        elif mode == "dataset":
            if not GESTURE_DATA.exists():
                log.error(f"gesture_dataset/ not found at {GESTURE_DATA}")
                log.error("To collect a dataset run a capture script, or use --mode live.")
                return 1
            y_true, y_pred, per_rows, _ = run_dataset_mode(
                classifier, results_dir, log
            )

        elif mode == "compare":
            if not GESTURE_DATA.exists():
                log.error(f"gesture_dataset/ not found at {GESTURE_DATA}")
                log.error("Cannot run compare mode without gesture_dataset/.")
                log.error("Collect images first, then re-run with --mode compare.")
                return 1
            # Run dataset eval first (populates y_true / y_pred for plots)
            y_true, y_pred, per_rows, _ = run_dataset_mode(
                classifier, results_dir, log
            )
            compare_rows = run_compare_mode(classifier, results_dir, log)

        # ── Print summary ─────────────────────────────────────────────────────
        print_summary(y_true, y_pred, log)

        # ── Save all outputs ──────────────────────────────────────────────────
        save_all_outputs(y_true, y_pred, per_rows, results_dir, log, compare_rows)

        log.info(f"\nAll results written to: {results_dir}")

    finally:
        classifier.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
