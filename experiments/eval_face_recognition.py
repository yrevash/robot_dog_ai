#!/usr/bin/env python3
"""
Phase 2 — Face Recognition Accuracy Evaluation
===============================================
Tests the two-gate identity matching system.

MODES:
  demo  — Uses existing known_faces/ and face_db.npz.
          Simulates impostors by applying synthetic lighting transforms.
          Compares 4 gate configurations. Runs immediately, no extra data needed.

  full  — Requires test_faces/ directory with:
            test_faces/ground_truth.csv
            test_faces/enrolled/<PersonName>/<lighting>/<img>.jpg
            test_faces/impostors/<Name>/<img>.jpg

OUTPUTS (saved to results/phase2/):
  recognition_results.csv          Raw per-image predictions for each config
  gate_comparison.csv              TAR / FAR / FRR / ACC per gate config
  lighting_ablation.csv            Accuracy per lighting condition w/ and w/o norm
  gate_comparison_bar.png          Bar chart: TAR/FAR per gate config
  lighting_ablation_bar.png        Grouped bar chart: accuracy per condition
  lbph_comparison.csv              LBPH baseline vs SFace two-gate
  logs/phase2_*.log                Full run log

USAGE:
  python experiments/eval_face_recognition.py              # auto-selects mode
  python experiments/eval_face_recognition.py --mode demo
  python experiments/eval_face_recognition.py --mode full
  python experiments/eval_face_recognition.py --mode full --test-dir /path/to/test_faces
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import (
    setup_logging, get_results_dir, save_csv, load_csv,
    compute_metrics, apply_paper_style, check_db_exists, print_metrics_table,
    DB_FILE, KNOWN_FACES, TEST_FACES
)
import face_embedding as fe


# ── Gate configurations to compare ─────────────────────────────────────────────
GATE_CONFIGS = {
    "A: Score only":        dict(threshold=0.42, margin=0.0,  centroid_threshold=0.0),
    "B: Score+Margin":      dict(threshold=0.42, margin=0.06, centroid_threshold=0.0),
    "C: Score+Centroid":    dict(threshold=0.42, margin=0.0,  centroid_threshold=0.40),
    "D: Full two-gate":     dict(threshold=0.42, margin=0.06, centroid_threshold=0.40),
}

# ── Synthetic lighting transforms for demo mode ─────────────────────────────────
LIGHTING_TRANSFORMS = {
    "L0_Normal":   lambda img: img,
    "L1_Dim":      lambda img: cv2.convertScaleAbs(img, alpha=0.45, beta=0),
    "L2_Bright":   lambda img: cv2.convertScaleAbs(img, alpha=1.0,  beta=60),
    "L3_LowContrast": lambda img: cv2.convertScaleAbs(img, alpha=0.65, beta=30),
    "L4_Overexposed": lambda img: np.clip(img.astype(np.int32) + 90, 0, 255).astype(np.uint8),
}


# ── Core evaluation function ───────────────────────────────────────────────────
def evaluate_image(
    img: np.ndarray,
    detector,
    recognizer,
    db_embeddings: np.ndarray,
    db_names: np.ndarray,
    centroids: np.ndarray,
    centroid_names: np.ndarray,
    gate_cfg: dict,
    normalize: bool,
) -> tuple[str, float, float, float]:
    """
    Run detection + recognition on one image.
    Returns: (predicted_name, sample_score, centroid_score, inference_ms)
    """
    t0 = time.perf_counter()

    proc = fe.normalize_lighting(img) if normalize else img
    faces = fe.detect_faces(detector, proc)

    if not faces:
        return "Unknown", 0.0, 0.0, (time.perf_counter() - t0) * 1000

    emb = fe.embedding_from_face(proc, faces[0], recognizer)
    if emb is None:
        return "Unknown", 0.0, 0.0, (time.perf_counter() - t0) * 1000

    name, s_score, c_score = fe.match_identity(
        emb, db_embeddings, db_names, centroids, centroid_names,
        **gate_cfg
    )
    return name, s_score, c_score, (time.perf_counter() - t0) * 1000


# ── Demo mode ─────────────────────────────────────────────────────────────────
def run_demo(log, out_dir: Path) -> None:
    log.info("=" * 50)
    log.info("MODE: DEMO  (using known_faces/ + synthetic lighting)")
    log.info("=" * 50)

    if not check_db_exists(log):
        return

    fe.check_opencv_requirements()
    fe.ensure_models()

    db_embs, db_names, centroids, centroid_names = fe.load_db(DB_FILE)
    enrolled_names = set(db_names.tolist())
    log.info(f"DB loaded: {len(db_embs)} embeddings, enrolled: {enrolled_names}")

    # Collect all known images
    image_paths = []
    for person_dir in sorted(KNOWN_FACES.iterdir()):
        if not person_dir.is_dir():
            continue
        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                image_paths.append((img_path, person_dir.name, True))  # (path, true_name, is_enrolled)

    if not image_paths:
        log.error("No images found in known_faces/")
        return
    log.info(f"Found {len(image_paths)} enrolled images.")

    detector   = fe.create_detector((640, 480), score_threshold=0.85)
    recognizer = fe.create_recognizer()

    # ── Part 1: Gate configuration comparison ────────────────────────────────
    log.info("\n--- Part 1: Gate Configuration Comparison ---")
    gate_rows   = []   # per-image results
    gate_metrics = {}  # per-config summary

    for config_name, gate_cfg in GATE_CONFIGS.items():
        log.info(f"\n  Testing: {config_name}")
        y_true, y_pred = [], []
        lat_list = []

        for img_path, true_name, is_enrolled in image_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            pred, s_sc, c_sc, lat = evaluate_image(
                img, detector, recognizer,
                db_embs, db_names, centroids, centroid_names,
                gate_cfg, normalize=True
            )
            y_true.append(true_name)
            y_pred.append(pred)
            lat_list.append(lat)
            gate_rows.append({
                "config":        config_name,
                "image":         img_path.name,
                "true_name":     true_name,
                "predicted":     pred,
                "sample_score":  round(s_sc, 4),
                "centroid_score": round(c_sc, 4),
                "latency_ms":    round(lat, 2),
                "correct":       pred == true_name,
            })

        m = compute_metrics(y_true, y_pred, enrolled_names)
        gate_metrics[config_name] = m
        avg_lat = sum(lat_list) / len(lat_list) if lat_list else float("nan")
        log.info(f"    TAR={m['TAR']:.3f}  FAR={m['FAR']:.3f}  FRR={m['FRR']:.3f}  ACC={m['ACC']:.3f}  AvgLat={avg_lat:.1f}ms")

    save_csv(out_dir / "recognition_results.csv", gate_rows,
             ["config","image","true_name","predicted","sample_score","centroid_score","latency_ms","correct"])
    log.info(f"\nPer-image results saved → results/phase2/recognition_results.csv")

    # Summary CSV
    summary_rows = []
    for cfg_name, m in gate_metrics.items():
        summary_rows.append({"config": cfg_name, **{k: round(v, 4) for k, v in m.items()}})
    save_csv(out_dir / "gate_comparison.csv", summary_rows,
             ["config","TAR","FAR","FRR","ACC","TP","FP","FN","TN","total_enrolled","total_impostor"])
    log.info("Gate comparison saved → results/phase2/gate_comparison.csv")
    print_metrics_table(gate_metrics, log)

    # ── Part 2: Lighting ablation ─────────────────────────────────────────────
    log.info("\n--- Part 2: Lighting Robustness (best gate: Full two-gate) ---")
    best_gate = GATE_CONFIGS["D: Full two-gate"]
    light_rows = []

    for light_name, transform in LIGHTING_TRANSFORMS.items():
        for normalize in [False, True]:
            norm_tag = "norm_ON" if normalize else "norm_OFF"
            y_true, y_pred = [], []

            for img_path, true_name, _ in image_paths:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img_transformed = transform(img)
                pred, s_sc, c_sc, lat = evaluate_image(
                    img_transformed, detector, recognizer,
                    db_embs, db_names, centroids, centroid_names,
                    best_gate, normalize=normalize
                )
                y_true.append(true_name)
                y_pred.append(pred)

            m = compute_metrics(y_true, y_pred, enrolled_names)
            light_rows.append({
                "lighting":    light_name,
                "normalize":   norm_tag,
                "TAR":         round(m["TAR"], 4),
                "FAR":         round(m["FAR"], 4),
                "FRR":         round(m["FRR"], 4),
                "ACC":         round(m["ACC"], 4),
            })
            log.info(f"  {light_name:<20} {norm_tag:<10}  ACC={m['ACC']:.3f}  FAR={m['FAR']:.3f}")

    save_csv(out_dir / "lighting_ablation.csv", light_rows,
             ["lighting","normalize","TAR","FAR","FRR","ACC"])
    log.info("Lighting ablation saved → results/phase2/lighting_ablation.csv")

    # ── Part 3: LBPH baseline comparison ─────────────────────────────────────
    log.info("\n--- Part 3: LBPH Baseline Comparison ---")
    try:
        lbph_rows = _run_lbph_comparison(detector, enrolled_names, log)
        save_csv(out_dir / "lbph_comparison.csv", lbph_rows,
                 ["method","TAR","FAR","FRR","ACC"])
        log.info("LBPH comparison saved → results/phase2/lbph_comparison.csv")
    except Exception as exc:
        log.warning(f"LBPH comparison failed: {exc}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    _plot_gate_comparison(gate_metrics, out_dir, log)
    _plot_lighting_ablation(light_rows, out_dir, log)

    log.info("\n✓ Phase 2 (demo) complete. All results in results/phase2/")


# ── Full mode ─────────────────────────────────────────────────────────────────
def run_full(log, out_dir: Path, test_dir: Path) -> None:
    log.info("=" * 50)
    log.info(f"MODE: FULL  (test_dir={test_dir})")
    log.info("=" * 50)

    gt_path = test_dir / "ground_truth.csv"
    if not gt_path.exists():
        log.error(f"ground_truth.csv not found at {gt_path}")
        log.error("Create test_faces/ dataset per RESEARCH.md Phase 1 instructions.")
        return

    if not check_db_exists(log):
        return

    fe.check_opencv_requirements()
    fe.ensure_models()

    db_embs, db_names, centroids, centroid_names = fe.load_db(DB_FILE)
    enrolled_names = set(db_names.tolist())
    log.info(f"DB loaded: {len(db_embs)} embeddings, enrolled: {enrolled_names}")

    gt_rows = load_csv(gt_path)
    log.info(f"Ground truth: {len(gt_rows)} images")

    detector   = fe.create_detector((640, 480), score_threshold=0.85)
    recognizer = fe.create_recognizer()

    # Evaluate each gate config
    gate_metrics = {}
    all_rows     = []

    for config_name, gate_cfg in GATE_CONFIGS.items():
        log.info(f"\n  Testing: {config_name}")
        y_true, y_pred = [], []

        for row in gt_rows:
            img_path = test_dir / row["image_path"]
            if not img_path.exists():
                log.warning(f"  Missing: {img_path}")
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            true_name = row["true_identity"]
            lighting  = row.get("lighting_condition", "unknown")

            pred, s_sc, c_sc, lat = evaluate_image(
                img, detector, recognizer,
                db_embs, db_names, centroids, centroid_names,
                gate_cfg, normalize=True
            )
            y_true.append(true_name)
            y_pred.append(pred)
            all_rows.append({
                "config":        config_name,
                "image":         row["image_path"],
                "lighting":      lighting,
                "true_name":     true_name,
                "predicted":     pred,
                "sample_score":  round(s_sc, 4),
                "centroid_score": round(c_sc, 4),
                "latency_ms":    round(lat, 2),
                "correct":       pred == true_name,
            })

        m = compute_metrics(y_true, y_pred, enrolled_names)
        gate_metrics[config_name] = m
        log.info(f"  TAR={m['TAR']:.3f}  FAR={m['FAR']:.3f}  FRR={m['FRR']:.3f}  ACC={m['ACC']:.3f}")

    save_csv(out_dir / "recognition_results.csv", all_rows,
             ["config","image","lighting","true_name","predicted","sample_score","centroid_score","latency_ms","correct"])

    summary_rows = [{"config": k, **{kk: round(vv, 4) for kk, vv in v.items()}}
                    for k, v in gate_metrics.items()]
    save_csv(out_dir / "gate_comparison.csv", summary_rows,
             ["config","TAR","FAR","FRR","ACC","TP","FP","FN","TN","total_enrolled","total_impostor"])

    print_metrics_table(gate_metrics, log)
    _plot_gate_comparison(gate_metrics, out_dir, log)

    # Lighting ablation from CSV
    _compute_lighting_ablation_from_results(all_rows, out_dir, log)

    # ── LBPH baseline comparison ──────────────────────────────────────────────
    log.info("\n--- LBPH Baseline Comparison ---")
    try:
        lbph_rows = _run_lbph_comparison(detector, enrolled_names, log, test_dir=test_dir)
        save_csv(out_dir / "lbph_comparison.csv", lbph_rows,
                 ["method","TAR","FAR","FRR","ACC"])
        log.info("LBPH comparison saved → results/phase2/lbph_comparison.csv")
    except Exception as exc:
        log.warning(f"LBPH comparison failed: {exc}")

    log.info("\n✓ Phase 2 (full) complete. All results in results/phase2/")


# ── LBPH comparison ────────────────────────────────────────────────────────────
def _run_lbph_comparison(detector, enrolled_names, log, test_dir: Path = None) -> list:
    """
    Train LBPH on known_faces/ (Yash + Aramaan, images 001-017).
    Test on data/test_faces/ using ground_truth.csv (images 018-025 enrolled +
    Harshhini impostors).  Never overlap training and test sets.

    LBPH confidence is a distance: lower = more similar.
    Reject (predict "Unknown") when confidence > CONF_THRESHOLD.
    """
    from utils import KNOWN_FACES, TEST_FACES, load_csv

    train_dir  = KNOWN_FACES
    test_dir   = test_dir if test_dir is not None else TEST_FACES
    gt_path    = test_dir / "ground_truth.csv"

    if not gt_path.exists():
        raise RuntimeError(f"ground_truth.csv not found at {gt_path}")

    # ── Build per-person exclusion sets so we never train on test images ──────
    # gt image_path format: "enrolled/Yash/018.jpg" or "impostors/Harshhini/003.jpg"
    # Key by (person_name, filename) to avoid excluding same-named files from
    # different people (e.g. Harshhini/003.jpg must not exclude Yash/003.jpg).
    gt_rows = load_csv(gt_path)
    test_per_person: dict[str, set[str]] = {}  # person_name -> {filename, ...}
    for row in gt_rows:
        parts = Path(row["image_path"]).parts  # e.g. ("enrolled", "Yash", "018.jpg")
        if len(parts) >= 3:
            person = parts[-2]
            fname  = parts[-1]
        else:
            person = row["true_identity"]
            fname  = Path(row["image_path"]).name
        test_per_person.setdefault(person, set()).add(fname)

    # ── Train on known_faces/ images that are NOT in the test set ─────────────
    # Use a lenient detector for LBPH training so we can harvest more ROIs
    # from the known_faces/ images (which may be lower quality).
    train_detector = fe.create_detector((640, 480), score_threshold=0.5)

    log.info("  Training LBPH recognizer on known_faces/ (excluding test images)...")
    train_images, train_labels, label_map = [], [], {}

    for person_dir in sorted(train_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        person_name = person_dir.name
        if person_name not in enrolled_names:
            # skip any non-enrolled persons (e.g. Harshhini if present)
            continue
        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            if img_path.name in test_per_person.get(person_name, set()):
                continue  # hold-out: skip test images for this person
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Try lenient detection first; fall back to full-image crop
            faces = fe.detect_faces(train_detector, img)
            if faces:
                x1, y1, x2, y2 = fe.bbox(faces[0])
                face_roi = gray[y1:y2, x1:x2]
            else:
                # Whole-image fallback: assume the image is already a face crop
                face_roi = gray
            if face_roi.size == 0:
                continue
            face_roi = cv2.resize(face_roi, (100, 100))
            lbl = label_map.setdefault(person_name, len(label_map))
            train_images.append(face_roi)
            train_labels.append(lbl)

    if not train_images:
        raise RuntimeError("No training images for LBPH after excluding test set")

    log.info(f"  LBPH trained on {len(train_images)} face ROIs "
             f"({list(label_map.keys())})")

    rev_map = {v: k for k, v in label_map.items()}
    lbph = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    lbph.train(train_images, np.array(train_labels))

    # ── Test on test_faces/ via ground_truth.csv ──────────────────────────────
    # LBPH confidence threshold: reject (→ "Unknown") when conf > threshold.
    # Typical LBPH distances for known faces: 40-70; impostors: 80-150+.
    CONF_THRESHOLD = 75.0

    y_true_lbph, y_pred_lbph = [], []
    for row in gt_rows:
        img_path  = test_dir / row["image_path"]
        true_name = row["true_identity"]   # "Unknown" for impostors
        if not img_path.exists():
            log.warning(f"  Missing test image: {img_path}")
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = fe.detect_faces(train_detector, img)
        if faces:
            x1, y1, x2, y2 = fe.bbox(faces[0])
            face_roi = gray[y1:y2, x1:x2]
            if face_roi.size == 0:
                face_roi = gray
        else:
            # Whole-image fallback for test images too
            face_roi = gray
        roi      = cv2.resize(face_roi, (100, 100))
        lbl, conf = lbph.predict(roi)
        pred = rev_map.get(lbl, "Unknown") if conf < CONF_THRESHOLD else "Unknown"
        y_true_lbph.append(true_name)
        y_pred_lbph.append(pred)
        log.debug(f"  {img_path.name}: true={true_name} pred={pred} conf={conf:.1f}")

    m_lbph = compute_metrics(y_true_lbph, y_pred_lbph, enrolled_names)
    log.info(
        f"  LBPH (conf<{CONF_THRESHOLD}):  "
        f"TAR={m_lbph['TAR']:.3f}  FAR={m_lbph['FAR']:.3f}  "
        f"FRR={m_lbph['FRR']:.3f}  ACC={m_lbph['ACC']:.3f}"
    )

    return [
        {
            "method": "LBPH baseline",
            "TAR": round(m_lbph["TAR"], 4),
            "FAR": round(m_lbph["FAR"], 4),
            "FRR": round(m_lbph["FRR"], 4),
            "ACC": round(m_lbph["ACC"], 4),
        },
    ]


# ── Lighting ablation from results CSV ────────────────────────────────────────
def _compute_lighting_ablation_from_results(all_rows, out_dir, log):
    log.info("\n--- Lighting Ablation from full results ---")
    from collections import defaultdict
    per_light = defaultdict(lambda: {"y_true": [], "y_pred": []})
    best = "D: Full two-gate"

    for row in all_rows:
        if row["config"] != best:
            continue
        light = row.get("lighting", "unknown")
        per_light[light]["y_true"].append(row["true_name"])
        per_light[light]["y_pred"].append(row["predicted"])

    rows = []
    for light, data in sorted(per_light.items()):
        correct = sum(1 for t, p in zip(data["y_true"], data["y_pred"]) if t == p)
        total   = len(data["y_true"])
        acc     = correct / total if total > 0 else 0
        rows.append({"lighting": light, "correct": correct, "total": total, "ACC": round(acc, 4)})
        log.info(f"  {light:<20}  ACC={acc:.3f}  ({correct}/{total})")

    save_csv(out_dir / "lighting_ablation.csv", rows, ["lighting","correct","total","ACC"])


# ── Plots ──────────────────────────────────────────────────────────────────────
def _plot_gate_comparison(gate_metrics: dict, out_dir: Path, log) -> None:
    apply_paper_style()
    import matplotlib.pyplot as plt

    configs = list(gate_metrics.keys())
    TAR = [gate_metrics[c]["TAR"] for c in configs]
    FAR = [gate_metrics[c]["FAR"] for c in configs]
    FRR = [gate_metrics[c]["FRR"] for c in configs]

    x   = np.arange(len(configs))
    w   = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w, TAR, w, label="TAR (↑ better)", color="#2196F3")
    ax.bar(x,     FAR, w, label="FAR (↓ better)", color="#F44336")
    ax.bar(x + w, FRR, w, label="FRR (↓ better)", color="#FF9800")

    ax.set_xticks(x)
    ax.set_xticklabels([c.split(":")[0] + ":" + c.split(":")[1][:15]
                        for c in configs], rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Rate")
    ax.set_title("Face Recognition: Gate Configuration Comparison")
    ax.legend()

    out = out_dir / "gate_comparison_bar.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Plot saved → {out}")


def _plot_lighting_ablation(light_rows: list, out_dir: Path, log) -> None:
    if not light_rows:
        return
    apply_paper_style()
    import matplotlib.pyplot as plt

    lights    = sorted(set(r["lighting"] for r in light_rows))
    norm_off  = {r["lighting"]: r["ACC"] for r in light_rows if r["normalize"] == "norm_OFF"}
    norm_on   = {r["lighting"]: r["ACC"] for r in light_rows if r["normalize"] == "norm_ON"}

    x = np.arange(len(lights))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, [norm_off.get(l, 0) for l in lights], w, label="Without CLAHE/Gamma", color="#90A4AE")
    ax.bar(x + w/2, [norm_on.get(l, 0)  for l in lights], w, label="With CLAHE/Gamma",    color="#42A5F5")

    ax.set_xticks(x)
    ax.set_xticklabels(lights, rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Lighting Robustness: Effect of CLAHE+Gamma Normalization")
    ax.legend()

    out = out_dir / "lighting_ablation_bar.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Plot saved → {out}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Phase 2 — Face Recognition Accuracy")
    parser.add_argument("--mode",     choices=["demo", "full", "auto"], default="auto",
                        help="demo=use known_faces, full=use test_faces, auto=pick based on data")
    parser.add_argument("--test-dir", type=Path, default=TEST_FACES,
                        help="Path to test_faces/ directory (full mode only)")
    args = parser.parse_args()

    log     = setup_logging("phase2_face_recognition")
    out_dir = get_results_dir("phase2")

    mode = args.mode
    if mode == "auto":
        gt = args.test_dir / "ground_truth.csv"
        mode = "full" if gt.exists() else "demo"
        log.info(f"Auto-selected mode: {mode}")

    if mode == "demo":
        run_demo(log, out_dir)
    else:
        run_full(log, out_dir, args.test_dir)


if __name__ == "__main__":
    main()
