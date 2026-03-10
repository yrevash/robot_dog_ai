#!/usr/bin/env python3
"""
experiments/sweep_threshold.py
================================
Phase 2.4 — ROC curve generation by sweeping threshold values.

Sweeps cosine-similarity thresholds from 0.20 to 0.70 to produce FAR/FRR/TAR
curves, identifies the Equal Error Rate (EER), and sweeps margin sensitivity
at a fixed threshold of 0.42.

Outputs (all under results/phase2/):
  roc_data.csv          — per-threshold FAR, FRR, TAR
  margin_sweep.csv      — FAR, FRR, TAR per margin at threshold=0.42
  roc_curve.png         — ROC plot (1-FRR vs FAR) with EER marked, one curve
                          per margin value
  threshold_sweep.png   — FAR and FRR vs threshold with EER crossover marked

Usage:
  python experiments/sweep_threshold.py
  python experiments/sweep_threshold.py --threshold-range 0.20 0.70 --step 0.02
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Bootstrap: ensure project root is importable ───────────────────────────────
# utils.py sets up sys.path; import it before anything else from the project.
_EXPERIMENTS = Path(__file__).resolve().parent
_PROJECT_ROOT = _EXPERIMENTS.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import (
    apply_paper_style,
    check_db_exists,
    get_results_dir,
    save_csv,
    setup_logging,
    DB_FILE,
    KNOWN_FACES,
)
import face_embedding as fe


# ── Synthetic impostor transforms (heavy — designed to defeat recognition) ─────
def _make_impostor(img: np.ndarray) -> np.ndarray:
    """
    Apply a stack of heavy transforms to an enrolled image to simulate an
    impostor: extreme brightness shift, heavy Gaussian blur, horizontal flip,
    and a channel colour-shift.  The result is plausible as a human face image
    but should look very different to the enrolled identity in embedding space.
    """
    out = img.copy()
    # Extreme brightness — push well outside training distribution
    out = cv2.convertScaleAbs(out, alpha=0.30, beta=120)
    # Heavy blur (kernel must be odd)
    k = 21
    out = cv2.GaussianBlur(out, (k, k), sigmaX=8)
    # Horizontal flip (different pose)
    out = cv2.flip(out, 1)
    # Colour channel shift
    shift = np.array([40, -30, 20], dtype=np.int32)
    out = np.clip(out.astype(np.int32) + shift, 0, 255).astype(np.uint8)
    return out


# ── Dataset builder ────────────────────────────────────────────────────────────
def _build_dataset(
    detector,
    recognizer,
    log,
) -> tuple[list[np.ndarray], list[str], list[np.ndarray], list[str]]:
    """
    Returns:
      enrolled_embs  — list of L2-normalised embeddings (positive class)
      enrolled_names — corresponding true identity strings
      impostor_embs  — embeddings of synthetic impostors (negative class, true="Unknown")
      impostor_names — all "Unknown"
    """
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    enrolled_embs: list[np.ndarray] = []
    enrolled_names: list[str] = []
    impostor_embs: list[np.ndarray] = []
    impostor_names: list[str] = []

    person_dirs = sorted(p for p in KNOWN_FACES.iterdir() if p.is_dir())
    if not person_dirs:
        log.error(f"No person folders found in {KNOWN_FACES}")
        return enrolled_embs, enrolled_names, impostor_embs, impostor_names

    for person_dir in person_dirs:
        name = person_dir.name
        images = sorted(p for p in person_dir.iterdir() if p.suffix.lower() in image_exts)
        if not images:
            log.warning(f"No images for {name}, skipping.")
            continue

        person_enrolled = 0
        person_impostors = 0
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                log.warning(f"Cannot read {img_path}")
                continue

            proc = fe.normalize_lighting(img)
            faces = fe.detect_faces(detector, proc)
            if not faces:
                continue

            emb = fe.embedding_from_face(proc, faces[0], recognizer)
            if emb is None:
                continue

            enrolled_embs.append(emb)
            enrolled_names.append(name)
            person_enrolled += 1

            # One synthetic impostor per enrolled image
            imp_img = _make_impostor(img)
            imp_faces = fe.detect_faces(detector, imp_img)
            if imp_faces:
                imp_emb = fe.embedding_from_face(imp_img, imp_faces[0], recognizer)
                if imp_emb is not None:
                    impostor_embs.append(imp_emb)
                    impostor_names.append("Unknown")
                    person_impostors += 1

        log.info(
            f"  {name}: {person_enrolled} enrolled embeddings, "
            f"{person_impostors} impostor embeddings"
        )

    return enrolled_embs, enrolled_names, impostor_embs, impostor_names


# ── Single-threshold evaluation ────────────────────────────────────────────────
def _evaluate_at_threshold(
    enrolled_embs: list[np.ndarray],
    enrolled_names: list[str],
    impostor_embs: list[np.ndarray],
    db_embeddings: np.ndarray,
    db_names: np.ndarray,
    centroids: np.ndarray,
    centroid_names: np.ndarray,
    threshold: float,
    margin: float,
) -> dict:
    """
    Run match_identity for every enrolled + impostor embedding at the given
    threshold and return a metrics dict with FAR, FRR, TAR.
    """
    centroid_threshold = threshold * 0.95

    TP = FN = FP = TN = 0
    enrolled_set = set(db_names.tolist())

    for emb, true_name in zip(enrolled_embs, enrolled_names):
        pred, _, _ = fe.match_identity(
            emb, db_embeddings, db_names, centroids, centroid_names,
            threshold=threshold, margin=margin,
            centroid_threshold=centroid_threshold,
        )
        if pred == true_name:
            TP += 1
        else:
            FN += 1

    for emb in impostor_embs:
        pred, _, _ = fe.match_identity(
            emb, db_embeddings, db_names, centroids, centroid_names,
            threshold=threshold, margin=margin,
            centroid_threshold=centroid_threshold,
        )
        if pred != "Unknown":
            FP += 1
        else:
            TN += 1

    total_enrolled = TP + FN
    total_impostor = FP + TN
    TAR = TP / total_enrolled  if total_enrolled  > 0 else 0.0
    FAR = FP / total_impostor  if total_impostor  > 0 else 0.0
    FRR = FN / total_enrolled  if total_enrolled  > 0 else 0.0

    return dict(TAR=TAR, FAR=FAR, FRR=FRR, TP=TP, FP=FP, FN=FN, TN=TN)


# ── EER finder ─────────────────────────────────────────────────────────────────
def _find_eer(thresholds: list[float], far_vals: list[float], frr_vals: list[float]) -> tuple[float, float, float]:
    """
    Find EER as the threshold where |FAR - FRR| is minimised.
    Returns (eer_threshold, eer_far, eer_frr).
    """
    diffs = [abs(f - r) for f, r in zip(far_vals, frr_vals)]
    idx = int(np.argmin(diffs))
    return thresholds[idx], far_vals[idx], frr_vals[idx]


# ── Plots ───────────────────────────────────────────────────────────────────────
def _plot_threshold_sweep(
    thresholds: list[float],
    far_vals: list[float],
    frr_vals: list[float],
    eer_thr: float,
    eer_rate: float,
    out_dir: Path,
    log,
) -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(thresholds, far_vals, color="#F44336", label="FAR (False Accept Rate)")
    ax.plot(thresholds, frr_vals, color="#2196F3", label="FRR (False Reject Rate)")

    ax.axvline(eer_thr, color="gray", linestyle="--", linewidth=1.4, label=f"EER @ {eer_thr:.2f}")
    ax.scatter([eer_thr], [eer_rate], color="black", zorder=5, s=60)
    ax.annotate(
        f"EER={eer_rate:.3f}",
        xy=(eer_thr, eer_rate),
        xytext=(eer_thr + 0.03, eer_rate + 0.04),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="black"),
    )

    ax.set_xlabel("Cosine Similarity Threshold")
    ax.set_ylabel("Error Rate")
    ax.set_title("FAR and FRR vs Threshold (margin=0.06)")
    ax.set_xlim(min(thresholds), max(thresholds))
    ax.set_ylim(-0.02, 1.05)
    ax.legend()

    out = out_dir / "threshold_sweep.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Plot saved -> {out}")


def _plot_roc_curve(
    margin_results: dict[float, dict],
    eer_thr: float,
    eer_far: float,
    eer_frr: float,
    out_dir: Path,
    log,
) -> None:
    """
    ROC plot: x=FAR, y=TAR (= 1-FRR).  One curve per margin value.
    EER is marked on the margin=0.06 curve.
    """
    apply_paper_style()

    MARGIN_COLORS = {
        0.00: "#9C27B0",
        0.03: "#2196F3",
        0.06: "#4CAF50",
        0.09: "#FF9800",
    }

    fig, ax = plt.subplots(figsize=(7, 6))

    # Diagonal reference line (random classifier)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4, label="Random")

    for margin, data in sorted(margin_results.items()):
        far_v = data["far"]
        tar_v = [1.0 - r for r in data["frr"]]
        color = MARGIN_COLORS.get(margin, "gray")
        ax.plot(far_v, tar_v, color=color, label=f"margin={margin:.2f}")

    # Mark EER on the default margin=0.06 curve
    eer_tar = 1.0 - eer_frr
    ax.scatter([eer_far], [eer_tar], color="black", zorder=6, s=70, marker="*")
    ax.annotate(
        f"EER ({eer_far:.3f}, {eer_tar:.3f})",
        xy=(eer_far, eer_tar),
        xytext=(eer_far + 0.05, eer_tar - 0.08),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="black"),
    )

    ax.set_xlabel("FAR (False Accept Rate)")
    ax.set_ylabel("TAR (True Accept Rate = 1 - FRR)")
    ax.set_title("ROC Curve — Threshold Sweep by Margin Value")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="lower right")

    out = out_dir / "roc_curve.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Plot saved -> {out}")


# ── Main ────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2.4 — ROC curve via threshold sweep"
    )
    parser.add_argument(
        "--threshold-range", nargs=2, type=float, metavar=("LO", "HI"),
        default=[0.20, 0.70],
        help="Threshold sweep range (default: 0.20 0.70)",
    )
    parser.add_argument(
        "--step", type=float, default=0.02,
        help="Threshold step size (default: 0.02)",
    )
    args = parser.parse_args()

    log = setup_logging("phase2_threshold_sweep")
    out_dir = get_results_dir("phase2")

    # ── Sanity-checks ──────────────────────────────────────────────────────────
    if not check_db_exists(log):
        sys.exit(1)

    if not KNOWN_FACES.exists() or not any(KNOWN_FACES.iterdir()):
        log.error(f"known_faces/ is empty or missing at {KNOWN_FACES}")
        log.error("Run: python face_embedding.py capture --name <Name>")
        sys.exit(1)

    # ── Load models and DB ─────────────────────────────────────────────────────
    log.info("Checking OpenCV requirements and ONNX models...")
    fe.check_opencv_requirements()
    fe.ensure_models()

    log.info(f"Loading database: {DB_FILE}")
    db_embeddings, db_names, centroids, centroid_names = fe.load_db(DB_FILE)
    enrolled_set = set(db_names.tolist())
    log.info(f"DB: {len(db_embeddings)} embeddings, {len(enrolled_set)} identities: {sorted(enrolled_set)}")

    # ── Build dataset (enrolled + synthetic impostors) ─────────────────────────
    log.info("Building evaluation dataset from known_faces/ ...")
    detector   = fe.create_detector((640, 480), score_threshold=0.85)
    recognizer = fe.create_recognizer()

    enrolled_embs, enrolled_names, impostor_embs, impostor_names = _build_dataset(
        detector, recognizer, log
    )

    if not enrolled_embs:
        log.error("No enrolled embeddings extracted. Cannot run sweep.")
        sys.exit(1)

    if not impostor_embs:
        log.warning(
            "WARNING: zero impostor embeddings extracted — the synthetic impostor "
            "transform produced images where no face was detected by YuNet. "
            "All FAR values will be 0.0 (misleadingly perfect). "
            "Consider loosening the detector score_threshold or using a different "
            "impostor generation strategy."
        )

    log.info(
        f"Dataset ready: {len(enrolled_embs)} enrolled, {len(impostor_embs)} impostors"
    )

    # ── Threshold sweep (margin=0.06 fixed) ───────────────────────────────────
    lo, hi = args.threshold_range
    n_steps = max(1, round((hi - lo) / args.step))
    thresholds = [round(lo + i * args.step, 4) for i in range(n_steps + 1)]
    thresholds = [t for t in thresholds if t <= hi + 1e-9]

    FIXED_MARGIN = 0.06
    log.info(
        f"\nSweeping threshold {lo:.2f} -> {hi:.2f} "
        f"(step={args.step}, margin={FIXED_MARGIN})"
    )

    roc_rows: list[dict] = []
    far_vals: list[float] = []
    frr_vals: list[float] = []

    for thr in thresholds:
        m = _evaluate_at_threshold(
            enrolled_embs, enrolled_names, impostor_embs,
            db_embeddings, db_names, centroids, centroid_names,
            threshold=thr, margin=FIXED_MARGIN,
        )
        far_vals.append(m["FAR"])
        frr_vals.append(m["FRR"])
        roc_rows.append({
            "threshold": round(thr, 4),
            "FAR":       round(m["FAR"], 5),
            "FRR":       round(m["FRR"], 5),
            "TAR":       round(m["TAR"], 5),
        })
        log.info(
            f"  thr={thr:.2f}  FAR={m['FAR']:.4f}  FRR={m['FRR']:.4f}  TAR={m['TAR']:.4f}"
        )

    save_csv(out_dir / "roc_data.csv", roc_rows, ["threshold", "FAR", "FRR", "TAR"])
    log.info(f"Saved -> {out_dir / 'roc_data.csv'}")

    # EER for default margin
    eer_thr, eer_far, eer_frr = _find_eer(thresholds, far_vals, frr_vals)
    log.info(
        f"\nEER (margin={FIXED_MARGIN}): threshold={eer_thr:.2f}  "
        f"FAR={eer_far:.4f}  FRR={eer_frr:.4f}  EER={(eer_far+eer_frr)/2:.4f}"
    )

    # ── Margin sweep at fixed threshold=0.42 ──────────────────────────────────
    MARGINS = [0.00, 0.03, 0.06, 0.09]
    FIXED_THRESHOLD = 0.42
    log.info(f"\nSweeping margin {MARGINS} at fixed threshold={FIXED_THRESHOLD}")

    margin_rows: list[dict] = []
    margin_results: dict[float, dict] = {}   # for ROC plot

    # Re-run threshold sweep per margin (for ROC curves)
    for margin in MARGINS:
        m_far: list[float] = []
        m_frr: list[float] = []

        for thr in thresholds:
            m = _evaluate_at_threshold(
                enrolled_embs, enrolled_names, impostor_embs,
                db_embeddings, db_names, centroids, centroid_names,
                threshold=thr, margin=margin,
            )
            m_far.append(m["FAR"])
            m_frr.append(m["FRR"])

        margin_results[margin] = {"far": m_far, "frr": m_frr}

        # Single point at fixed threshold for the CSV
        m_fixed = _evaluate_at_threshold(
            enrolled_embs, enrolled_names, impostor_embs,
            db_embeddings, db_names, centroids, centroid_names,
            threshold=FIXED_THRESHOLD, margin=margin,
        )
        margin_rows.append({
            "margin": margin,
            "FAR":    round(m_fixed["FAR"], 5),
            "FRR":    round(m_fixed["FRR"], 5),
            "TAR":    round(m_fixed["TAR"], 5),
        })
        log.info(
            f"  margin={margin:.2f}  FAR={m_fixed['FAR']:.4f}  "
            f"FRR={m_fixed['FRR']:.4f}  TAR={m_fixed['TAR']:.4f}"
        )

    save_csv(out_dir / "margin_sweep.csv", margin_rows, ["margin", "FAR", "FRR", "TAR"])
    log.info(f"Saved -> {out_dir / 'margin_sweep.csv'}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    log.info("\nGenerating plots...")

    _plot_threshold_sweep(
        thresholds, far_vals, frr_vals,
        eer_thr=eer_thr, eer_rate=(eer_far + eer_frr) / 2,
        out_dir=out_dir, log=log,
    )

    _plot_roc_curve(
        margin_results=margin_results,
        eer_thr=eer_thr, eer_far=eer_far, eer_frr=eer_frr,
        out_dir=out_dir, log=log,
    )

    log.info("\n" + "=" * 60)
    log.info("Phase 2.4 complete. Outputs in results/phase2/:")
    log.info("  roc_data.csv          — threshold vs FAR/FRR/TAR")
    log.info("  margin_sweep.csv      — margin sensitivity at thr=0.42")
    log.info("  threshold_sweep.png   — FAR+FRR vs threshold with EER")
    log.info("  roc_curve.png         — ROC curves per margin value")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
