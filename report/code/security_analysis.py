#!/usr/bin/env python3
"""
Phase 7 — Security Analysis
============================
Formally documents the security properties and known limitations of the
REVO face-recognition pipeline.  The system has NO liveness detection —
a printed photograph of an enrolled person is indistinguishable from a
live face in embedding space.

Attack scenarios tested
-----------------------
  A1  Replay / photo-attack simulation
        Enrolled test-set images (data/test_faces/enrolled/) that were NOT
        used to build face_db.npz serve as proxy printed-photo attacks.
        Because the embedding model is deterministic and has no temporal /
        depth cue, a static image == a printed photo from the model's
        perspective.

  A2  Cross-identity confusion (near-threshold probe)
        Harshhini impostor images scored against the Yash/Aramaan DB.
        Checks whether any image scores above the 0.40 near-threshold
        warning band.

  A3  Threshold stress test (security margin)
        For each enrolled training image (known_faces/), how far above
        0.42 does the best-sample score sit?  A small margin means the
        system is close to its own operating limit.

  A4  Centroid gate robustness
        Compare centroid similarity scores for enrolled vs impostor images.

  A5  Single-frame spoof (summary from Phase 2/3)
        Without temporal voting a single impostor frame would be accepted
        iff its cosine score exceeds 0.42.  Answered by Phase 2 data —
        summarised here, no new inference needed.

OUTPUTS
-------
  results/phase7/security_summary.csv
  results/phase7/score_distribution.png
  results/phase7/security_margins.csv
  results/logs/phase7_*.log
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import (
    setup_logging, get_results_dir, save_csv,
    apply_paper_style,
    DB_FILE, KNOWN_FACES, TEST_FACES,
)
import face_embedding as fe

# ── Thresholds (production defaults) ──────────────────────────────────────────
THRESHOLD         = 0.42   # Gate 1: cosine similarity
MARGIN            = 0.06   # Gate 1: margin over second-best
CENTROID_THRESH   = 0.40   # Gate 2: centroid similarity
NEAR_THRESH_WARN  = 0.40   # Warning band — impostors scoring above this are "close"


# ── Helper: embed one image file ───────────────────────────────────────────────
def embed_image(
    img_path: Path,
    detector,
    recognizer,
    normalize: bool = True,
) -> tuple[np.ndarray | None, str]:
    """
    Load an image from disk, detect a face, extract SFace embedding.
    Returns (embedding_or_None, status_string).
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None, "load_failed"

    proc = fe.normalize_lighting(img) if normalize else img
    faces = fe.detect_faces(detector, proc)
    if not faces:
        return None, "no_face"

    emb = fe.embedding_from_face(proc, faces[0], recognizer)
    if emb is None:
        return None, "embed_failed"

    return emb, "ok"


# ── Helper: cosine similarity against DB without full two-gate logic ───────────
def best_sample_score(emb: np.ndarray, db_embeddings: np.ndarray, db_names: np.ndarray):
    """Return (best_score, candidate_name, second_score)."""
    sims = (db_embeddings @ emb).astype(np.float32)
    unique_names, inverse = np.unique(db_names, return_inverse=True)
    best_per = np.full(len(unique_names), -1.0, dtype=np.float32)
    np.maximum.at(best_per, inverse, sims)

    best_idx   = int(np.argmax(best_per))
    best_score = float(best_per[best_idx])
    cand_name  = str(unique_names[best_idx])

    if len(best_per) > 1:
        second = float(np.partition(best_per, -2)[-2])
    else:
        second = -1.0

    return best_score, cand_name, second


def centroid_score(emb: np.ndarray, centroids: np.ndarray, centroid_names: np.ndarray):
    """Return (best_centroid_score, centroid_name)."""
    c_sims = centroids @ emb
    idx    = int(np.argmax(c_sims))
    return float(c_sims[idx]), str(centroid_names[idx])


# ── Attack A1: Replay / photo-attack simulation ────────────────────────────────
def attack_a1_replay(
    db_embs, db_names, centroids, centroid_names,
    detector, recognizer, log,
) -> tuple[list, list]:
    """
    Test enrolled test-set images (NOT used for DB construction) through the
    full two-gate matcher.  These are the closest proxy for a printed-photo
    attack available without physical props.

    Returns (summary_row, margin_rows).
    """
    log.info("\n=== A1: Replay / Photo-Attack Simulation ===")

    enrolled_dir = TEST_FACES / "enrolled"
    if not enrolled_dir.exists():
        log.warning(f"  test_faces/enrolled/ not found at {enrolled_dir}. Skipping A1.")
        return [{"attack_type": "A1_replay", "n_tested": 0,
                 "n_succeeded": 0, "success_rate": "N/A",
                 "note": "test_faces/enrolled/ missing"}], []

    image_paths = []
    for person_dir in sorted(enrolled_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                image_paths.append((img_path, person_dir.name))

    if not image_paths:
        log.warning("  No images found in test_faces/enrolled/. Skipping A1.")
        return [{"attack_type": "A1_replay", "n_tested": 0,
                 "n_succeeded": 0, "success_rate": "N/A",
                 "note": "no images"}], []

    log.info(f"  Images to test: {len(image_paths)}")

    n_tested = 0
    n_accepted = 0
    margin_rows = []

    for img_path, true_name in image_paths:
        emb, status = embed_image(img_path, detector, recognizer)
        if status != "ok":
            log.debug(f"  SKIP {img_path.name} ({true_name}): {status}")
            continue

        n_tested += 1
        pred, s_score, c_score_val = fe.match_identity(
            emb, db_embs, db_names, centroids, centroid_names,
            threshold=THRESHOLD, margin=MARGIN, centroid_threshold=CENTROID_THRESH,
        )
        accepted = (pred != "Unknown")
        if accepted:
            n_accepted += 1

        b_score, cand, second = best_sample_score(emb, db_embs, db_names)
        c_sc, c_name = centroid_score(emb, centroids, centroid_names)
        margin_above = b_score - THRESHOLD

        log.info(
            f"  {img_path.name:<12} true={true_name:<10} "
            f"pred={pred:<10} score={b_score:.4f} "
            f"margin_above_thresh={margin_above:+.4f} "
            f"centroid={c_sc:.4f} accepted={accepted}"
        )
        margin_rows.append({
            "attack":          "A1_replay",
            "image":           img_path.name,
            "true_name":       true_name,
            "predicted":       pred,
            "sample_score":    round(b_score, 4),
            "second_score":    round(second, 4),
            "margin_vs_second": round(b_score - second, 4),
            "margin_above_threshold": round(margin_above, 4),
            "centroid_score":  round(c_sc, 4),
            "centroid_name":   c_name,
            "accepted":        accepted,
        })

    success_rate = n_accepted / n_tested if n_tested > 0 else 0.0
    log.info(f"  RESULT: {n_accepted}/{n_tested} test-set images accepted "
             f"({success_rate:.1%})")
    log.info("  Interpretation: accepted rate for enrolled test images == photo-attack "
             "success rate (no liveness detection)")

    summary = {
        "attack_type":   "A1_replay",
        "n_tested":      n_tested,
        "n_succeeded":   n_accepted,
        "success_rate":  round(success_rate, 4),
        "note":          "photo-attack proxy: test-set enrolled images, same identity as DB",
    }
    return [summary], margin_rows


# ── Attack A2: Cross-identity confusion ───────────────────────────────────────
def attack_a2_cross_identity(
    db_embs, db_names, centroids, centroid_names,
    detector, recognizer, log,
) -> tuple[list, list]:
    """
    Score Harshhini impostor images against the Yash/Aramaan DB.
    Flag any image scoring above NEAR_THRESH_WARN (0.40).
    """
    log.info("\n=== A2: Cross-Identity Confusion (Harshhini impostor) ===")

    impostor_dir = TEST_FACES / "impostors" / "Harshhini"
    if not impostor_dir.exists():
        log.warning(f"  Impostor dir not found: {impostor_dir}. Skipping A2.")
        return [{"attack_type": "A2_cross_identity", "n_tested": 0,
                 "n_succeeded": 0, "success_rate": "N/A",
                 "note": "impostor dir missing"}], []

    image_paths = sorted(
        p for p in impostor_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not image_paths:
        log.warning("  No impostor images found. Skipping A2.")
        return [{"attack_type": "A2_cross_identity", "n_tested": 0,
                 "n_succeeded": 0, "success_rate": "N/A",
                 "note": "no images"}], []

    log.info(f"  Impostor images: {len(image_paths)}")

    n_tested  = 0
    n_above   = 0        # above near-threshold warning (0.40)
    n_accepted = 0       # above actual threshold (0.42) AND pass both gates
    margin_rows = []

    for img_path in image_paths:
        emb, status = embed_image(img_path, detector, recognizer)
        if status != "ok":
            log.debug(f"  SKIP {img_path.name}: {status}")
            continue

        n_tested += 1
        b_score, cand, second = best_sample_score(emb, db_embs, db_names)
        c_sc, c_name = centroid_score(emb, centroids, centroid_names)

        pred, s_score, c_score_val = fe.match_identity(
            emb, db_embs, db_names, centroids, centroid_names,
            threshold=THRESHOLD, margin=MARGIN, centroid_threshold=CENTROID_THRESH,
        )
        accepted = (pred != "Unknown")
        near_threshold = (b_score >= NEAR_THRESH_WARN)

        if near_threshold:
            n_above += 1
        if accepted:
            n_accepted += 1

        log.info(
            f"  {img_path.name:<12} best_score={b_score:.4f} "
            f"candidate={cand:<10} centroid={c_sc:.4f} "
            f"near_thresh={near_threshold} accepted={accepted}"
        )
        margin_rows.append({
            "attack":          "A2_cross_identity",
            "image":           img_path.name,
            "true_name":       "Harshhini",
            "predicted":       pred,
            "sample_score":    round(b_score, 4),
            "second_score":    round(second, 4),
            "margin_vs_second": round(b_score - second, 4),
            "margin_above_threshold": round(b_score - THRESHOLD, 4),
            "centroid_score":  round(c_sc, 4),
            "centroid_name":   c_name,
            "accepted":        accepted,
        })

    near_rate = n_above   / n_tested if n_tested > 0 else 0.0
    acc_rate  = n_accepted / n_tested if n_tested > 0 else 0.0
    log.info(
        f"  RESULT: {n_accepted}/{n_tested} accepted by full two-gate "
        f"({acc_rate:.1%});  {n_above}/{n_tested} above 0.40 near-threshold "
        f"warning ({near_rate:.1%})"
    )

    summaries = [
        {
            "attack_type":  "A2_cross_identity_near_thresh",
            "n_tested":     n_tested,
            "n_succeeded":  n_above,
            "success_rate": round(near_rate, 4),
            "note":         "Harshhini images scoring above 0.40 near-threshold warn",
        },
        {
            "attack_type":  "A2_cross_identity_accepted",
            "n_tested":     n_tested,
            "n_succeeded":  n_accepted,
            "success_rate": round(acc_rate, 4),
            "note":         "Harshhini images accepted by full two-gate matcher",
        },
    ]
    return summaries, margin_rows


# ── Attack A3: Threshold stress test (enrolled training images) ────────────────
def attack_a3_stress_test(
    db_embs, db_names, centroids, centroid_names,
    detector, recognizer, log,
) -> tuple[list, list]:
    """
    For each known_faces/ training image, report how far above 0.42 the
    best-sample score sits.  Low margins mean the system is operating close
    to its own limit and small perturbations (lighting, pose) could cause
    false rejects.
    """
    log.info("\n=== A3: Threshold Stress Test (enrolled training images) ===")

    image_paths = []
    for person_dir in sorted(KNOWN_FACES.iterdir()):
        if not person_dir.is_dir():
            continue
        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                image_paths.append((img_path, person_dir.name))

    if not image_paths:
        log.warning("  No training images found in known_faces/. Skipping A3.")
        return [{"attack_type": "A3_stress_test", "n_tested": 0,
                 "n_succeeded": 0, "success_rate": "N/A",
                 "note": "no images"}], []

    log.info(f"  Training images: {len(image_paths)}")

    margins = []
    margin_rows = []
    n_tested   = 0
    n_above    = 0

    for img_path, true_name in image_paths:
        emb, status = embed_image(img_path, detector, recognizer)
        if status != "ok":
            log.debug(f"  SKIP {img_path.name} ({true_name}): {status}")
            continue

        n_tested += 1
        b_score, cand, second = best_sample_score(emb, db_embs, db_names)
        c_sc, c_name = centroid_score(emb, centroids, centroid_names)
        margin_above = b_score - THRESHOLD

        if b_score >= THRESHOLD:
            n_above += 1

        margins.append(margin_above)
        margin_rows.append({
            "attack":           "A3_stress_test",
            "image":            img_path.name,
            "true_name":        true_name,
            "predicted":        cand,
            "sample_score":     round(b_score, 4),
            "second_score":     round(second, 4),
            "margin_vs_second": round(b_score - second, 4),
            "margin_above_threshold": round(margin_above, 4),
            "centroid_score":   round(c_sc, 4),
            "centroid_name":    c_name,
            "accepted":         b_score >= THRESHOLD,
        })
        log.debug(
            f"  {img_path.name:<12} ({true_name:<10}) "
            f"score={b_score:.4f}  margin={margin_above:+.4f}  centroid={c_sc:.4f}"
        )

    if margins:
        log.info(f"  Min margin above threshold : {min(margins):+.4f}")
        log.info(f"  Mean margin above threshold: {np.mean(margins):+.4f}")
        log.info(f"  Max margin above threshold : {max(margins):+.4f}")
        log.info(f"  Images above threshold     : {n_above}/{n_tested}")

    summary = {
        "attack_type":  "A3_stress_test_enrolled_training",
        "n_tested":     n_tested,
        "n_succeeded":  n_above,
        "success_rate": round(n_above / n_tested, 4) if n_tested > 0 else 0.0,
        "note": (f"enrolled training images above 0.42 threshold; "
                 f"min_margin={min(margins):+.4f} "
                 f"mean_margin={np.mean(margins):+.4f}") if margins else "no data",
    }
    return [summary], margin_rows


# ── Attack A4: Centroid gate robustness ───────────────────────────────────────
def attack_a4_centroid(
    db_embs, db_names, centroids, centroid_names,
    detector, recognizer, log,
) -> tuple[list, list]:
    """
    Compare centroid similarity for enrolled vs impostor images.
    A robust centroid gate should show a clear separation between the two
    distributions.
    """
    log.info("\n=== A4: Centroid Gate Robustness ===")

    enrolled_centroid_scores  = []
    impostor_centroid_scores  = []
    margin_rows = []

    # Enrolled: known_faces/
    for person_dir in sorted(KNOWN_FACES.iterdir()):
        if not person_dir.is_dir():
            continue
        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            emb, status = embed_image(img_path, detector, recognizer)
            if status != "ok":
                continue
            b_score, cand, second = best_sample_score(emb, db_embs, db_names)
            c_sc, c_name = centroid_score(emb, centroids, centroid_names)
            enrolled_centroid_scores.append(c_sc)
            margin_rows.append({
                "attack":           "A4_centroid_enrolled",
                "image":            img_path.name,
                "true_name":        person_dir.name,
                "predicted":        cand,
                "sample_score":     round(b_score, 4),
                "second_score":     round(second, 4),
                "margin_vs_second": round(b_score - second, 4),
                "margin_above_threshold": round(b_score - THRESHOLD, 4),
                "centroid_score":   round(c_sc, 4),
                "centroid_name":    c_name,
                "accepted":         b_score >= THRESHOLD and c_sc >= CENTROID_THRESH,
            })

    # Impostor: test_faces/impostors/Harshhini/
    impostor_dir = TEST_FACES / "impostors" / "Harshhini"
    if impostor_dir.exists():
        for img_path in sorted(
            p for p in impostor_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ):
            emb, status = embed_image(img_path, detector, recognizer)
            if status != "ok":
                continue
            b_score, cand, second = best_sample_score(emb, db_embs, db_names)
            c_sc, c_name = centroid_score(emb, centroids, centroid_names)
            impostor_centroid_scores.append(c_sc)
            margin_rows.append({
                "attack":           "A4_centroid_impostor",
                "image":            img_path.name,
                "true_name":        "Harshhini",
                "predicted":        cand,
                "sample_score":     round(b_score, 4),
                "second_score":     round(second, 4),
                "margin_vs_second": round(b_score - second, 4),
                "margin_above_threshold": round(b_score - THRESHOLD, 4),
                "centroid_score":   round(c_sc, 4),
                "centroid_name":    c_name,
                "accepted":         False,
            })

    if enrolled_centroid_scores:
        log.info(f"  Enrolled centroid scores:  "
                 f"min={min(enrolled_centroid_scores):.4f}  "
                 f"mean={np.mean(enrolled_centroid_scores):.4f}  "
                 f"max={max(enrolled_centroid_scores):.4f}  "
                 f"n={len(enrolled_centroid_scores)}")
    if impostor_centroid_scores:
        log.info(f"  Impostor centroid scores:  "
                 f"min={min(impostor_centroid_scores):.4f}  "
                 f"mean={np.mean(impostor_centroid_scores):.4f}  "
                 f"max={max(impostor_centroid_scores):.4f}  "
                 f"n={len(impostor_centroid_scores)}")

    if enrolled_centroid_scores and impostor_centroid_scores:
        gap = min(enrolled_centroid_scores) - max(impostor_centroid_scores)
        log.info(f"  Centroid separation gap (min_enrolled - max_impostor): {gap:+.4f}")

    summaries = [
        {
            "attack_type":  "A4_centroid_enrolled_mean",
            "n_tested":     len(enrolled_centroid_scores),
            "n_succeeded":  sum(1 for s in enrolled_centroid_scores if s >= CENTROID_THRESH),
            "success_rate": round(
                sum(1 for s in enrolled_centroid_scores if s >= CENTROID_THRESH)
                / len(enrolled_centroid_scores), 4
            ) if enrolled_centroid_scores else 0.0,
            "note": (f"enrolled images with centroid_score >= {CENTROID_THRESH}; "
                     f"mean={np.mean(enrolled_centroid_scores):.4f}") if enrolled_centroid_scores else "no data",
        },
        {
            "attack_type":  "A4_centroid_impostor_above_thresh",
            "n_tested":     len(impostor_centroid_scores),
            "n_succeeded":  sum(1 for s in impostor_centroid_scores if s >= CENTROID_THRESH),
            "success_rate": round(
                sum(1 for s in impostor_centroid_scores if s >= CENTROID_THRESH)
                / len(impostor_centroid_scores), 4
            ) if impostor_centroid_scores else 0.0,
            "note": (f"impostor images with centroid_score >= {CENTROID_THRESH}; "
                     f"mean={np.mean(impostor_centroid_scores):.4f}") if impostor_centroid_scores else "no data",
        },
    ]
    return summaries, margin_rows, enrolled_centroid_scores, impostor_centroid_scores


# ── Attack A5: Single-frame spoof summary (from Phase 2 data) ─────────────────
def attack_a5_single_frame_summary(log) -> list:
    """
    Summarise the single-frame spoof vulnerability from Phase 2 findings.
    No new inference needed — Phase 2 measured that max Harshhini score = 0.363.
    """
    log.info("\n=== A5: Single-Frame Spoof (Phase 2 summary) ===")
    log.info("  Source: Phase 2 gate comparison (N=29, 13 Harshhini impostor images).")
    log.info("  Max impostor cosine similarity: 0.363  (threshold = 0.42)")
    log.info("  Min enrolled cosine similarity: 0.661")
    log.info("  Result: With voting disabled (single-frame), NO Harshhini image would")
    log.info("  have been accepted because max impostor score (0.363) < threshold (0.42).")
    log.info("  HOWEVER: this is specific to the Harshhini identity.  A visually similar")
    log.info("  impostor or morphed image could score above 0.42 on a single frame and")
    log.info("  be accepted immediately (no liveness, no voting required).")

    return [
        {
            "attack_type":  "A5_single_frame_known_impostor",
            "n_tested":     13,
            "n_succeeded":  0,
            "success_rate": 0.0,
            "note": ("Phase 2: max Harshhini score=0.363 < threshold 0.42. "
                     "Known impostor cannot spoof in single frame. "
                     "Unknown visually-similar impostor NOT tested."),
        }
    ]


# ── Score distribution plot ────────────────────────────────────────────────────
def plot_score_distribution(
    all_margin_rows: list,
    enrolled_centroid: list,
    impostor_centroid: list,
    out_dir: Path,
    log,
) -> None:
    apply_paper_style()

    enrolled_scores  = [r["sample_score"] for r in all_margin_rows
                        if r["true_name"] in {"Yash", "Aramaan"}]
    impostor_scores  = [r["sample_score"] for r in all_margin_rows
                        if r["true_name"] == "Harshhini"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Left: sample cosine score histogram ──────────────────────────────────
    ax = axes[0]
    bins = np.linspace(0, 1, 30)

    if enrolled_scores:
        ax.hist(enrolled_scores, bins=bins, alpha=0.65, color="#2196F3",
                label=f"Enrolled (n={len(enrolled_scores)})", density=False)
    if impostor_scores:
        ax.hist(impostor_scores, bins=bins, alpha=0.65, color="#F44336",
                label=f"Impostor/Harshhini (n={len(impostor_scores)})", density=False)

    ax.axvline(THRESHOLD, color="black", linestyle="--", linewidth=1.5,
               label=f"Threshold = {THRESHOLD}")
    ax.axvline(NEAR_THRESH_WARN, color="orange", linestyle=":", linewidth=1.2,
               label=f"Near-threshold = {NEAR_THRESH_WARN}")
    ax.set_xlabel("Best-sample cosine similarity")
    ax.set_ylabel("Count")
    ax.set_title("Sample Score Distribution\n(Enrolled vs Impostor)")
    ax.legend(fontsize=9)

    # ── Right: centroid score histogram ──────────────────────────────────────
    ax = axes[1]
    if enrolled_centroid:
        ax.hist(enrolled_centroid, bins=bins, alpha=0.65, color="#4CAF50",
                label=f"Enrolled centroids (n={len(enrolled_centroid)})", density=False)
    if impostor_centroid:
        ax.hist(impostor_centroid, bins=bins, alpha=0.65, color="#FF5722",
                label=f"Impostor centroids (n={len(impostor_centroid)})", density=False)

    ax.axvline(CENTROID_THRESH, color="black", linestyle="--", linewidth=1.5,
               label=f"Centroid threshold = {CENTROID_THRESH}")
    ax.set_xlabel("Centroid cosine similarity")
    ax.set_ylabel("Count")
    ax.set_title("Centroid Score Distribution\n(Enrolled vs Impostor)")
    ax.legend(fontsize=9)

    fig.suptitle("Phase 7 — Security Analysis: Score Distributions", fontsize=13, y=1.01)

    out = out_dir / "score_distribution.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Plot saved -> {out}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    log     = setup_logging("phase7_security_analysis")
    out_dir = get_results_dir("phase7")

    log.info("Phase 7 — Security Analysis")
    log.info(f"  DB file      : {DB_FILE}")
    log.info(f"  Threshold    : {THRESHOLD}  Margin: {MARGIN}  Centroid: {CENTROID_THRESH}")
    log.info(f"  Output dir   : {out_dir}")

    # ── Load DB ──────────────────────────────────────────────────────────────
    if not DB_FILE.exists():
        log.error(f"face_db.npz not found at {DB_FILE}. Run: python face_embedding.py build")
        sys.exit(1)

    fe.check_opencv_requirements()
    fe.ensure_models()

    db_embs, db_names, centroids, centroid_names = fe.load_db(DB_FILE)
    enrolled_names = sorted(set(db_names.tolist()))
    log.info(f"  DB loaded: {len(db_embs)} embeddings, identities: {enrolled_names}")

    detector   = fe.create_detector((640, 480), score_threshold=0.85)
    recognizer = fe.create_recognizer()

    # ── Run attacks ──────────────────────────────────────────────────────────
    all_summary_rows  = []
    all_margin_rows   = []

    # A1
    rows_a1, margins_a1 = attack_a1_replay(
        db_embs, db_names, centroids, centroid_names, detector, recognizer, log
    )
    all_summary_rows.extend(rows_a1)
    all_margin_rows.extend(margins_a1)

    # A2
    rows_a2, margins_a2 = attack_a2_cross_identity(
        db_embs, db_names, centroids, centroid_names, detector, recognizer, log
    )
    all_summary_rows.extend(rows_a2)
    all_margin_rows.extend(margins_a2)

    # A3
    rows_a3, margins_a3 = attack_a3_stress_test(
        db_embs, db_names, centroids, centroid_names, detector, recognizer, log
    )
    all_summary_rows.extend(rows_a3)
    all_margin_rows.extend(margins_a3)

    # A4
    rows_a4, margins_a4, enrolled_centroid, impostor_centroid = attack_a4_centroid(
        db_embs, db_names, centroids, centroid_names, detector, recognizer, log
    )
    all_summary_rows.extend(rows_a4)
    all_margin_rows.extend(margins_a4)

    # A5
    rows_a5 = attack_a5_single_frame_summary(log)
    all_summary_rows.extend(rows_a5)

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    summary_fields = ["attack_type", "n_tested", "n_succeeded", "success_rate", "note"]
    save_csv(out_dir / "security_summary.csv", all_summary_rows, summary_fields)
    log.info(f"\nSummary CSV saved -> {out_dir / 'security_summary.csv'}")

    margin_fields = [
        "attack", "image", "true_name", "predicted",
        "sample_score", "second_score", "margin_vs_second",
        "margin_above_threshold", "centroid_score", "centroid_name", "accepted",
    ]
    save_csv(out_dir / "security_margins.csv", all_margin_rows, margin_fields)
    log.info(f"Margins CSV saved -> {out_dir / 'security_margins.csv'}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    log.info("\nGenerating score distribution plot...")
    plot_score_distribution(all_margin_rows, enrolled_centroid, impostor_centroid,
                            out_dir, log)

    # ── Final summary ─────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("PHASE 7 SECURITY ANALYSIS — SUMMARY")
    log.info("=" * 60)
    for row in all_summary_rows:
        rate = row["success_rate"]
        rate_str = f"{float(rate):.1%}" if rate != "N/A" else "N/A"
        log.info(
            f"  {row['attack_type']:<45} "
            f"n={row['n_tested']:>4}  "
            f"succeeded={row['n_succeeded']:>4}  "
            f"rate={rate_str}"
        )
    log.info("=" * 60)
    log.info("\n  CRITICAL FINDING: No liveness detection. Static images == live faces")
    log.info("  in embedding space. Photo-attack success rate equals TAR of enrolled")
    log.info("  test-set images (typically very high for enrolled identities).")
    log.info("\n  All results in " + str(out_dir))
    log.info("Phase 7 complete.")


if __name__ == "__main__":
    main()
