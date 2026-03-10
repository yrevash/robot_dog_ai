#!/usr/bin/env python3
"""
experiments/sweep_voting.py
============================
Phase 3 — Voting window analysis: latency vs security tradeoff.

Simulates temporal voting across (history_len, stable_count) combinations and
frame-skip values, measuring:
  - Authorization latency (frames and wall-clock seconds)
  - Whether a simulated impostor stream would be authorized (security risk)

Data source priority:
  1. results/phase2/recognition_results.csv (from eval_face_recognition.py)
  2. Fallback: runs face detection + recognition on known_faces/ images to
     build per-frame result sequences.  Fully runnable with no extra data.

Outputs (all under results/phase3/):
  voting_sweep.csv             — (H, V, latency_frames, latency_s, impostor_auth)
  frame_skip_sweep.csv         — frame_skip, effective_latency_s
  voting_heatmap_latency.png   — heatmap: latency (frames) indexed by H x V
  voting_heatmap_security.png  — heatmap: impostor auth (red=unsafe, green=safe)
  frame_skip_bar.png           — bar chart: adjusted latency vs frame_skip

Usage:
  python experiments/sweep_voting.py
  python experiments/sweep_voting.py --fps 30
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, deque
from pathlib import Path
from typing import Optional

# ── Bootstrap: ensure project root is importable ───────────────────────────────
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
    load_csv,
    save_csv,
    setup_logging,
    DB_FILE,
    KNOWN_FACES,
)
import face_embedding as fe


# ── Parameter grids ────────────────────────────────────────────────────────────
HISTORY_LENS  = [3, 4, 5, 6, 8, 10]
STABLE_COUNTS = [2, 3, 4, 5]
FRAME_SKIPS   = [1, 2, 3, 4, 5]

# Fixed parameters for the frame-skip sweep
FS_HISTORY = 6
FS_STABLE  = 4


# ── Voting simulator ───────────────────────────────────────────────────────────
def simulate_voting(
    frame_results: list[str],
    history_len: int,
    stable_count: int,
) -> Optional[int]:
    """
    Feed frame_results (a list of predicted name strings) one at a time into
    a deque-based voting window of size history_len.

    Returns the frame index (1-based) at which authorization is first granted
    (i.e., any non-Unknown name reaches stable_count votes within the window),
    or None if no authorization occurs within the sequence.
    """
    history: deque[set] = deque(maxlen=history_len)

    for frame_idx, pred in enumerate(frame_results, start=1):
        # Each frame contributes a singleton set (mimics real code where a frame
        # may contain multiple recognized faces; here we have at most one).
        frame_set = {pred} if pred != "Unknown" else set()
        history.append(frame_set)

        if len(history) < history_len:
            # Window not full yet — no authorization
            continue

        vote_counts: Counter[str] = Counter()
        for s in history:
            vote_counts.update(s)

        for person, count in vote_counts.items():
            if count >= stable_count:
                return frame_idx

    return None  # never authorized


# ── Data builders ──────────────────────────────────────────────────────────────
def _load_results_csv(csv_path: Path, log) -> tuple[list[str], list[str]]:
    """
    Parse recognition_results.csv produced by eval_face_recognition.py.

    Uses only rows from the "D: Full two-gate" config (the strongest gate).
    Returns (enrolled_preds, impostor_preds):
      enrolled_preds — predicted names for enrolled-image rows
      impostor_preds — predicted names for synthetic impostor rows
                       (true_name == "Unknown" in that CSV)
    """
    rows = load_csv(csv_path)
    log.info(f"Loaded {len(rows)} rows from {csv_path}")

    best_config = "D: Full two-gate"
    enrolled_preds: list[str] = []
    impostor_preds: list[str] = []

    for row in rows:
        if row.get("config", "") != best_config:
            continue
        true_name = row.get("true_name", "")
        predicted  = row.get("predicted", "Unknown")
        if true_name == "Unknown":
            impostor_preds.append(predicted)
        else:
            enrolled_preds.append(predicted)

    log.info(
        f"  Config '{best_config}': "
        f"{len(enrolled_preds)} enrolled rows, {len(impostor_preds)} impostor rows"
    )
    return enrolled_preds, impostor_preds


def _make_impostor_image(img: np.ndarray) -> np.ndarray:
    """Heavy transform to simulate an impostor (same as sweep_threshold.py)."""
    out = cv2.convertScaleAbs(img, alpha=0.30, beta=120)
    out = cv2.GaussianBlur(out, (21, 21), sigmaX=8)
    out = cv2.flip(out, 1)
    shift = np.array([40, -30, 20], dtype=np.int32)
    return np.clip(out.astype(np.int32) + shift, 0, 255).astype(np.uint8)


def _build_from_known_faces(log) -> tuple[list[str], list[str]]:
    """
    Fallback: run detection + recognition on known_faces/ images using the
    default (threshold=0.42, margin=0.06) gate and collect per-frame predictions.

    Returns (enrolled_preds, impostor_preds).
    """
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    fe.check_opencv_requirements()
    fe.ensure_models()

    db_embeddings, db_names, centroids, centroid_names = fe.load_db(DB_FILE)
    log.info(f"DB loaded: {len(db_embeddings)} embeddings")

    detector   = fe.create_detector((640, 480), score_threshold=0.85)
    recognizer = fe.create_recognizer()

    enrolled_preds: list[str] = []
    impostor_preds: list[str] = []

    DEFAULT_GATE = dict(threshold=0.42, margin=0.06, centroid_threshold=0.40)

    person_dirs = sorted(p for p in KNOWN_FACES.iterdir() if p.is_dir())
    if not person_dirs:
        log.error(f"No person folders in {KNOWN_FACES}")
        return enrolled_preds, impostor_preds

    for person_dir in person_dirs:
        images = sorted(p for p in person_dir.iterdir() if p.suffix.lower() in image_exts)
        if not images:
            continue

        person_enrolled = 0
        person_impostor = 0

        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            proc = fe.normalize_lighting(img)
            faces = fe.detect_faces(detector, proc)

            if faces:
                emb = fe.embedding_from_face(proc, faces[0], recognizer)
                if emb is not None:
                    pred, _, _ = fe.match_identity(
                        emb, db_embeddings, db_names, centroids, centroid_names,
                        **DEFAULT_GATE,
                    )
                    enrolled_preds.append(pred)
                    person_enrolled += 1
                else:
                    enrolled_preds.append("Unknown")
            else:
                enrolled_preds.append("Unknown")

            # Synthetic impostor version
            imp = _make_impostor_image(img)
            imp_faces = fe.detect_faces(detector, imp)
            if imp_faces:
                imp_emb = fe.embedding_from_face(imp, imp_faces[0], recognizer)
                if imp_emb is not None:
                    imp_pred, _, _ = fe.match_identity(
                        imp_emb, db_embeddings, db_names, centroids, centroid_names,
                        **DEFAULT_GATE,
                    )
                    impostor_preds.append(imp_pred)
                    person_impostor += 1
                    continue

            # Face not detected in impostor image → clearly rejected
            impostor_preds.append("Unknown")
            person_impostor += 1

        log.info(
            f"  {person_dir.name}: {person_enrolled} enrolled predictions, "
            f"{person_impostor} impostor predictions"
        )

    return enrolled_preds, impostor_preds


# ── Heatmap plotter (pure imshow, no seaborn) ──────────────────────────────────
def _plot_heatmap_latency(
    h_vals: list[int],
    v_vals: list[int],
    latency_matrix: np.ndarray,
    out_dir: Path,
    log,
) -> None:
    """
    Heatmap of mean authorization latency (frames).
    Rows = stable_count (V), cols = history_len (H).
    """
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    im = ax.imshow(latency_matrix, aspect="auto", cmap="viridis",
                   interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Mean auth latency (frames)")

    ax.set_xticks(range(len(h_vals)))
    ax.set_xticklabels([str(h) for h in h_vals])
    ax.set_yticks(range(len(v_vals)))
    ax.set_yticklabels([str(v) for v in v_vals])
    ax.set_xlabel("History Length (H)")
    ax.set_ylabel("Stable Count (V)")
    ax.set_title("Voting Window: Authorization Latency (frames)\n(NaN = never authorized)")

    # Annotate each cell
    for ri, v in enumerate(v_vals):
        for ci, h in enumerate(h_vals):
            val = latency_matrix[ri, ci]
            txt = f"{val:.0f}" if not np.isnan(val) else "—"
            color = "white" if val > (np.nanmax(latency_matrix) * 0.6) else "black"
            ax.text(ci, ri, txt, ha="center", va="center", fontsize=8, color=color)

    out = out_dir / "voting_heatmap_latency.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Plot saved -> {out}")


def _plot_heatmap_security(
    h_vals: list[int],
    v_vals: list[int],
    security_matrix: np.ndarray,
    out_dir: Path,
    log,
) -> None:
    """
    Heatmap of impostor authorization (1=unsafe/red, 0=safe/green).
    """
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    # RdYlGn: 0 -> green (safe), 1 -> red (unsafe)
    im = ax.imshow(security_matrix, aspect="auto", cmap="RdYlGn_r",
                   vmin=0.0, vmax=1.0, interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_ticks([0.0, 1.0])
    cbar.set_ticklabels(["SAFE", "UNSAFE"])

    ax.set_xticks(range(len(h_vals)))
    ax.set_xticklabels([str(h) for h in h_vals])
    ax.set_yticks(range(len(v_vals)))
    ax.set_yticklabels([str(v) for v in v_vals])
    ax.set_xlabel("History Length (H)")
    ax.set_ylabel("Stable Count (V)")
    ax.set_title("Voting Window: Impostor Authorization Risk\n(red=UNSAFE, green=SAFE)")

    for ri, v in enumerate(v_vals):
        for ci, h in enumerate(h_vals):
            val = security_matrix[ri, ci]
            label = "UNSAFE" if val > 0.5 else "SAFE"
            ax.text(ci, ri, label, ha="center", va="center", fontsize=8, color="black")

    out = out_dir / "voting_heatmap_security.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Plot saved -> {out}")


def _plot_frame_skip_bar(
    frame_skips: list[int],
    latencies_s: list[float],
    fps: float,
    out_dir: Path,
    log,
) -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7, 4))

    colors = ["#42A5F5" if lat < 1.0 else "#FF7043" for lat in latencies_s]
    bars = ax.bar([str(fs) for fs in frame_skips], latencies_s, color=colors, edgecolor="gray")

    for bar, lat in zip(bars, latencies_s):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{lat:.2f}s",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_xlabel("Frame Skip (process every N-th frame)")
    ax.set_ylabel("Auth Latency (seconds)")
    ax.set_title(
        f"Latency vs Frame Skip  (H={FS_HISTORY}, V={FS_STABLE}, FPS={fps:.0f})\n"
        "Blue < 1.0s, Orange >= 1.0s"
    )
    valid_lats = [v for v in latencies_s if v == v and v > 0]  # exclude nan and 0
    ylim_top = max(valid_lats) * 1.25 + 0.1 if valid_lats else 2.0
    ax.set_ylim(0, ylim_top)

    out = out_dir / "frame_skip_bar.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Plot saved -> {out}")


# ── Main ────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3 — Voting window latency / security sweep"
    )
    parser.add_argument(
        "--fps", type=float, default=30.0,
        help="Camera FPS for converting frame latency to seconds (default: 30)",
    )
    args = parser.parse_args()

    fps = args.fps
    log = setup_logging("phase3_voting_sweep")
    out_dir = get_results_dir("phase3")

    # ── Acquire per-frame recognition results ──────────────────────────────────
    recog_csv = _PROJECT_ROOT / "results" / "phase2" / "recognition_results.csv"

    if recog_csv.exists():
        log.info(f"Loading recognition results from {recog_csv}")
        enrolled_preds, impostor_preds = _load_results_csv(recog_csv, log)
        if not enrolled_preds:
            log.warning(
                "No usable rows from recognition_results.csv "
                "(expected 'D: Full two-gate' config rows). "
                "Falling back to known_faces/ inference."
            )
            enrolled_preds, impostor_preds = _build_from_known_faces(log)
    else:
        log.info(
            f"recognition_results.csv not found at {recog_csv}. "
            "Running inference on known_faces/ (fallback mode)."
        )
        if not check_db_exists(log):
            sys.exit(1)
        enrolled_preds, impostor_preds = _build_from_known_faces(log)

    if not enrolled_preds:
        log.error(
            "No frame-level recognition results available. "
            "Run eval_face_recognition.py or ensure known_faces/ has images."
        )
        sys.exit(1)

    log.info(
        f"Frame sequences: {len(enrolled_preds)} enrolled frames, "
        f"{len(impostor_preds)} impostor frames"
    )

    # ── Voting sweep: (history_len x stable_count) ────────────────────────────
    log.info(
        f"\nSweeping history_len={HISTORY_LENS}, stable_count={STABLE_COUNTS} "
        f"at FPS={fps}"
    )

    sweep_rows: list[dict] = []

    # Matrices for heatmaps — rows=V (stable_count), cols=H (history_len)
    latency_matrix  = np.full((len(STABLE_COUNTS), len(HISTORY_LENS)), np.nan)
    security_matrix = np.zeros((len(STABLE_COUNTS), len(HISTORY_LENS)), dtype=float)

    for ci, h in enumerate(HISTORY_LENS):
        for ri, v in enumerate(STABLE_COUNTS):

            # Skip logically impossible: stable_count > history_len
            if v > h:
                log.debug(f"  H={h}, V={v}: impossible (V>H), skipping")
                sweep_rows.append({
                    "history_len":     h,
                    "stable_count":    v,
                    "latency_frames":  "N/A",
                    "latency_s":       "N/A",
                    "impostor_auth":   False,
                })
                latency_matrix[ri, ci]  = np.nan
                security_matrix[ri, ci] = 0.0
                continue

            # --- Enrolled stream ---
            auth_frame = simulate_voting(enrolled_preds, h, v)
            if auth_frame is not None:
                lat_frames = float(auth_frame)
                lat_s      = auth_frame / fps
            else:
                lat_frames = np.nan   # never authorized (very high V relative to H)
                lat_s      = np.nan

            # --- Impostor stream (all frames predict Unknown) ---
            # Simulate a pure-Unknown stream of the same length
            imp_stream  = ["Unknown"] * max(len(impostor_preds), h + 1)
            imp_auth    = simulate_voting(imp_stream, h, v) is not None

            # Robustness check: also test actual impostor predictions
            if not imp_auth and impostor_preds:
                imp_auth = simulate_voting(impostor_preds, h, v) is not None

            latency_matrix[ri, ci]  = lat_frames
            security_matrix[ri, ci] = 1.0 if imp_auth else 0.0

            lat_log = f"{lat_frames:.1f}" if not np.isnan(lat_frames) else "never"
            sec_log = "UNSAFE" if imp_auth else "SAFE"
            log.info(f"  H={h:2d}, V={v}: latency={lat_log} frames  ({sec_log})")

            sweep_rows.append({
                "history_len":     h,
                "stable_count":    v,
                "latency_frames":  lat_frames if not np.isnan(lat_frames) else "N/A",
                "latency_s":       round(lat_s, 4) if not np.isnan(lat_s) else "N/A",
                "impostor_auth":   imp_auth,
            })

    save_csv(
        out_dir / "voting_sweep.csv",
        sweep_rows,
        ["history_len", "stable_count", "latency_frames", "latency_s", "impostor_auth"],
    )
    log.info(f"Saved -> {out_dir / 'voting_sweep.csv'}")

    # ── Frame-skip sweep (fixed H=6, V=4) ─────────────────────────────────────
    log.info(f"\nSweeping frame_skip={FRAME_SKIPS} at H={FS_HISTORY}, V={FS_STABLE}, FPS={fps}")

    fs_rows: list[dict] = []
    fs_latencies: list[float] = []

    for frame_skip in FRAME_SKIPS:
        # Sub-sample the enrolled stream (simulate only processing every N-th frame)
        sampled = enrolled_preds[::frame_skip]
        auth_frame = simulate_voting(sampled, FS_HISTORY, FS_STABLE)

        if auth_frame is not None:
            # Convert sampled-frame index back to real-frame index
            real_frame_idx = auth_frame * frame_skip
            # Wall-clock latency: real frames elapsed / camera fps
            effective_lat_s = real_frame_idx / fps
        else:
            real_frame_idx  = None
            effective_lat_s = np.nan

        fs_latencies.append(effective_lat_s if not np.isnan(effective_lat_s) else 0.0)
        auth_log   = str(auth_frame)    if auth_frame    is not None else "never"
        real_log   = str(real_frame_idx) if real_frame_idx is not None else "N/A"
        lat_log    = f"{effective_lat_s:.3f}" if not np.isnan(effective_lat_s) else "N/A"
        log.info(
            f"  frame_skip={frame_skip}: "
            f"auth at sampled frame {auth_log} "
            f"(real frame ~{real_log})  "
            f"wall-clock ~{lat_log}s"
        )
        fs_rows.append({
            "frame_skip":          frame_skip,
            "effective_latency_s": round(effective_lat_s, 4) if not np.isnan(effective_lat_s) else "N/A",
        })

    save_csv(
        out_dir / "frame_skip_sweep.csv",
        fs_rows,
        ["frame_skip", "effective_latency_s"],
    )
    log.info(f"Saved -> {out_dir / 'frame_skip_sweep.csv'}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    log.info("\nGenerating plots...")

    _plot_heatmap_latency(
        h_vals=HISTORY_LENS,
        v_vals=STABLE_COUNTS,
        latency_matrix=latency_matrix,
        out_dir=out_dir,
        log=log,
    )

    _plot_heatmap_security(
        h_vals=HISTORY_LENS,
        v_vals=STABLE_COUNTS,
        security_matrix=security_matrix,
        out_dir=out_dir,
        log=log,
    )

    _plot_frame_skip_bar(
        frame_skips=FRAME_SKIPS,
        latencies_s=fs_latencies,
        fps=fps,
        out_dir=out_dir,
        log=log,
    )

    log.info("\n" + "=" * 60)
    log.info("Phase 3 complete. Outputs in results/phase3/:")
    log.info("  voting_sweep.csv             — full H x V grid")
    log.info("  frame_skip_sweep.csv         — latency vs frame_skip")
    log.info("  voting_heatmap_latency.png   — latency heatmap")
    log.info("  voting_heatmap_security.png  — impostor safety heatmap")
    log.info("  frame_skip_bar.png           — latency bar chart")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
