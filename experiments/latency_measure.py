#!/usr/bin/env python3
"""
experiments/latency_measure.py
===============================
Phase 6 — End-to-end system latency measurement.

Times each pipeline component individually and end-to-end across N frames,
then generates per-frame CSVs, summary statistics, and plots.

Components timed per frame:
  t_detect_ms  — YuNet detection (or simulated equivalent)
  t_embed_ms   — SFace alignCrop + feature extraction (or simulated)
  t_match_ms   — Identity match (dot product against face_db.npz)
  t_vote_ms    — Temporal voting logic
  t_gesture_ms — MediaPipe gesture inference (once authorized)
  t_http_ms    — HTTP POST to mock server (if --server-url provided)
  t_total_ms   — Wall-clock total for the frame

Also sweeps gesture_votes = [1, 2, 3, 4] and records mean gesture
decision latency.

Usage:
  python experiments/latency_measure.py
  python experiments/latency_measure.py --server-url http://localhost:8080/cmd
  python experiments/latency_measure.py --camera -1 --frames 100
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── Shared utilities ──────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils import setup_logging, get_results_dir  # noqa: E402

# ── Optional matplotlib ───────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_OK = True
except ImportError:
    plt = None  # type: ignore[assignment]
    _MPL_OK = False

# ── Optional mediapipe ────────────────────────────────────────────────────────
try:
    import mediapipe as mp
    _MP_OK = True
except ImportError:
    mp = None  # type: ignore[assignment]
    _MP_OK = False

# ── Constants ─────────────────────────────────────────────────────────────────
DETECT_W, DETECT_H = 320, 240
FULL_W,   FULL_H   = 640, 480
HISTORY_LEN  = 6
STABLE_VOTES = 4

# Authorize after this many frames regardless of real detection (for timing)
_FORCE_AUTH_AFTER = 6


# ══════════════════════════════════════════════════════════════════════════════
# Model / DB loaders
# ══════════════════════════════════════════════════════════════════════════════

def _load_detector(models_dir: Path, score: float = 0.88) -> Optional[cv2.FaceDetectorYN]:
    model_path = models_dir / "face_detection_yunet_2023mar.onnx"
    if not model_path.exists():
        return None
    try:
        fn = getattr(cv2, "FaceDetectorYN_create", None) or cv2.FaceDetectorYN.create
        return fn(str(model_path), "", (DETECT_W, DETECT_H), score, 0.3, 5000)
    except Exception:
        return None


def _load_recognizer(models_dir: Path) -> Optional[cv2.FaceRecognizerSF]:
    model_path = models_dir / "face_recognition_sface_2021dec.onnx"
    if not model_path.exists():
        return None
    try:
        fn = getattr(cv2, "FaceRecognizerSF_create", None) or cv2.FaceRecognizerSF.create
        return fn(str(model_path), "")
    except Exception:
        return None


def _load_face_db(db_path: Path):
    """
    Returns (embeddings: np.ndarray N×512, names: np.ndarray N,
             centroids: np.ndarray K×512, centroid_names: np.ndarray K)
    or None if DB missing.
    """
    if not db_path.exists():
        return None
    data = np.load(db_path, allow_pickle=False)
    embs  = data["embeddings"].astype(np.float32)
    names = data["names"].astype(str)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    embs  = embs / norms

    if "centroids" in data and "centroid_names" in data:
        cents  = data["centroids"].astype(np.float32)
        cn     = np.linalg.norm(cents, axis=1, keepdims=True)
        cn     = np.where(cn < 1e-8, 1.0, cn)
        centroids      = cents / cn
        centroid_names = data["centroid_names"].astype(str)
    else:
        unique = sorted(set(names.tolist()))
        cents, cnames = [], []
        for p in unique:
            c = embs[names == p].mean(axis=0)
            n = np.linalg.norm(c)
            if n > 1e-8:
                cents.append(c / n)
                cnames.append(p)
        centroids      = np.vstack(cents).astype(np.float32) if cents else embs[:1]
        centroid_names = np.array(cnames, dtype=str)

    return embs, names, centroids, centroid_names


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic camera (mirrors bench_rpi.py)
# ══════════════════════════════════════════════════════════════════════════════

class SyntheticCamera:
    def __init__(self, seed: int = 7) -> None:
        self._rng = np.random.default_rng(seed)
        self._n   = 0

    def read(self) -> Tuple[bool, np.ndarray]:
        frame = self._rng.integers(0, 64, (FULL_H, FULL_W, 3), dtype=np.uint8)
        cx, cy = FULL_W // 2, FULL_H // 2
        frame[cy - 50: cy + 50, cx - 50: cx + 50] = self._rng.integers(
            180, 255, (100, 100, 3), dtype=np.uint8
        )
        self._n += 1
        return True, frame

    def isOpened(self) -> bool:  # noqa: N802
        return True

    def release(self) -> None:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline timing components
# ══════════════════════════════════════════════════════════════════════════════

class TimedPipeline:
    """
    Executes detection → embedding → match → vote → (gesture) → (http)
    with per-stage timing.  Falls back to numpy proxies when models are absent.
    """

    def __init__(
        self,
        detector,
        recognizer,
        face_db,
        gesture_hands,
        server_url: Optional[str],
        gesture_votes: int,
        log,
    ) -> None:
        self._detector      = detector
        self._recognizer    = recognizer
        self._db            = face_db      # tuple or None
        self._hands         = gesture_hands
        self._server_url    = server_url
        self._gesture_votes = gesture_votes
        self._log           = log

        # Temporal vote state
        self._history: deque = deque(maxlen=HISTORY_LEN)
        self._authorized     = False
        self._frame_no       = 0

        # Gesture vote buffer (for decision latency sweep)
        self._gesture_hist: deque = deque(maxlen=gesture_votes)

    # ── Detection ─────────────────────────────────────────────────────────────

    def _detect(self, frame: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
        t0 = time.perf_counter()
        if self._detector is not None:
            small = cv2.resize(frame, (DETECT_W, DETECT_H))
            _, faces = self._detector.detect(small)
            face_row = faces[0] if (faces is not None and len(faces) > 0) else None
        else:
            # Proxy: Canny on resized frame
            small = cv2.resize(frame, (DETECT_W, DETECT_H))
            gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            _edges = cv2.Canny(gray, 50, 150)
            face_row = None   # synthetic mode — no real detections
        t_ms = (time.perf_counter() - t0) * 1000.0
        return t_ms, face_row

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed(
        self, frame: np.ndarray, face_row: Optional[np.ndarray]
    ) -> Tuple[float, Optional[np.ndarray]]:
        t0 = time.perf_counter()
        emb = None
        if self._recognizer is not None and face_row is not None:
            try:
                sx = FULL_W / DETECT_W
                sy = FULL_H / DETECT_H
                scaled = face_row.copy()
                scaled[0] *= sx; scaled[1] *= sy
                scaled[2] *= sx; scaled[3] *= sy
                for i in range(5):
                    scaled[4 + i * 2]     *= sx
                    scaled[4 + i * 2 + 1] *= sy
                crop = self._recognizer.alignCrop(frame, scaled)
                feat = self._recognizer.feature(crop)
                norm = np.linalg.norm(feat)
                emb  = feat / norm if norm > 1e-8 else feat
            except Exception:
                emb = None
        else:
            # Proxy: matmul of similar FLOP count to SFace 512-d embedding
            cx, cy = FULL_W // 2, FULL_H // 2
            crop = frame[cy - 50: cy + 50, cx - 50: cx + 50]
            if crop.size == 0:
                crop = np.zeros((100, 100, 3), dtype=np.uint8)
            resized = cv2.resize(crop, (112, 112)).astype(np.float32) / 255.0
            flat = resized.reshape(1, -1)
            W    = np.ones((flat.shape[1], 512), dtype=np.float32) * 0.001
            feat = (flat @ W)[0]
            norm = np.linalg.norm(feat)
            emb  = feat / norm if norm > 1e-8 else feat
        t_ms = (time.perf_counter() - t0) * 1000.0
        return t_ms, emb

    # ── Identity match ────────────────────────────────────────────────────────

    def _match(self, emb: Optional[np.ndarray]) -> Tuple[float, str]:
        t0 = time.perf_counter()
        identity = "Unknown"
        if emb is not None and self._db is not None:
            embeddings, names, centroids, centroid_names = self._db
            unique_names, inverse = np.unique(names, return_inverse=True)
            sims     = embeddings @ emb
            best_per = np.full(len(unique_names), -1.0, dtype=np.float32)
            np.maximum.at(best_per, inverse, sims)
            best_idx   = int(np.argmax(best_per))
            candidate  = str(unique_names[best_idx])
            best_score = float(best_per[best_idx])
            second     = float(np.partition(best_per, -2)[-2]) if len(best_per) > 1 else -1.0
            if best_score >= 0.42 and (best_score - second) >= 0.06:
                csims  = centroids @ emb
                ci     = int(np.argmax(csims))
                cscore = float(csims[ci])
                cname  = str(centroid_names[ci])
                if cname == candidate and cscore >= 0.40:
                    identity = candidate
        elif emb is not None:
            # No DB — simulate alternating match for timing variety
            dot = float(np.dot(emb, emb))   # trivially cheap
            identity = "Alice" if (self._frame_no % 4) != 0 else "Unknown"
        t_ms = (time.perf_counter() - t0) * 1000.0
        return t_ms, identity

    # ── Temporal vote ─────────────────────────────────────────────────────────

    def _vote(self, identity: str) -> Tuple[float, bool]:
        t0 = time.perf_counter()
        self._history.append(identity)
        known = sum(1 for v in self._history if v != "Unknown")
        authorized = known >= STABLE_VOTES
        t_ms = (time.perf_counter() - t0) * 1000.0
        return t_ms, authorized

    # ── Gesture inference ─────────────────────────────────────────────────────

    def _gesture(self, frame: np.ndarray) -> float:
        t0 = time.perf_counter()
        if self._hands is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _result = self._hands.process(rgb)
        else:
            # Proxy: GaussianBlur on a resized ROI
            cx, cy = FULL_W // 2, FULL_H // 2
            roi = frame[max(0, cy - 100): cy + 100, max(0, cx - 100): cx + 100]
            gray = cv2.cvtColor(roi if roi.size > 0 else frame[:200, :200],
                                cv2.COLOR_BGR2GRAY)
            _ = cv2.GaussianBlur(gray, (21, 21), 0)
        return (time.perf_counter() - t0) * 1000.0

    # ── HTTP POST ─────────────────────────────────────────────────────────────

    def _http_post(self, identity: str, gesture: str) -> float:
        if not self._server_url:
            return 0.0
        payload = json.dumps({
            "person":    identity,
            "command":   gesture,
            "source":    "latency_measure",
            "timestamp": time.time(),
        }).encode("utf-8")
        req = urllib.request.Request(
            self._server_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=2.0):
                pass
        except (urllib.error.URLError, OSError):
            pass
        return (time.perf_counter() - t0) * 1000.0

    # ── Full frame ────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> Dict[str, float]:
        t_wall_start = time.perf_counter()
        self._frame_no += 1

        t_detect_ms, face_row = self._detect(frame)

        if face_row is not None or self._recognizer is None:
            t_embed_ms, emb = self._embed(frame, face_row)
        else:
            t_embed_ms, emb = 0.0, None

        t_match_ms, identity = self._match(emb)

        # Force authorization after _FORCE_AUTH_AFTER frames for gesture timing
        if self._frame_no >= _FORCE_AUTH_AFTER:
            t_vote_ms, auth = self._vote(identity if identity != "Unknown" else "Alice")
            self._authorized = True
        else:
            t_vote_ms, auth = self._vote(identity)
            self._authorized = auth

        t_gesture_ms = 0.0
        if self._authorized:
            t_gesture_ms = self._gesture(frame)

        t_http_ms = 0.0
        if self._authorized and self._server_url:
            t_http_ms = self._http_post(identity, "FORWARD")

        t_total_ms = (time.perf_counter() - t_wall_start) * 1000.0

        return {
            "t_detect_ms":  t_detect_ms,
            "t_embed_ms":   t_embed_ms,
            "t_match_ms":   t_match_ms,
            "t_vote_ms":    t_vote_ms,
            "t_gesture_ms": t_gesture_ms,
            "t_http_ms":    t_http_ms,
            "t_total_ms":   t_total_ms,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Gesture vote sweep
# ══════════════════════════════════════════════════════════════════════════════

def _gesture_vote_sweep(frames: List[np.ndarray], log) -> List[Dict]:
    """
    For each gesture_votes value in [1,2,3,4]: simulate gesture timing
    with that many votes required for a stable decision.
    Returns rows for gesture_vote_sweep.csv.
    """
    sweep_results = []
    for votes in [1, 2, 3, 4]:
        gesture_times = []
        history: deque = deque(maxlen=votes)
        for frame in frames:
            t0 = time.perf_counter()
            # Proxy gesture
            cx, cy = FULL_W // 2, FULL_H // 2
            roi = frame[max(0, cy - 100): cy + 100, max(0, cx - 100): cx + 100]
            gray = cv2.cvtColor(roi if roi.size > 0 else frame[:200, :200],
                                cv2.COLOR_BGR2GRAY)
            _ = cv2.GaussianBlur(gray, (21, 21), 0)
            g_ms = (time.perf_counter() - t0) * 1000.0

            history.append("FORWARD")
            decided = len(history) >= votes and all(h == history[0] for h in history)
            gesture_times.append(g_ms)
            if decided:
                break

        mean_ms = float(np.mean(gesture_times)) * votes
        log.info("  gesture_votes=%d  mean_decision_ms=%.2f", votes, mean_ms)
        sweep_results.append({"votes": votes, "mean_decision_ms": f"{mean_ms:.4f}"})
    return sweep_results


# ══════════════════════════════════════════════════════════════════════════════
# Statistics helpers
# ══════════════════════════════════════════════════════════════════════════════

_COMPONENTS = [
    "t_detect_ms", "t_embed_ms", "t_match_ms",
    "t_vote_ms", "t_gesture_ms", "t_http_ms", "t_total_ms",
]


def _compute_summary(rows: List[Dict]) -> List[Dict]:
    summary = []
    for comp in _COMPONENTS:
        vals = [float(r[comp]) for r in rows]
        arr  = np.array(vals, dtype=np.float64)
        nonzero = arr[arr > 0]
        if len(nonzero) == 0:
            nonzero = arr
        summary.append({
            "component": comp,
            "mean":   f"{np.mean(arr):.4f}",
            "median": f"{np.median(arr):.4f}",
            "p95":    f"{np.percentile(arr, 95):.4f}",
            "p99":    f"{np.percentile(arr, 99):.4f}",
            "min":    f"{np.min(arr):.4f}",
            "max":    f"{np.max(arr):.4f}",
        })
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

_COMP_LABELS = {
    "t_detect_ms":  "Detect",
    "t_embed_ms":   "Embed",
    "t_match_ms":   "Match",
    "t_vote_ms":    "Vote",
    "t_gesture_ms": "Gesture",
    "t_http_ms":    "HTTP",
}
_STACKED_COMPS = [c for c in _COMPONENTS if c != "t_total_ms"]
_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]


def _plot_latency_breakdown(summary: List[Dict], out_dir: Path) -> None:
    if not _MPL_OK:
        return
    means = {row["component"]: float(row["mean"]) for row in summary}
    comps = _STACKED_COMPS
    vals  = [means.get(c, 0.0) for c in comps]
    labels = [_COMP_LABELS.get(c, c) for c in comps]

    fig, ax = plt.subplots(figsize=(8, 5))
    bottom  = 0.0
    for i, (v, lbl) in enumerate(zip(vals, labels)):
        ax.bar(["Mean Latency"], [v], bottom=bottom, label=lbl,
               color=_COLORS[i % len(_COLORS)], alpha=0.88)
        bottom += v

    ax.set_ylabel("Latency (ms)")
    ax.set_title("Phase 6 — Per-Component Mean Latency Breakdown")
    ax.legend(loc="upper right")
    fig.tight_layout()
    path = out_dir / "latency_breakdown.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


def _plot_latency_cdf(rows: List[Dict], out_dir: Path) -> None:
    if not _MPL_OK:
        return
    vals = sorted(float(r["t_total_ms"]) for r in rows)
    cdf  = np.arange(1, len(vals) + 1) / len(vals)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(vals, cdf, linewidth=2, color="#4C72B0")
    ax.set_xlabel("Total latency per frame (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("Phase 6 — CDF of Per-Frame Total Latency")
    ax.axvline(np.percentile(vals, 95), color="#C44E52", linestyle="--",
               linewidth=1.5, label="p95")
    ax.axvline(np.percentile(vals, 99), color="#DD8452", linestyle=":",
               linewidth=1.5, label="p99")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "latency_cdf.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


def _plot_gesture_sweep(sweep: List[Dict], out_dir: Path) -> None:
    if not _MPL_OK:
        return
    votes = [int(r["votes"]) for r in sweep]
    ms    = [float(r["mean_decision_ms"]) for r in sweep]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(votes, ms, marker="o", linewidth=2, color="#55A868")
    ax.set_xlabel("gesture_votes threshold")
    ax.set_ylabel("Mean decision latency (ms)")
    ax.set_title("Phase 6 — Gesture Vote Sweep")
    ax.set_xticks(votes)
    fig.tight_layout()
    path = out_dir / "gesture_vote_sweep.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# CSV savers
# ══════════════════════════════════════════════════════════════════════════════

def _save_raw_csv(rows: List[Dict], out_dir: Path) -> None:
    fieldnames = ["frame_id"] + _COMPONENTS
    path = out_dir / "latency_raw.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _save_summary_csv(summary: List[Dict], out_dir: Path) -> None:
    fieldnames = ["component", "mean", "median", "p95", "p99", "min", "max"]
    path = out_dir / "latency_summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary)


def _save_sweep_csv(sweep: List[Dict], out_dir: Path) -> None:
    path = out_dir / "gesture_vote_sweep.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["votes", "mean_decision_ms"])
        w.writeheader()
        w.writerows(sweep)


# ══════════════════════════════════════════════════════════════════════════════
# Frame collection helper
# ══════════════════════════════════════════════════════════════════════════════

def _collect_frames(camera_idx: int, n: int, log) -> List[np.ndarray]:
    if camera_idx < 0:
        log.info("Generating %d synthetic frames", n)
        cam = SyntheticCamera()
    else:
        log.info("Opening camera %d for %d frames", camera_idx, n)
        cam = cv2.VideoCapture(camera_idx)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH,  FULL_W)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FULL_H)
        if not cam.isOpened():
            log.warning("Camera %d unavailable — using synthetic", camera_idx)
            cam.release()
            cam = SyntheticCamera()
    frames = []
    for _ in range(n):
        ok, f = cam.read()
        if not ok:
            f = np.zeros((FULL_H, FULL_W, 3), dtype=np.uint8)
        frames.append(f)
    cam.release()
    return frames


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 6 — End-to-end pipeline latency measurement",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--camera",     type=int,  default=-1,
                   help="Camera index. Use -1 for synthetic frames.")
    p.add_argument("--frames",     type=int,  default=100,
                   help="Number of frames to time.")
    p.add_argument("--server-url", type=str,  default="",
                   help="HTTP server URL for round-trip timing, e.g. http://localhost:8080/cmd")
    p.add_argument("--gesture-votes", type=int, default=STABLE_VOTES,
                   help="Stable gesture vote threshold.")
    return p.parse_args()


def main() -> None:
    args    = _parse_args()
    log     = setup_logging("phase6_latency")
    out_dir = get_results_dir("phase6")

    models_dir = _ROOT / "models"
    db_path    = _ROOT / "face_db.npz"

    # ── Load models (graceful degradation to proxies) ─────────────────────────
    log.info("Loading models from %s", models_dir)
    detector   = _load_detector(models_dir)
    recognizer = _load_recognizer(models_dir)
    face_db    = _load_face_db(db_path)

    if detector is None:
        log.warning("YuNet model not found — using Canny proxy for detect timing")
    if recognizer is None:
        log.warning("SFace model not found — using matmul proxy for embed timing")
    if face_db is None:
        log.warning("face_db.npz not found — using synthetic identity for match timing")

    # ── MediaPipe hands ───────────────────────────────────────────────────────
    hands = None
    if _MP_OK:
        try:
            hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=0,
                min_detection_confidence=0.55,
                min_tracking_confidence=0.45,
            )
            log.info("MediaPipe Hands loaded (complexity=0)")
        except Exception as exc:
            log.warning("MediaPipe Hands init failed: %s — using blur proxy", exc)
            hands = None
    else:
        log.warning("mediapipe not installed — using blur proxy for gesture timing")

    # ── Collect frames ────────────────────────────────────────────────────────
    frames = _collect_frames(args.camera, args.frames, log)
    log.info("Collected %d frames", len(frames))

    # ── Run pipeline ──────────────────────────────────────────────────────────
    pipeline = TimedPipeline(
        detector=detector,
        recognizer=recognizer,
        face_db=face_db,
        gesture_hands=hands,
        server_url=args.server_url if args.server_url else None,
        gesture_votes=args.gesture_votes,
        log=log,
    )

    raw_rows: List[Dict] = []
    log.info("Timing %d frames …", len(frames))
    for i, frame in enumerate(frames):
        timing = pipeline.process_frame(frame)
        row = {"frame_id": i + 1}
        row.update({k: f"{v:.4f}" for k, v in timing.items()})
        raw_rows.append(row)
        if (i + 1) % 25 == 0:
            log.info("  … %d/%d frames", i + 1, len(frames))

    # ── Summary statistics ────────────────────────────────────────────────────
    summary = _compute_summary(raw_rows)

    log.info("=" * 70)
    log.info("%-15s %8s %8s %8s %8s %8s %8s",
             "Component", "Mean", "Median", "p95", "p99", "Min", "Max")
    log.info("-" * 70)
    for row in summary:
        log.info("%-15s %8s %8s %8s %8s %8s %8s",
                 row["component"], row["mean"], row["median"],
                 row["p95"], row["p99"], row["min"], row["max"])
    log.info("=" * 70)

    # ── Gesture vote sweep ────────────────────────────────────────────────────
    log.info("Running gesture vote sweep …")
    sweep = _gesture_vote_sweep(frames, log)

    # ── Save outputs ──────────────────────────────────────────────────────────
    _save_raw_csv(raw_rows, out_dir)
    _save_summary_csv(summary, out_dir)
    _save_sweep_csv(sweep, out_dir)

    _plot_latency_breakdown(summary, out_dir)
    _plot_latency_cdf(raw_rows, out_dir)
    _plot_gesture_sweep(sweep, out_dir)

    if _MPL_OK:
        log.info("Plots saved to %s", out_dir)
    else:
        log.warning("matplotlib not available — plots skipped. pip install matplotlib")

    log.info("All outputs written to: %s", out_dir)

    if hands is not None:
        hands.close()


if __name__ == "__main__":
    main()
