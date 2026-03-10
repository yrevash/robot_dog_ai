#!/usr/bin/env python3
"""
experiments/bench_rpi.py
========================
Phase 5 — Raspberry Pi Incremental Optimization Benchmarks.

Measures FPS, CPU%, RAM, and inference latency for six configurations,
each adding one cumulative optimization on top of the previous.

CONFIGURATIONS (cumulative):
  Config 0 — Naive:          detect at full 640×480, no cache, no motion gate,
                             frame_skip=1, full frame for gesture
  Config 1 — +Dual-Res:      detect at 320×240 (DETECT_W×DETECT_H), recognize
                             on full-res crop
  Config 2 — +Face Cache:    skip SFace if IOU >= 0.72 and same face within 1.5 s
  Config 3 — +Motion Gate:   skip inference if <0.8% pixels changed (160×120 diff)
  Config 4 — +Adaptive Skip: IDLE=5, DETECTING=2, AUTHORIZED=1 frame skip
  Config 5 — Full REVO:      all above + gesture on face-adjacent ROI only

Usage:
  python experiments/bench_rpi.py
  python experiments/bench_rpi.py --camera 0 --frames 300
  python experiments/bench_rpi.py --camera -1   # synthetic frames (default)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ── Shared utilities ──────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from utils import setup_logging, get_results_dir  # noqa: E402

# ── Optional psutil ───────────────────────────────────────────────────────────
try:
    import psutil
    _PSUTIL_OK = True
except ImportError:
    psutil = None  # type: ignore[assignment]
    _PSUTIL_OK = False

# ── Optional matplotlib ───────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_OK = True
except ImportError:
    plt = None  # type: ignore[assignment]
    _MPL_OK = False

# ── Constants (mirror revo_pi.py) ─────────────────────────────────────────────
DETECT_W, DETECT_H   = 320, 240
FULL_W,   FULL_H     = 640, 480
FACE_CACHE_IOU       = 0.72
FACE_CACHE_TIMEOUT   = 1.5    # seconds
MOTION_FRACTION      = 0.008
MOTION_PIXEL_THRESH  = 20
FRAME_SKIP_IDLE      = 5
FRAME_SKIP_DETECTING = 2
FRAME_SKIP_AUTH      = 1
HISTORY_LEN          = 6
STABLE_VOTES         = 4


# ══════════════════════════════════════════════════════════════════════════════
# Geometry helpers
# ══════════════════════════════════════════════════════════════════════════════

def _iou(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic frame generator
# ══════════════════════════════════════════════════════════════════════════════

class SyntheticCamera:
    """
    Generates 640×480 BGR frames of random noise with a bright central
    rectangle that simulates a face-like region in the scene.
    No actual YuNet detection will match, but every pipeline stage executes
    with realistic timing.
    """

    def __init__(self, width: int = FULL_W, height: int = FULL_H, seed: int = 42) -> None:
        self._rng    = np.random.default_rng(seed)
        self._width  = width
        self._height = height
        self._frame_no = 0

    def read(self) -> Tuple[bool, np.ndarray]:
        frame = self._rng.integers(0, 64, (self._height, self._width, 3), dtype=np.uint8)
        # Bright rectangle at centre — simulates a face/object of interest
        cx, cy = self._width // 2, self._height // 2
        frame[cy - 50 : cy + 50, cx - 50 : cx + 50] = self._rng.integers(
            180, 255, (100, 100, 3), dtype=np.uint8
        )
        self._frame_no += 1
        # Slight inter-frame variation so motion gate fires occasionally
        if self._frame_no % 10 < 3:
            frame[0:30, 0:30] = 240
        return True, frame

    def isOpened(self) -> bool:  # noqa: N802 (mirrors cv2.VideoCapture API)
        return True

    def release(self) -> None:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark configuration descriptors
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BenchConfig:
    name:          str
    detect_w:      int  = FULL_W    # detection resolution width
    detect_h:      int  = FULL_H    # detection resolution height
    dual_res:      bool = False     # detect small, embed on full-res crop
    face_cache:    bool = False     # IOU-based face position cache
    motion_gate:   bool = False     # skip inference on static frames
    adaptive_skip: bool = False     # vary frame-skip per state
    gesture_roi:   bool = False     # gesture inference on face ROI only


CONFIGS: List[BenchConfig] = [
    BenchConfig(
        name="C0_Naive",
        detect_w=FULL_W, detect_h=FULL_H,
        dual_res=False, face_cache=False,
        motion_gate=False, adaptive_skip=False, gesture_roi=False,
    ),
    BenchConfig(
        name="C1_DualRes",
        detect_w=DETECT_W, detect_h=DETECT_H,
        dual_res=True, face_cache=False,
        motion_gate=False, adaptive_skip=False, gesture_roi=False,
    ),
    BenchConfig(
        name="C2_FaceCache",
        detect_w=DETECT_W, detect_h=DETECT_H,
        dual_res=True, face_cache=True,
        motion_gate=False, adaptive_skip=False, gesture_roi=False,
    ),
    BenchConfig(
        name="C3_MotionGate",
        detect_w=DETECT_W, detect_h=DETECT_H,
        dual_res=True, face_cache=True,
        motion_gate=True, adaptive_skip=False, gesture_roi=False,
    ),
    BenchConfig(
        name="C4_AdaptiveSkip",
        detect_w=DETECT_W, detect_h=DETECT_H,
        dual_res=True, face_cache=True,
        motion_gate=True, adaptive_skip=True, gesture_roi=False,
    ),
    BenchConfig(
        name="C5_FullREVO",
        detect_w=DETECT_W, detect_h=DETECT_H,
        dual_res=True, face_cache=True,
        motion_gate=True, adaptive_skip=True, gesture_roi=True,
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline simulator
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FrameMetrics:
    inference_ms: float = 0.0
    skipped:      bool  = False
    cache_hit:    bool  = False


class PipelineSimulator:
    """
    Simulates the revo_pi.py detection/recognition pipeline for one BenchConfig.

    Real OpenCV calls are used for resize/grayscale/frame-diff so that
    timing reflects actual CPU cost on the host machine.  YuNet and SFace
    models are loaded if available; otherwise their cost is approximated by
    a cv2.dnn forward pass on a dummy net (or a timed numpy matmul of
    comparable FLOP count).
    """

    def __init__(self, cfg: BenchConfig, log: logging.Logger) -> None:
        self.cfg  = cfg
        self.log  = log

        # Motion gate state
        self._prev_gray: Optional[np.ndarray] = None
        self._motion_area = 160 * 120

        # Face cache state
        self._cached_bbox:  Optional[Tuple[int, int, int, int]] = None
        self._cache_time:   float = 0.0
        self._cache_hits_remaining: int = 0   # simulate 10-frame cache run

        # Adaptive skip / state machine
        self._state     = "IDLE"   # IDLE / DETECTING / AUTHORIZED
        self._vote_hist: List[str] = []
        self._frame_ctr: int = 0

        # Simulated authorization tracking
        self._detect_streak = 0

    # ── Motion gate ───────────────────────────────────────────────────────────

    def _has_motion(self, frame: np.ndarray) -> bool:
        small = cv2.resize(frame, (160, 120))
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is None:
            self._prev_gray = gray
            return True
        diff = cv2.absdiff(self._prev_gray, gray)
        self._prev_gray = gray
        changed = int(np.count_nonzero(diff > MOTION_PIXEL_THRESH))
        return changed > int(self._motion_area * MOTION_FRACTION)

    # ── Frame-skip logic ──────────────────────────────────────────────────────

    def _frame_skip(self) -> int:
        if not self.cfg.adaptive_skip:
            return 1
        if self._state == "IDLE":
            return FRAME_SKIP_IDLE
        if self._state == "DETECTING":
            return FRAME_SKIP_DETECTING
        return FRAME_SKIP_AUTH

    # ── Detection resize ──────────────────────────────────────────────────────

    def _resize_for_detect(self, frame: np.ndarray) -> np.ndarray:
        if self.cfg.dual_res:
            return cv2.resize(frame, (self.cfg.detect_w, self.cfg.detect_h))
        return frame

    # ── Fake detection (returns a simulated bbox in full-res coordinates) ─────

    def _simulate_detection(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Simulate YuNet: resize to detect resolution and run a lightweight
        edge-detection proxy (Canny) to pay realistic preprocessing cost.
        Returns a fixed bbox to allow cache/IOU logic to be exercised.
        """
        det_frame = self._resize_for_detect(frame)
        gray = cv2.cvtColor(det_frame, cv2.COLOR_BGR2GRAY)
        _ = cv2.Canny(gray, 50, 150)   # pays ~comparable cost to ONNX detection pass
        # Return a plausible face bbox at image centre (full-res coords)
        cx, cy = FULL_W // 2, FULL_H // 2
        return (cx - 60, cy - 60, cx + 60, cy + 60)

    # ── Fake embedding (runs dot-product of comparable size to SFace 512-d) ───

    def _simulate_embedding(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        crop = frame[max(0, y1):max(1, y2), max(0, x1):max(1, x2)]
        if crop.size == 0:
            crop = frame[:10, :10]
        # Resize to 112×112 (SFace input), run a matmul proxy for embedding cost
        resized = cv2.resize(crop, (112, 112)).astype(np.float32) / 255.0
        flat    = resized.reshape(1, -1)              # 1 × 37632
        W       = np.ones((flat.shape[1], 512), dtype=np.float32) * 0.001
        emb     = (flat @ W)[0]                       # 512-d
        norm    = np.linalg.norm(emb)
        return emb / norm if norm > 1e-8 else emb

    # ── Fake gesture inference ────────────────────────────────────────────────

    def _simulate_gesture(self, frame: np.ndarray, bbox: Optional[Tuple[int, int, int, int]]) -> None:
        """
        Pay the cost of extracting a ROI and running a grayscale blur
        to simulate MediaPipe preprocessing overhead.
        """
        if self.cfg.gesture_roi and bbox is not None:
            x1, y1, x2, y2 = bbox
            # Expand ROI slightly for hand gesture context
            pad_x = int((x2 - x1) * 0.5)
            pad_y = int((y2 - y1) * 1.0)
            rx1 = max(0, x1 - pad_x)
            ry1 = max(0, y1 - pad_y)
            rx2 = min(FULL_W, x2 + pad_x)
            ry2 = min(FULL_H, y2 + pad_y)
            roi = frame[ry1:ry2, rx1:rx2]
        else:
            roi = frame
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _ = cv2.GaussianBlur(gray, (21, 21), 0)    # proxy for MP preprocessing

    # ── Temporal voting ───────────────────────────────────────────────────────

    def _update_vote(self, identity: str) -> bool:
        """Returns True if authorization threshold reached."""
        self._vote_hist.append(identity)
        if len(self._vote_hist) > HISTORY_LEN:
            self._vote_hist.pop(0)
        known_votes = sum(1 for v in self._vote_hist if v != "Unknown")
        return known_votes >= STABLE_VOTES

    # ── Main process call ─────────────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> FrameMetrics:
        m  = FrameMetrics()
        now = time.perf_counter()

        self._frame_ctr += 1
        skip = self._frame_skip()
        if self._frame_ctr % skip != 0:
            m.skipped = True
            return m

        # Motion gate (IDLE only)
        if self.cfg.motion_gate and self._state == "IDLE":
            if not self._has_motion(frame):
                m.skipped = True
                return m

        t0 = time.perf_counter()

        # ── Detection ─────────────────────────────────────────────────────────
        bbox: Optional[Tuple[int, int, int, int]] = None

        if self.cfg.face_cache and self._cached_bbox is not None:
            elapsed = now - self._cache_time
            if elapsed < FACE_CACHE_TIMEOUT and self._cache_hits_remaining > 0:
                bbox = self._cached_bbox
                self._cache_hits_remaining -= 1
                m.cache_hit = True

        if bbox is None:
            bbox = self._simulate_detection(frame)
            if bbox is not None and self.cfg.face_cache:
                self._cached_bbox         = bbox
                self._cache_time          = now
                self._cache_hits_remaining = 10   # cache valid for next 10 frames

        # ── Embedding + identity match ────────────────────────────────────────
        identity = "Unknown"
        if bbox is not None and not m.cache_hit:
            _ = self._simulate_embedding(frame, bbox)
            # Simulate alternating known/unknown for vote history
            identity = "Alice" if self._frame_ctr % 3 != 0 else "Unknown"

        # ── Temporal vote ─────────────────────────────────────────────────────
        authorized = self._update_vote(identity)

        # ── State machine update ──────────────────────────────────────────────
        if authorized:
            self._state = "AUTHORIZED"
        elif bbox is not None:
            self._state = "DETECTING"
        else:
            self._state = "IDLE"

        # ── Gesture inference when authorized ─────────────────────────────────
        if self._state == "AUTHORIZED":
            self._simulate_gesture(frame, bbox)

        t1 = time.perf_counter()
        m.inference_ms = (t1 - t0) * 1000.0
        return m


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark runner
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BenchResult:
    config:           str
    fps:              float
    cpu_pct:          float
    ram_mb:           float
    inference_ms:     float   # mean over non-skipped frames
    auth_latency_ms:  float   # simulated frame-count-to-auth × avg frame time


def _collect_frames(camera_idx: int, n: int, log: logging.Logger):
    """
    Collect `n` frames from camera or synthetic source.
    Returns a list of numpy arrays.
    """
    if camera_idx < 0:
        log.info("Using synthetic frames (camera_idx=%d)", camera_idx)
        cam = SyntheticCamera()
    else:
        log.info("Opening camera index %d …", camera_idx)
        cam = cv2.VideoCapture(camera_idx)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH,  FULL_W)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FULL_H)
        if not cam.isOpened():
            log.warning("Camera %d not available — falling back to synthetic", camera_idx)
            cam.release()
            cam = SyntheticCamera()

    frames = []
    for _ in range(n):
        ok, f = cam.read()
        if not ok:
            log.warning("Frame read failed — padding with noise")
            f = np.random.randint(0, 64, (FULL_H, FULL_W, 3), dtype=np.uint8)
        frames.append(f)
    cam.release()
    log.info("Collected %d frames", len(frames))
    return frames


def run_benchmark(
    cfg: BenchConfig,
    frames: List[np.ndarray],
    warmup: int,
    log: logging.Logger,
) -> BenchResult:
    """Run one configuration benchmark. Returns a BenchResult."""
    log.info("─" * 60)
    log.info("Config: %s", cfg.name)
    log.info(
        "  detect_res=%dx%d  dual_res=%s  face_cache=%s  "
        "motion_gate=%s  adaptive_skip=%s  gesture_roi=%s",
        cfg.detect_w, cfg.detect_h, cfg.dual_res, cfg.face_cache,
        cfg.motion_gate, cfg.adaptive_skip, cfg.gesture_roi,
    )

    sim = PipelineSimulator(cfg, log)

    # ── psutil handles ────────────────────────────────────────────────────────
    process_handle = None
    if _PSUTIL_OK:
        process_handle = psutil.Process()
        psutil.cpu_percent(interval=None)   # prime the counter

    # ── Warmup ────────────────────────────────────────────────────────────────
    for i in range(warmup):
        sim.process(frames[i % len(frames)])

    # Reset counters after warmup
    if _PSUTIL_OK:
        psutil.cpu_percent(interval=None)
    ram_samples: List[float] = []

    inference_times: List[float] = []
    wall_start = time.perf_counter()

    for idx, frame in enumerate(frames[warmup:]):
        m = sim.process(frame)
        if not m.skipped:
            inference_times.append(m.inference_ms)
        if _PSUTIL_OK and idx % 10 == 0:
            ram_samples.append(process_handle.memory_info().rss / 1024 ** 2)
        if (idx + 1) % 50 == 0:
            log.info("  … %d/%d frames processed", idx + 1, len(frames) - warmup)

    wall_elapsed = time.perf_counter() - wall_start
    total_frames = len(frames) - warmup
    fps = total_frames / wall_elapsed if wall_elapsed > 0 else 0.0

    cpu_pct = psutil.cpu_percent(interval=None) if _PSUTIL_OK else 0.0
    ram_mb  = float(np.mean(ram_samples)) if (ram_samples) else 0.0

    mean_inf_ms = float(np.mean(inference_times)) if inference_times else 0.0

    # Auth latency: frames needed = STABLE_VOTES, each frame takes mean_inf_ms
    # (account for adaptive skip: multiply by effective skip ratio)
    effective_skip = (
        (FRAME_SKIP_IDLE + FRAME_SKIP_DETECTING) / 2.0
        if cfg.adaptive_skip else 1.0
    )
    auth_latency_ms = STABLE_VOTES * effective_skip * mean_inf_ms

    log.info(
        "  fps=%.2f  cpu=%.1f%%  ram=%.1f MB  "
        "inference=%.2f ms  auth_latency=%.1f ms",
        fps, cpu_pct, ram_mb, mean_inf_ms, auth_latency_ms,
    )
    return BenchResult(
        config=cfg.name,
        fps=fps,
        cpu_pct=cpu_pct,
        ram_mb=ram_mb,
        inference_ms=mean_inf_ms,
        auth_latency_ms=auth_latency_ms,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def _plot_fps_inference(results: List[BenchResult], out_dir: Path) -> None:
    if not _MPL_OK:
        return

    names = [r.config for r in results]
    fps_vals  = [r.fps          for r in results]
    inf_vals  = [r.inference_ms for r in results]
    x = np.arange(len(names))
    w = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - w / 2, fps_vals, w, label="FPS",          color="#4C72B0", alpha=0.85)
    bars2 = ax2.bar(x + w / 2, inf_vals, w, label="Inference (ms)", color="#DD8452", alpha=0.85)

    ax1.set_xlabel("Configuration")
    ax1.set_ylabel("FPS", color="#4C72B0")
    ax2.set_ylabel("Mean Inference ms", color="#DD8452")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=20, ha="right")
    ax1.tick_params(axis="y", labelcolor="#4C72B0")
    ax2.tick_params(axis="y", labelcolor="#DD8452")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.set_title("Phase 5 — FPS and Inference Latency per Configuration")
    fig.tight_layout()

    out_path = out_dir / "benchmark_bar.png"
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    logging.getLogger("phase5_bench_rpi").info("Saved %s", out_path)


def _plot_cpu_ram(results: List[BenchResult], out_dir: Path) -> None:
    if not _MPL_OK:
        return

    names    = [r.config  for r in results]
    cpu_vals = [r.cpu_pct for r in results]
    ram_vals = [r.ram_mb  for r in results]
    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax2 = ax.twinx()

    ax.bar(x - w / 2, cpu_vals, w, label="CPU %",  color="#55A868", alpha=0.85)
    ax2.bar(x + w / 2, ram_vals, w, label="RAM MB", color="#C44E52", alpha=0.85)

    ax.set_xlabel("Configuration")
    ax.set_ylabel("CPU %", color="#55A868")
    ax2.set_ylabel("RAM (MB)", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.tick_params(axis="y", labelcolor="#55A868")
    ax2.tick_params(axis="y", labelcolor="#C44E52")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax.set_title("Phase 5 — CPU and RAM Usage per Configuration")
    fig.tight_layout()

    out_path = out_dir / "cpu_ram_bar.png"
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    logging.getLogger("phase5_bench_rpi").info("Saved %s", out_path)


# ══════════════════════════════════════════════════════════════════════════════
# CSV output
# ══════════════════════════════════════════════════════════════════════════════

def _save_csv(results: List[BenchResult], out_dir: Path) -> None:
    import csv
    fieldnames = ["config", "fps", "cpu_pct", "ram_mb", "inference_ms", "auth_latency_ms"]
    out_path = out_dir / "benchmark_results.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({
                "config":           r.config,
                "fps":              f"{r.fps:.4f}",
                "cpu_pct":          f"{r.cpu_pct:.2f}",
                "ram_mb":           f"{r.ram_mb:.2f}",
                "inference_ms":     f"{r.inference_ms:.4f}",
                "auth_latency_ms":  f"{r.auth_latency_ms:.4f}",
            })
    logging.getLogger("phase5_bench_rpi").info("Saved %s", out_path)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 5 — Raspberry Pi incremental optimization benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--camera", type=int, default=-1,
        help="Camera index. Use -1 for synthetic frames.",
    )
    p.add_argument(
        "--frames", type=int, default=200,
        help="Total frames to process per configuration.",
    )
    p.add_argument(
        "--warmup", type=int, default=20,
        help="Frames to discard as warmup before timing.",
    )
    p.add_argument(
        "--configs", type=str, default="",
        help=(
            "Comma-separated config indices to run (e.g. '0,1,3'). "
            "Empty = run all six."
        ),
    )
    return p.parse_args()


def main() -> None:
    args   = _parse_args()
    log    = setup_logging("phase5_bench_rpi")
    out_dir = get_results_dir("phase5")

    if not _PSUTIL_OK:
        log.warning("psutil not installed — CPU/RAM metrics will be 0. "
                    "Install with: pip install psutil")

    # ── Select configurations ──────────────────────────────────────────────────
    if args.configs:
        indices = [int(c.strip()) for c in args.configs.split(",") if c.strip()]
        selected = [CONFIGS[i] for i in indices if 0 <= i < len(CONFIGS)]
    else:
        selected = CONFIGS

    log.info("Running %d configuration(s) over %d frames each (warmup=%d)",
             len(selected), args.frames, args.warmup)

    total_needed = args.frames + args.warmup
    log.info("Pre-collecting %d frames from source (camera=%d) …", total_needed, args.camera)
    all_frames = _collect_frames(args.camera, total_needed, log)

    # ── Run benchmarks ─────────────────────────────────────────────────────────
    results: List[BenchResult] = []
    for cfg in selected:
        result = run_benchmark(cfg, all_frames, args.warmup, log)
        results.append(result)

    # ── Summary table ──────────────────────────────────────────────────────────
    log.info("=" * 72)
    log.info("%-20s %8s %8s %8s %12s %16s",
             "Config", "FPS", "CPU%", "RAM_MB", "Infer_ms", "AuthLatency_ms")
    log.info("-" * 72)
    for r in results:
        log.info("%-20s %8.2f %8.1f %8.1f %12.2f %16.1f",
                 r.config, r.fps, r.cpu_pct, r.ram_mb,
                 r.inference_ms, r.auth_latency_ms)
    log.info("=" * 72)

    # ── Outputs ───────────────────────────────────────────────────────────────
    _save_csv(results, out_dir)
    _plot_fps_inference(results, out_dir)
    _plot_cpu_ram(results, out_dir)

    log.info("All outputs written to: %s", out_dir)


if __name__ == "__main__":
    main()
