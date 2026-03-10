#!/usr/bin/env python3
"""
revo_pi.py — SOTA Raspberry Pi Runtime for REVO Robot Dog
==========================================================

Optimisations vs the desktop face_control_center.py
─────────────────────────────────────────────────────
1. Dual-resolution pipeline   – detect at 320×240, recognise on full-res crop
2. Face-position cache        – re-use identity when bbox barely moved (IOU gate)
3. Adaptive frame skip        – IDLE=5 / DETECTING=2 / AUTHORIZED=1
4. Motion gate                – skip inference on static frames (frame-diff, cheap)
5. Gesture ROI crop           – MediaPipe runs only on face-adjacent region (~50% faster)
6. MediaPipe Lite (complexity=0) – 2× faster hand inference, same gesture accuracy
7. Pre-normalised embeddings  – matching = pure dot-product, no sqrt at runtime
8. Producer-consumer threads  – capture thread never blocks on inference
9. cv2.setNumThreads(4)       – all four RPi Cortex cores for ONNX DNN
10. V4L2 backend              – lower latency than auto-select on Linux
11. Linux audio               – aplay / mpg123 / afplay, no winsound
12. JSON config               – tune without touching code

Requirements
────────────
    pip install opencv-contrib-python numpy mediapipe

Usage
─────
    python revo_pi.py                          # uses revo_config.json if present
    python revo_pi.py --config my.json
    python revo_pi.py --enroll Alice --samples 25   # enroll then run
    python revo_pi.py --save-config                 # write default config and exit
    python revo_pi.py --iot-url http://robot/cmd    # override IoT URL
    python revo_pi.py --no-gesture                  # disable gesture (faster)
    python revo_pi.py --verbose                     # debug logging
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import queue
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, deque
from dataclasses import asdict, dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

# ── Force OpenCV to use all available cores BEFORE any inference ───────────────
cv2.setNumThreads(4)
cv2.setUseOptimized(True)

# ── Optional MediaPipe ─────────────────────────────────────────────────────────
try:
    import mediapipe as mp
    _MP_OK = True
except ImportError:
    mp = None
    _MP_OK = False

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("revo_pi")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR          = Path(__file__).resolve().parent.parent   # project root
KNOWN_FACES_DIR   = BASE_DIR / "data" / "known_faces"
MODELS_DIR        = BASE_DIR / "models"
DB_FILE           = BASE_DIR / "data" / "face_db.npz"
CONFIG_FILE       = BASE_DIR / "revo_config.json"
DETECTOR_MODEL    = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
RECOGNIZER_MODEL  = MODELS_DIR / "face_recognition_sface_2021dec.onnx"

MODEL_URLS = {
    DETECTOR_MODEL:   "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    RECOGNIZER_MODEL: "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}

# Detection resolution — lower = faster ONNX inference, recognition still uses full-res crop
DETECT_W, DETECT_H = 320, 240


# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # Camera
    camera_index:    int   = 0
    capture_width:   int   = 640
    capture_height:  int   = 480
    use_v4l2:        bool  = True       # V4L2 backend on Linux (lower latency)
    unmirror:        bool  = True       # flip horizontally (standard webcam)

    # Face detection (at 320×240)
    det_score:             float = 0.88   # slightly lower than 0.9 because faces are smaller at 320×240
    min_face_area_ratio:   float = 0.04   # relative to 320×240 frame

    # Recognition thresholds
    threshold:          float = 0.42
    margin:             float = 0.06
    centroid_threshold: float = 0.40

    # Temporal voting
    history_len:   int = 6
    stable_votes:  int = 4

    # Face-position cache — skip SFace if face barely moved
    face_cache_iou:     float = 0.72
    face_cache_timeout: float = 1.5    # seconds before forcing re-recognition

    # Adaptive frame skip per state
    frame_skip_idle:       int = 5
    frame_skip_detecting:  int = 2
    frame_skip_authorized: int = 1

    # Gesture
    gesture_enabled:    bool  = True
    gesture_every_n:    int   = 2      # run gesture every N inference frames when authorized
    gesture_history:    int   = 4
    gesture_votes:      int   = 2
    gesture_cooldown:   float = 1.2
    hand_max_dim:       int   = 320    # max edge for MediaPipe input (ROI resized if larger)

    # Motion gate — only active in IDLE state
    motion_gate_fraction:    float = 0.008   # fraction of 160×120 pixels that must change
    motion_gate_pixel_thresh: int  = 20

    # Authorization timing
    track_timeout: float = 2.5
    cooldown:      float = 3.0
    greet_cooldown: float = 20.0
    light_normalize: bool = False      # CLAHE+gamma; enable if lighting is inconsistent

    # Robot HTTP
    iot_url:     str   = ""
    iot_timeout: float = 2.0

    # Audio
    bark_audio: str = ""               # path to .wav or .mp3

    @classmethod
    def from_json(cls, path: Path) -> "Config":
        if not path.exists():
            return cls()
        with open(path) as f:
            data = json.load(f)
        cfg = cls()
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        log.info("Config saved → %s", path)


# ══════════════════════════════════════════════════════════════════════════════
# State machine
# ══════════════════════════════════════════════════════════════════════════════

class State(Enum):
    IDLE       = auto()   # No face seen recently
    DETECTING  = auto()   # Face seen, building identity votes
    AUTHORIZED = auto()   # Identity confirmed, gestures active


GESTURE_COMMANDS: Dict[str, str] = {
    "FORWARD":  "forward",
    "BACKWARD": "backward",
    "LEFT":     "left",
    "RIGHT":    "right",
    "BARK":     "bark",
    "STAND":    "stand",
    "TAIL_WAG": "tail_wag",
    "WALK":     "walk",
    "SIT":      "sit",
    "STOP":     "stop",
    "GREET":    "greet",
}


# ══════════════════════════════════════════════════════════════════════════════
# Geometry helpers
# ══════════════════════════════════════════════════════════════════════════════

def _bbox(face: np.ndarray) -> Tuple[int, int, int, int]:
    x, y, w, h = face[:4]
    return int(max(0, x)), int(max(0, y)), int(max(0, x + w)), int(max(0, y + h))


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _scale_face_row(face: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Scale a YuNet face row [x,y,w,h, 5×(lx,ly), score] from detect to full resolution."""
    r = face.copy()
    r[0] *= sx; r[1] *= sy   # x, y
    r[2] *= sx; r[3] *= sy   # w, h
    for i in range(5):        # 5 landmark pairs at indices 4..13
        r[4 + i * 2]     *= sx
        r[4 + i * 2 + 1] *= sy
    # r[14] = score — unchanged
    return r


# ══════════════════════════════════════════════════════════════════════════════
# Lighting normalisation (CLAHE + adaptive gamma)
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_lighting(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    mean_l = max(1.0, float(np.mean(l_eq)))
    gamma = float(np.clip(np.log(128.0 / 255.0) / np.log(mean_l / 255.0), 0.75, 1.45))
    l_fix = np.clip(np.power(l_eq.astype(np.float32) / 255.0, gamma) * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((l_fix, a, b)), cv2.COLOR_LAB2BGR)


# ══════════════════════════════════════════════════════════════════════════════
# Model management
# ══════════════════════════════════════════════════════════════════════════════

def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s …", dst.name)
    urllib.request.urlretrieve(url, str(dst))


def ensure_models() -> None:
    for path, url in MODEL_URLS.items():
        if not path.exists():
            _download(url, path)


def _create_detector(score: float) -> cv2.FaceDetectorYN:
    fn = getattr(cv2, "FaceDetectorYN_create", None) or cv2.FaceDetectorYN.create
    return fn(str(DETECTOR_MODEL), "", (DETECT_W, DETECT_H), score, 0.3, 5000)


def _create_recognizer() -> cv2.FaceRecognizerSF:
    fn = getattr(cv2, "FaceRecognizerSF_create", None) or cv2.FaceRecognizerSF.create
    return fn(str(RECOGNIZER_MODEL), "")


# ══════════════════════════════════════════════════════════════════════════════
# Face embedding database
# ══════════════════════════════════════════════════════════════════════════════

class FaceDB:
    """Pre-normalised database — matching is a pure dot-product, no sqrt at runtime."""

    def __init__(self, path: Path = DB_FILE) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"face_db.npz not found at {path}.\n"
                "Run:  python face_embedding.py enroll --name YourName --samples 25"
            )
        data = np.load(path, allow_pickle=False)
        embs  = data["embeddings"].astype(np.float32)
        names = data["names"].astype(str)

        # Ensure unit-norm (face_embedding.py already normalises, but be safe)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        self.embeddings = (embs / norms).astype(np.float32)
        self.names      = names

        if "centroids" in data and "centroid_names" in data:
            cents  = data["centroids"].astype(np.float32)
            cn     = np.linalg.norm(cents, axis=1, keepdims=True)
            cn     = np.where(cn < 1e-8, 1.0, cn)
            self.centroids      = (cents / cn).astype(np.float32)
            self.centroid_names = data["centroid_names"].astype(str)
        else:
            self._build_centroids()

        self.unique_names, self.inverse = np.unique(self.names, return_inverse=True)
        log.info("DB: %d embeddings, %d identities: %s",
                 len(self.names), len(self.unique_names), list(self.unique_names))

    def _build_centroids(self) -> None:
        cents, cnames = [], []
        for person in sorted(set(self.names.tolist())):
            c = self.embeddings[self.names == person].mean(axis=0)
            n = np.linalg.norm(c)
            if n > 1e-8:
                cents.append(c / n)
                cnames.append(person)
        self.centroids      = np.vstack(cents).astype(np.float32)
        self.centroid_names = np.array(cnames, dtype=str)

    def match(self, emb: np.ndarray, threshold: float, margin: float,
              centroid_threshold: float) -> Tuple[str, float, float]:
        # Pure dot-product — O(N), no sqrt needed (embeddings pre-normalised)
        sims = self.embeddings @ emb
        best_per = np.full(len(self.unique_names), -1.0, dtype=np.float32)
        np.maximum.at(best_per, self.inverse, sims)

        best_idx   = int(np.argmax(best_per))
        candidate  = str(self.unique_names[best_idx])
        best_score = float(best_per[best_idx])
        second     = float(np.partition(best_per, -2)[-2]) if len(best_per) > 1 else -1.0

        if best_score < threshold or (best_score - second) < margin:
            return "Unknown", best_score, 0.0

        csims  = self.centroids @ emb
        ci     = int(np.argmax(csims))
        cscore = float(csims[ci])
        cname  = str(self.centroid_names[ci])

        if cname != candidate or cscore < centroid_threshold:
            return "Unknown", best_score, cscore

        return candidate, best_score, cscore


# ══════════════════════════════════════════════════════════════════════════════
# Motion gate
# ══════════════════════════════════════════════════════════════════════════════

class MotionGate:
    """Cheap frame-diff at 160×120 grayscale — skip inference on static scenes."""

    def __init__(self, pixel_thresh: int, fraction: float) -> None:
        self._thresh   = pixel_thresh
        self._fraction = fraction
        self._prev: Optional[np.ndarray] = None
        self._area = 160 * 120

    def has_motion(self, frame: np.ndarray) -> bool:
        small = cv2.resize(frame, (160, 120))
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if self._prev is None:
            self._prev = gray
            return True
        diff = cv2.absdiff(self._prev, gray)
        self._prev = gray
        return int(np.count_nonzero(diff > self._thresh)) > int(self._area * self._fraction)


# ══════════════════════════════════════════════════════════════════════════════
# Gesture classifier — exact logic ported from face_control_center.py
# ══════════════════════════════════════════════════════════════════════════════

class GestureClassifier:
    """
    MediaPipe hand landmark classifier + rule-based gesture recognition.
    Uses Tasks API (mediapipe 0.10+) with hand_landmarker.task model.
    Falls back to legacy mp.solutions.hands for mediapipe 0.9.x.
    Runs on a face-adjacent ROI for speed.
    """

    def __init__(self, hand_max_dim: int = 320) -> None:
        if not _MP_OK:
            raise RuntimeError("mediapipe is not installed.")
        self._hand_max_dim = hand_max_dim
        self._use_tasks    = False

        # Try Tasks API first (mediapipe 0.10+)
        try:
            from mediapipe.tasks import python as _mp_tasks
            from mediapipe.tasks.python import vision as _mp_vision

            _model_path = str(MODELS_DIR / "hand_landmarker.task")
            if not Path(_model_path).exists():
                raise FileNotFoundError(f"hand_landmarker.task not found at {_model_path}")

            _base  = _mp_tasks.BaseOptions(model_asset_path=_model_path)
            _opts  = _mp_vision.HandLandmarkerOptions(
                base_options=_base,
                running_mode=_mp_vision.RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=0.55,
                min_hand_presence_confidence=0.45,
                min_tracking_confidence=0.45,
            )
            self._landmarker = _mp_vision.HandLandmarker.create_from_options(_opts)
            self._use_tasks  = True
            log.info("GestureClassifier: using MediaPipe Tasks API (0.10+)")
        except Exception as _exc:
            # Fall back to legacy solutions API (mediapipe 0.9.x)
            log.debug("Tasks API unavailable (%s), trying mp.solutions", _exc)
            try:
                self._hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    model_complexity=0,
                    min_detection_confidence=0.55,
                    min_tracking_confidence=0.45,
                )
                log.info("GestureClassifier: using MediaPipe solutions API (0.9.x)")
            except Exception as _exc2:
                raise RuntimeError(
                    f"Could not initialise MediaPipe hand detector.\n"
                    f"Tasks API error: {_exc}\n"
                    f"Solutions API error: {_exc2}"
                ) from _exc2

    # ── Geometry helpers (identical to desktop) ───────────────────────────────

    @staticmethod
    def _dist2d(p1, p2) -> float:
        dx = float(p1.x - p2.x)
        dy = float(p1.y - p2.y)
        return (dx * dx + dy * dy) ** 0.5

    @staticmethod
    def _palm_orientation(lmk, hand_label: str) -> Tuple[bool, bool]:
        wx, wy = float(lmk[0].x), float(lmk[0].y)
        ix, iy = float(lmk[5].x), float(lmk[5].y)
        px, py = float(lmk[17].x), float(lmk[17].y)
        cross = (ix - wx) * (py - wy) - (iy - wy) * (px - wx)
        palm_confident = abs(cross) > 0.012
        if hand_label == "Right":
            return (cross < 0), palm_confident
        return (cross > 0), palm_confident

    def _finger_states(self, lmk, hand_label: str) -> Dict:
        def _open_closed(tip, pip):
            delta = float(lmk[pip].y - lmk[tip].y)
            return delta > 0.026, delta < 0.010

        idx_up, idx_cl  = _open_closed(8,  6)
        mid_up, mid_cl  = _open_closed(12, 10)
        rng_up, rng_cl  = _open_closed(16, 14)
        pky_up, pky_cl  = _open_closed(20, 18)

        tdx = float(lmk[4].x - lmk[3].x)
        tdy = float(lmk[4].y - lmk[3].y)
        tmin = 0.042
        thumb_up    = tdy < -tmin  and abs(tdy) > abs(tdx) * 1.15
        thumb_down  = tdy >  tmin  and abs(tdy) > abs(tdx) * 1.15
        thumb_left  = tdx < -tmin  and abs(tdx) > abs(tdy) * 1.15
        thumb_right = tdx >  tmin  and abs(tdx) > abs(tdy) * 1.15

        hs = max(self._dist2d(lmk[0], lmk[9]), 1e-6)
        thumb_folded   = (self._dist2d(lmk[4], lmk[5]) / hs) < 1.03 and abs(tdy) < 0.085 and abs(tdx) < 0.105
        thumb_extended = (self._dist2d(lmk[4], lmk[2]) / hs) > 0.74
        thumb_lateral  = abs(tdx) > 0.05
        thumb_open     = thumb_extended and not thumb_folded

        open_count   = int(idx_up) + int(mid_up) + int(rng_up) + int(pky_up)
        closed_count = int(idx_cl) + int(mid_cl) + int(rng_cl) + int(pky_cl)

        palm_facing, palm_confident = self._palm_orientation(lmk, hand_label)

        return dict(
            index=idx_up,  middle=mid_up,  ring=rng_up,  pinky=pky_up,
            index_closed=idx_cl, middle_closed=mid_cl, ring_closed=rng_cl, pinky_closed=pky_cl,
            thumb_up=thumb_up, thumb_down=thumb_down,
            thumb_left=thumb_left, thumb_right=thumb_right,
            thumb_folded=thumb_folded, thumb_extended=thumb_extended,
            thumb_open=thumb_open, thumb_lateral=thumb_lateral,
            all_four_open=(open_count == 4),
            all_five_open=(open_count == 4 and (thumb_open or thumb_lateral or thumb_up or thumb_down)),
            all_four_closed=(closed_count >= 3 and open_count == 0),
            mostly_closed=(closed_count >= 3 and open_count <= 1),
            open_count=open_count, closed_count=closed_count,
            palm_facing=palm_facing, palm_confident=palm_confident,
        )

    def _classify(self, lmk, hand_label: str) -> Optional[str]:
        s = self._finger_states(lmk, hand_label)

        if not s["palm_confident"] or not s["palm_facing"]:
            return None

        # Index horizontal lean (for LEFT/RIGHT)
        idx_dx = float(lmk[8].x - lmk[6].x)

        if s["closed_count"] >= 4 and s["thumb_folded"]:
            return "STOP"

        hs = max(self._dist2d(lmk[0], lmk[9]), 1e-6)
        if self._dist2d(lmk[4], lmk[8]) / hs < 0.40 and s["middle"] and s["ring"] and s["pinky"]:
            return "BARK"

        if s["thumb_down"] and s["index_closed"] and s["middle_closed"] and s["ring_closed"] and s["pinky_closed"]:
            return "BACKWARD"

        if s["all_five_open"]:
            return "FORWARD"

        if s["all_four_open"] and s["thumb_folded"]:
            return "WALK"

        if s["index"] and s["middle"] and s["ring_closed"] and s["pinky_closed"]:
            return "SIT"

        if s["index"] and s["middle"] and s["ring"] and s["pinky_closed"]:
            return "STAND"

        if s["pinky"] and s["index_closed"] and s["middle_closed"] and s["ring_closed"] and s["thumb_folded"]:
            return "TAIL_WAG"

        if s["index"] and s["middle_closed"] and s["ring_closed"] and s["pinky_closed"]:
            if idx_dx < -0.03:
                return "LEFT"
            if idx_dx > 0.03:
                return "RIGHT"

        return None

    def _gesture_roi(
        self,
        shape: Tuple[int, int, int],
        face_bbox: Optional[Tuple[int, int, int, int]],
    ) -> Tuple[int, int, int, int]:
        h, w = shape[:2]
        if face_bbox is None:
            return 0, 0, w, h
        x1, y1, x2, y2 = face_bbox
        fw = max(1, x2 - x1)
        fh = max(1, y2 - y1)
        rx1 = max(0, int(x1 - 1.00 * fw))
        rx2 = min(w, int(x2 + 1.00 * fw))
        ry1 = max(0, int(y1 - 0.20 * fh))
        ry2 = min(h, int(y2 + 2.00 * fh))
        if rx2 <= rx1 or ry2 <= ry1:
            return 0, 0, w, h
        return rx1, ry1, rx2, ry2

    def _pick_hand(
        self,
        wrist_points: List[Tuple[int, int]],
        face_bbox: Optional[Tuple[int, int, int, int]],
    ) -> Optional[int]:
        if not wrist_points:
            return None
        if face_bbox is None:
            return 0
        x1, y1, x2, y2 = face_bbox
        fw = max(float(x2 - x1), 1.0)
        fh = max(float(y2 - y1), 1.0)
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        face_size = max(fw, fh)
        max_dist_sq = (face_size * 1.8) ** 2
        gx1, gx2 = x1 - 1.2 * fw, x2 + 1.2 * fw
        gy1, gy2 = y1 - 0.35 * fh, y2 + 2.2 * fh

        best_idx, best_dsq = None, None
        for i, (hx, hy) in enumerate(wrist_points):
            if not (gx1 <= hx <= gx2 and gy1 <= hy <= gy2):
                continue
            dsq = (float(hx) - cx) ** 2 + (float(hy) - cy) ** 2
            if best_dsq is None or dsq < best_dsq:
                best_dsq, best_idx = dsq, i
        if best_dsq is None or best_dsq > max_dist_sq:
            return None
        return best_idx

    def detect(
        self,
        frame_bgr: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[str]:
        """
        Run MediaPipe on a face-adjacent ROI, classify gesture, return label or None.
        """
        rx1, ry1, rx2, ry2 = self._gesture_roi(frame_bgr.shape, face_bbox)
        roi = frame_bgr[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return None
        roi_h, roi_w = roi.shape[:2]

        # Resize ROI if too large (saves MediaPipe inference time)
        infer = roi
        max_dim = max(roi_w, roi_h)
        if max_dim > self._hand_max_dim:
            scale   = self._hand_max_dim / float(max_dim)
            infer_w = max(1, int(roi_w * scale))
            infer_h = max(1, int(roi_h * scale))
            infer   = cv2.resize(roi, (infer_w, infer_h), interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(infer, cv2.COLOR_BGR2RGB)

        hands_data, wrist_pts = [], []
        try:
            if self._use_tasks:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result   = self._landmarker.detect(mp_image)
                if not result.hand_landmarks:
                    return None
                for idx, lmk in enumerate(result.hand_landmarks):
                    label = "Right"
                    if result.handedness and idx < len(result.handedness):
                        label = result.handedness[idx][0].category_name
                    gesture = self._classify(lmk, label)
                    wx = rx1 + int(lmk[0].x * roi_w)
                    wy = ry1 + int(lmk[0].y * roi_h)
                    hands_data.append(gesture)
                    wrist_pts.append((wx, wy))
            else:
                result = self._hands.process(rgb)
                if not result.multi_hand_landmarks:
                    return None
                for idx, hand_lm in enumerate(result.multi_hand_landmarks):
                    label = "Right"
                    if result.multi_handedness and idx < len(result.multi_handedness):
                        label = result.multi_handedness[idx].classification[0].label
                    lmk = hand_lm.landmark
                    gesture = self._classify(lmk, label)
                    wx = rx1 + int(lmk[0].x * roi_w)
                    wy = ry1 + int(lmk[0].y * roi_h)
                    hands_data.append(gesture)
                    wrist_pts.append((wx, wy))
        except Exception:
            return None

        chosen = self._pick_hand(wrist_pts, face_bbox)
        if chosen is None:
            return None
        return hands_data[chosen]

    def close(self) -> None:
        if self._use_tasks:
            self._landmarker.close()
        else:
            self._hands.close()


# ══════════════════════════════════════════════════════════════════════════════
# Command dispatcher
# ══════════════════════════════════════════════════════════════════════════════

class CommandDispatcher:
    """Non-blocking HTTP + Linux/macOS audio. Inference is never stalled by network."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._q: queue.Queue = queue.Queue(maxsize=16)
        threading.Thread(target=self._worker, daemon=True, name="cmd-dispatch").start()

    def send(self, person: str, command: str, source: str = "gesture") -> None:
        payload = {"person": person, "command": command, "source": source, "timestamp": time.time()}
        try:
            self._q.put_nowait(payload)
        except queue.Full:
            pass

    def _worker(self) -> None:
        while True:
            payload = self._q.get()
            self._dispatch(payload)

    def _dispatch(self, payload: dict) -> None:
        cmd = payload.get("command", "")
        log.info("[ROBOT] %-12s → %s", payload.get("person"), cmd)

        if cmd == "bark":
            self._play_audio()

        url = self._cfg.iot_url.strip()
        if not url:
            return
        try:
            data = json.dumps(payload).encode()
            req  = urllib.request.Request(url=url, data=data,
                                          headers={"Content-Type": "application/json"},
                                          method="POST")
            with urllib.request.urlopen(req, timeout=self._cfg.iot_timeout):
                pass
        except Exception as exc:
            log.warning("IoT send failed: %s", exc)

    def _play_audio(self) -> None:
        path = self._cfg.bark_audio.strip()
        if not path or not Path(path).exists():
            return
        try:
            sys_name = platform.system()
            ext = Path(path).suffix.lower()
            if sys_name == "Linux":
                player = "mpg123" if ext == ".mp3" else "aplay"
                subprocess.Popen([player, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif sys_name == "Darwin":
                subprocess.Popen(["afplay", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            log.warning("Audio player not found. Install aplay (sudo apt install alsa-utils) or mpg123.")
        except Exception as exc:
            log.warning("Audio failed: %s", exc)


# ══════════════════════════════════════════════════════════════════════════════
# Camera capture thread
# ══════════════════════════════════════════════════════════════════════════════

class CameraCapture:
    """
    Dedicated capture thread that always stores the LATEST frame.
    maxsize=1 queue means inference always sees the freshest frame;
    slower inference naturally drops stale frames without any busy-wait.
    """

    def __init__(self, cfg: Config, stop_evt: threading.Event) -> None:
        self._cfg      = cfg
        self._stop_evt = stop_evt
        self._q: queue.Queue = queue.Queue(maxsize=1)
        threading.Thread(target=self._run, daemon=True, name="cam-capture").start()

    def _open(self) -> cv2.VideoCapture:
        is_linux = platform.system() == "Linux"
        backend  = cv2.CAP_V4L2 if (self._cfg.use_v4l2 and is_linux) else cv2.CAP_ANY
        cap = cv2.VideoCapture(self._cfg.camera_index, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._cfg.capture_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.capture_height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # always grab latest frame
        try:
            cap.set(cv2.CAP_PROP_FPS, 30)
        except Exception:
            pass
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self._cfg.camera_index}")
        log.info("Camera open: %dx%d", self._cfg.capture_width, self._cfg.capture_height)
        return cap

    def _run(self) -> None:
        cap = self._open()
        try:
            while not self._stop_evt.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                # Drain stale frame then put new one (non-blocking)
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._q.put_nowait(frame)
                except queue.Full:
                    pass
        finally:
            cap.release()
            log.info("Camera released.")

    def get(self, timeout: float = 0.2) -> Optional[np.ndarray]:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None


# ══════════════════════════════════════════════════════════════════════════════
# Main runtime
# ══════════════════════════════════════════════════════════════════════════════

class PIRuntime:
    """
    Adaptive-skip, dual-resolution, state-machine pipeline.

    State transitions
    ─────────────────
    IDLE       → DETECTING   : face detected in frame
    DETECTING  → AUTHORIZED  : stable_votes / history_len frames agree on identity
    DETECTING  → IDLE        : no face for > track_timeout + 1 s
    AUTHORIZED → DETECTING   : authorized face lost for > track_timeout
    AUTHORIZED → IDLE        : no face at all for > track_timeout + 1 s
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg        = cfg
        self._stop_evt   = threading.Event()

        log.info("Loading models …")
        ensure_models()
        self._detector   = _create_detector(cfg.det_score)
        self._recognizer = _create_recognizer()

        log.info("Loading face database …")
        self._db = FaceDB()

        self._dispatcher = CommandDispatcher(cfg)
        self._motion     = MotionGate(cfg.motion_gate_pixel_thresh, cfg.motion_gate_fraction)

        self._gesture: Optional[GestureClassifier] = None
        if cfg.gesture_enabled and _MP_OK:
            try:
                self._gesture = GestureClassifier(cfg.hand_max_dim)
                log.info("Gesture classifier ready (MediaPipe Lite).")
            except Exception as exc:
                log.warning("Gesture disabled: %s", exc)
        elif not _MP_OK:
            log.warning("MediaPipe not installed — gesture commands disabled.")

        # ── State ─────────────────────────────────────────────────────────────
        self._state: State = State.IDLE
        self._face_history: deque  = deque(maxlen=cfg.history_len)
        self._gesture_history: deque = deque(maxlen=cfg.gesture_history)
        self._authorized_until: Dict[str, float] = {}
        self._last_face_time  = 0.0
        self._last_trigger: Dict[str, float] = {}
        self._last_greet:   Dict[str, float] = {}

        # Face-position cache (skip SFace when person barely moved)
        self._cached_id:    Optional[str]                        = None
        self._cached_bbox:  Optional[Tuple[int, int, int, int]]  = None
        self._cached_bbox_detect: Optional[Tuple[int, int, int, int]] = None  # full-res bbox of cached face
        self._cached_at:    float = 0.0

        # Gesture de-bounce
        self._last_gesture_sent = ""
        self._last_gesture_time = 0.0

        # Scale factors from detect resolution to full capture resolution
        self._sx = cfg.capture_width  / DETECT_W
        self._sy = cfg.capture_height / DETECT_H

    # ── Frame skip ────────────────────────────────────────────────────────────
    def _skip(self) -> int:
        return {
            State.IDLE:       self._cfg.frame_skip_idle,
            State.DETECTING:  self._cfg.frame_skip_detecting,
            State.AUTHORIZED: self._cfg.frame_skip_authorized,
        }[self._state]

    # ── Core inference ────────────────────────────────────────────────────────
    def _process(self, frame: np.ndarray, frame_id: int) -> None:
        cfg = self._cfg
        now = time.time()

        # Motion gate (IDLE only — cheap CPU save)
        if self._state == State.IDLE and not self._motion.has_motion(frame):
            return

        # Optional lighting normalisation
        if cfg.light_normalize:
            frame = _normalize_lighting(frame)

        # ── 1. Detection at 320×240 ───────────────────────────────────────────
        small = cv2.resize(frame, (DETECT_W, DETECT_H))
        self._detector.setInputSize((DETECT_W, DETECT_H))
        _, faces = self._detector.detect(small)

        if faces is None or len(faces) == 0:
            self._face_history.append(set())
            elapsed = now - self._last_face_time
            if self._state == State.AUTHORIZED and elapsed > cfg.track_timeout:
                self._state = State.DETECTING
                log.info("→ DETECTING  (face lost)")
            elif elapsed > cfg.track_timeout + 1.0:
                if self._state != State.IDLE:
                    self._state = State.IDLE
                    self._face_history.clear()
                    self._authorized_until.clear()
                    self._cached_id = None
                    log.info("→ IDLE")
            return

        self._last_face_time = now
        if self._state == State.IDLE:
            self._state = State.DETECTING
            log.info("→ DETECTING")

        # Sort by bbox area (largest face first)
        faces = sorted(faces, key=lambda f: float(f[2] * f[3]), reverse=True)

        # ── 2. Recognition ────────────────────────────────────────────────────
        names_this_frame: Set[str] = set()
        controller_bbox: Optional[Tuple[int, int, int, int]] = None  # for gesture ROI

        for face in faces:
            # Area filter at detect resolution
            if float(face[2] * face[3]) / (DETECT_W * DETECT_H) < cfg.min_face_area_ratio:
                continue

            # Scale bbox to full-res
            dx1, dy1, dx2, dy2 = _bbox(face)
            fx1 = int(dx1 * self._sx); fy1 = int(dy1 * self._sy)
            fx2 = int(dx2 * self._sx); fy2 = int(dy2 * self._sy)
            curr_bbox = (fx1, fy1, fx2, fy2)

            # Face-position cache check
            use_cache = (
                self._cached_id is not None
                and self._cached_bbox is not None
                and (now - self._cached_at) < cfg.face_cache_timeout
                and _iou(curr_bbox, self._cached_bbox) >= cfg.face_cache_iou
            )

            if use_cache:
                name = self._cached_id
            else:
                # Scale full face row for SFace alignCrop
                full_face = _scale_face_row(face, self._sx, self._sy)
                h, w = frame.shape[:2]
                # Clamp to frame bounds
                cx1 = max(0, min(w - 1, fx1))
                cy1 = max(0, min(h - 1, fy1))
                cx2 = max(cx1 + 1, min(w, fx2))
                cy2 = max(cy1 + 1, min(h, fy2))
                if cx2 - cx1 < 4 or cy2 - cy1 < 4:
                    continue

                try:
                    aligned = self._recognizer.alignCrop(frame, full_face)
                    feat    = self._recognizer.feature(aligned).reshape(-1).astype(np.float32)
                    norm    = float(np.linalg.norm(feat))
                    if norm < 1e-8:
                        continue
                    emb  = feat / norm
                    name, _, _ = self._db.match(emb, cfg.threshold, cfg.margin, cfg.centroid_threshold)
                except Exception as exc:
                    log.debug("SFace error: %s", exc)
                    continue

                if name != "Unknown":
                    self._cached_id   = name
                    self._cached_bbox = curr_bbox
                    self._cached_at   = now

            if name != "Unknown":
                names_this_frame.add(name)
                if controller_bbox is None:
                    controller_bbox = curr_bbox   # largest authorized face owns the gesture

        self._face_history.append(names_this_frame)

        # ── 3. Temporal voting → authorization ────────────────────────────────
        if len(self._face_history) == cfg.history_len:
            vote: Counter = Counter()
            for s in self._face_history:
                vote.update(s)
            for person, count in vote.items():
                if count >= cfg.stable_votes:
                    newly = person not in self._authorized_until or now > self._authorized_until[person]
                    self._authorized_until[person] = now + cfg.track_timeout

                    if (now - self._last_trigger.get(person, 0.0)) > cfg.cooldown:
                        self._last_trigger[person] = now
                        if self._state != State.AUTHORIZED:
                            self._state = State.AUTHORIZED
                            log.info("→ AUTHORIZED: %s", person)

                    if newly and (now - self._last_greet.get(person, 0.0)) > cfg.greet_cooldown:
                        self._last_greet[person] = now
                        self._dispatcher.send(person, "greet", source="face")

        # Expire stale authorizations
        expired = [p for p, until in self._authorized_until.items() if now > until]
        for p in expired:
            del self._authorized_until[p]
        if not self._authorized_until and self._state == State.AUTHORIZED:
            self._state = State.DETECTING
            log.info("→ DETECTING  (auth expired)")

        # ── 4. Gesture (AUTHORIZED only, every N frames) ──────────────────────
        if (
            self._state == State.AUTHORIZED
            and self._gesture is not None
            and self._authorized_until
            and frame_id % cfg.gesture_every_n == 0
        ):
            person = next(iter(self._authorized_until))  # primary controller
            label  = self._gesture.detect(frame, face_bbox=controller_bbox)
            self._gesture_history.append(label)

            if len(self._gesture_history) >= cfg.gesture_votes:
                counts = Counter(g for g in self._gesture_history if g is not None)
                if counts:
                    top, top_n = counts.most_common(1)[0]
                    if top_n >= cfg.gesture_votes:
                        elapsed = now - self._last_gesture_time
                        if top != self._last_gesture_sent or elapsed > cfg.gesture_cooldown:
                            self._last_gesture_sent = top
                            self._last_gesture_time = now
                            cmd = GESTURE_COMMANDS.get(top, top.lower())
                            self._dispatcher.send(person, cmd, source="gesture")
                            self._gesture_history.clear()
                            log.info("GESTURE: %-10s → %s", top, cmd)

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self) -> None:
        signal.signal(signal.SIGINT,  lambda *_: self._stop_evt.set())
        signal.signal(signal.SIGTERM, lambda *_: self._stop_evt.set())

        camera     = CameraCapture(self._cfg, self._stop_evt)
        frame_id   = 0
        skip_count = 0
        t_status   = time.time()

        log.info("REVO Pi runtime started. Ctrl+C to stop.")

        try:
            while not self._stop_evt.is_set():
                frame = camera.get(timeout=0.2)
                if frame is None:
                    continue

                if self._cfg.unmirror:
                    frame = cv2.flip(frame, 1)

                frame_id  += 1
                skip_count += 1
                if skip_count < self._skip():
                    continue
                skip_count = 0

                self._process(frame, frame_id)

                # Periodic heartbeat log every 10 s
                now = time.time()
                if now - t_status > 10.0:
                    t_status = now
                    log.info("State=%-12s | Auth=%s | Frames=%d",
                             self._state.name,
                             list(self._authorized_until.keys()) or "none",
                             frame_id)
        finally:
            self._stop_evt.set()
            if self._gesture:
                self._gesture.close()
            log.info("REVO Pi stopped.")


# ══════════════════════════════════════════════════════════════════════════════
# Enrollment helper
# ══════════════════════════════════════════════════════════════════════════════

def enroll_and_run(name: str, samples: int, cfg: Config) -> None:
    """Capture face images, build DB, then start the runtime."""
    import argparse as _ap
    _src_dir = str(Path(__file__).resolve().parent)
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)
    import face_embedding as fe

    log.info("Enrolling '%s' (%d samples) …", name, samples)
    args = _ap.Namespace(
        name=name, samples=samples, replace=False,
        camera=cfg.camera_index, width=cfg.capture_width, height=cfg.capture_height,
        det_score=0.9, no_download=False, db=DB_FILE,
        no_light_normalize=True, light_normalize=False,
        augment_lighting=True, no_augment_lighting=False,
        min_area_ratio=0.08,
    )
    fe.run_enroll(args)
    log.info("Enrollment done. Starting runtime …")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="REVO Robot Dog — Raspberry Pi SOTA Runtime",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",      type=Path,  default=CONFIG_FILE, help="JSON config file")
    parser.add_argument("--enroll",      type=str,   default="",          help="Enroll person then run")
    parser.add_argument("--samples",     type=int,   default=25,          help="Samples for --enroll")
    parser.add_argument("--save-config", action="store_true",             help="Write default config and exit")
    parser.add_argument("--iot-url",     type=str,   default="",          help="Override IoT URL")
    parser.add_argument("--no-gesture",  action="store_true",             help="Disable gesture commands")
    parser.add_argument("--verbose",     action="store_true",             help="Debug logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg = Config.from_json(args.config)

    if args.save_config:
        cfg.save(args.config)
        return

    # CLI overrides
    if args.iot_url:
        cfg.iot_url = args.iot_url
    if args.no_gesture:
        cfg.gesture_enabled = False

    if args.enroll:
        enroll_and_run(args.enroll, args.samples, cfg)

    try:
        PIRuntime(cfg).run()
    except FileNotFoundError as exc:
        log.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
