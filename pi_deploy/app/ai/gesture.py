"""
MediaPipe hand landmark → gesture classifier.

Rules match docs/GESTURE_CHEAT_SHEET.md. Palm must face the camera, hand must
hold the sign for roughly `gesture_hold_s` before we emit a command.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:  # pragma: no cover
    mp = None
    mp_python = None
    mp_vision = None

from ..config import settings


class GestureClassifier:
    def __init__(self):
        if mp is None:
            raise RuntimeError("mediapipe is not installed. pip install mediapipe")

        model_path = settings.models_dir / "hand_landmarker.task"
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} missing — run the main app once to auto-download, "
                "or copy it from ../models/"
            )

        opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self.detector = mp_vision.HandLandmarker.create_from_options(opts)

        self._current: Optional[str] = None
        self._held_since: float = 0.0
        self._last_emitted: Optional[str] = None
        self._last_emit_ts: float = 0.0

    # ---------- rule-based gesture recognition ----------

    @staticmethod
    def _finger_states(lm) -> Tuple[bool, bool, bool, bool, bool]:
        """Return (thumb, index, middle, ring, pinky) booleans for up/extended."""
        # lm: list of 21 NormalizedLandmark with x,y in [0,1]
        def up(tip_idx, pip_idx):
            return lm[tip_idx].y < lm[pip_idx].y - 0.02
        thumb = lm[4].x < lm[3].x  # mirror-friendly heuristic for right hand; good enough
        index = up(8, 6)
        middle = up(12, 10)
        ring = up(16, 14)
        pinky = up(20, 18)
        return thumb, index, middle, ring, pinky

    @staticmethod
    def _palm_facing_camera(lm) -> bool:
        # Rough check: wrist (0) → index_mcp (5) and pinky_mcp (17) form palm plane.
        # Z axis of mediapipe is depth (negative = closer). If index_mcp.z and pinky_mcp.z
        # are both close to wrist.z (small diff), palm is roughly frontal.
        wrist_z = lm[0].z
        return (lm[5].z - wrist_z < 0.05) and (lm[17].z - wrist_z < 0.05)

    def _classify(self, lm) -> str:
        if not self._palm_facing_camera(lm):
            return "unknown"
        t, i, m, r, p = self._finger_states(lm)

        if not any((t, i, m, r, p)):
            return "stop"                              # fist
        if all((t, i, m, r, p)):
            return "forward"                           # open palm
        if i and m and r and p and not t:
            return "walk"                              # 4 fingers up, thumb folded
        if i and m and not r and not p:
            return "sit"                               # V sign
        if i and m and r and not p:
            return "stand"                             # index+middle+ring up
        if p and not i and not m and not r:
            return "tail_wag"
        if i and not m and not r and not p:
            # direction by index tip vs. wrist x
            return "left" if lm[8].x < lm[0].x else "right"
        if not t and not i and not m and not r and p:
            return "tail_wag"
        if t and not i and not m and not r and not p:
            return "backward"                          # thumb down (we don't distinguish up/down strictly)
        return "unknown"

    # ---------- public API ----------

    def process(self, frame_bgr: np.ndarray) -> Optional[str]:
        """Run one frame through MediaPipe. Returns a gesture name only when the
        current sign has been held long enough AND cooldown has elapsed since the
        last emission — so AI worker can forward it directly to the UART bridge."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)

        now = time.time()
        if not result.hand_landmarks:
            self._current = None
            self._held_since = 0.0
            return None

        gesture = self._classify(result.hand_landmarks[0])
        if gesture == "unknown":
            self._current = None
            self._held_since = 0.0
            return None

        if gesture != self._current:
            self._current = gesture
            self._held_since = now
            return None

        held_for = now - self._held_since
        if held_for < settings.gesture_hold_s:
            return None

        if (
            self._last_emitted == gesture
            and (now - self._last_emit_ts) < settings.gesture_cooldown_s
        ):
            return None

        self._last_emitted = gesture
        self._last_emit_ts = now
        return gesture
