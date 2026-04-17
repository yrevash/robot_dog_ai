"""
Unified frame source. Swappable at runtime when the UI changes mode.

- `PiCameraSource`   — OpenCV capture from a USB webcam plugged into the Pi.
- `RemoteFrameSource` — frames pushed over WebSocket from a laptop/phone browser
                         (modes "laptop_stream" and "laptop_client" with preview).
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np

from .config import settings


class FrameSource:
    name: str = "base"

    def read(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class PiCameraSource(FrameSource):
    name = "pi_camera"

    def __init__(self, index: int = None):
        index = settings.pi_camera_index if index is None else index
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.frame_height)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open USB webcam at index {index}")

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def close(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass


class RemoteFrameSource(FrameSource):
    """Thread-safe single-slot buffer. Latest frame wins."""
    name = "remote"

    def __init__(self):
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._last_push = 0.0

    def push_jpeg(self, jpeg_bytes: bytes) -> bool:
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return False
        with self._lock:
            self._frame = img
            self._last_push = time.time()
        return True

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._frame is None:
                return None
            # Stale frame (>2s old) treated as no frame
            if time.time() - self._last_push > 2.0:
                return None
            return self._frame.copy()

    def close(self) -> None:
        with self._lock:
            self._frame = None
