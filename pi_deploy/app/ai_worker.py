"""
AI worker thread. Pulls frames from the current FrameSource, runs face +
gesture, and pushes recognised gestures through the ESP32 UART bridge.

Also maintains the latest annotated frame as a JPEG so the HTTP /video_feed
endpoint can stream it.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np

from .camera_source import FrameSource, PiCameraSource, RemoteFrameSource
from .config import settings
from .gesture_map import dispatch, walking_effect
from .robot_state import store
from .uart_bridge import UartBridge, get_bridge

log = logging.getLogger(__name__)


class AiWorker:
    def __init__(self):
        self._life_lock = threading.Lock()     # guards start/stop lifecycle
        self._mode_lock = threading.Lock()     # guards set_mode / source swap
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

        self._source: Optional[FrameSource] = None
        self._bridge: UartBridge = get_bridge()

        # Latest annotated frame as JPEG bytes for MJPEG stream
        self._latest_jpeg: Optional[bytes] = None
        self._jpeg_lock = threading.Lock()

        # Lazy — face/gesture models loaded on start() so import time stays fast
        self._face = None
        self._gesture = None

        self.mode = settings.default_mode
        self.remote_source = RemoteFrameSource()  # always exists; used in streaming modes

    def is_running(self) -> bool:
        t = self._thread
        return t is not None and t.is_alive()

    # ---------- lifecycle ----------

    def start(self) -> None:
        with self._life_lock:
            if self.is_running():
                return
            self._ensure_models()
            # Fresh identity vote + gesture state so old frames don't auto-authorize
            if self._face is not None:
                self._face.reset()
            self._open_source_for_mode(self.mode)
            store.update(uart_connected=not self._bridge.is_mock, ai_enabled=True)
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="ai-worker", daemon=True)
            self._thread.start()
            log.info("AI worker started in mode=%s", self.mode)

    def stop(self) -> None:
        # Take the lock only long enough to capture refs + null them out,
        # then do the slow join/close work without holding it.
        with self._life_lock:
            self._stop.set()
            th = self._thread
            self._thread = None
            source = self._source
            self._source = None
        if th is not None:
            th.join(timeout=2.0)
        if source is not None:
            try:
                source.close()
            except Exception:
                log.exception("source close failed")
        store.update(ai_enabled=False, fps=0.0)

    def _ensure_models(self) -> None:
        if self._face is None:
            from .ai.face import FaceRecognizer
            self._face = FaceRecognizer()
        if self._gesture is None:
            try:
                from .ai.gesture import GestureClassifier
                self._gesture = GestureClassifier()
            except Exception as e:
                log.warning("Gesture model unavailable: %s", e)
                self._gesture = None

    # ---------- mode switching ----------

    def set_mode(self, mode: str) -> None:
        if mode not in ("pi_camera", "laptop_client", "laptop_stream"):
            raise ValueError(mode)
        with self._mode_lock:
            self.mode = mode
            # Only touch the camera if the worker is actually running — otherwise
            # we'd grab and hold the webcam with nobody reading from it.
            if self.is_running():
                self._open_source_for_mode(mode)
                if self._face is not None:
                    self._face.reset()
        store.update(mode=mode)
        store.log_line(f"mode → {mode}")

    def _open_source_for_mode(self, mode: str) -> None:
        if self._source is not None:
            self._source.close()
            self._source = None

        if mode == "pi_camera":
            try:
                self._source = PiCameraSource()
            except Exception as e:
                log.error("Pi camera open failed: %s", e)
                self._source = None
        elif mode == "laptop_stream":
            self._source = self.remote_source
        elif mode == "laptop_client":
            # No local inference source — the laptop runs AI and posts commands directly.
            # We still keep the remote_source around so the web UI can optionally show
            # a preview the laptop uploads.
            self._source = self.remote_source

    # ---------- frame push (from WS endpoint) ----------

    def push_remote_jpeg(self, jpeg_bytes: bytes) -> bool:
        return self.remote_source.push_jpeg(jpeg_bytes)

    # ---------- MJPEG output ----------

    def latest_jpeg(self) -> Optional[bytes]:
        with self._jpeg_lock:
            return self._latest_jpeg

    # ---------- main loop ----------

    def _run(self) -> None:
        t_prev = time.time()
        frames = 0
        while not self._stop.is_set():
            try:
                advanced = self._iterate()
            except Exception:
                log.exception("ai worker iteration failed")
                time.sleep(0.1)
                continue

            if not advanced:
                continue

            frames += 1
            now = time.time()
            if now - t_prev >= 1.0:
                store.update(fps=round(frames / (now - t_prev), 1))
                frames = 0
                t_prev = now

    def _iterate(self) -> bool:
        """Run one frame through the pipeline. Returns True if a real frame was processed."""
        # Snapshot source under the mode lock to avoid racing with set_mode / stop
        with self._mode_lock:
            source = self._source
            mode = self.mode

        if source is None:
            time.sleep(0.1)
            return False

        frame = source.read()
        if frame is None:
            time.sleep(0.03)
            return False

        authorized_name: Optional[str] = None
        box = None
        gesture: Optional[str] = None

        # In laptop_client mode the laptop handles AI — we only show a preview if
        # the laptop optionally uploads frames. No inference here.
        if mode != "laptop_client":
            try:
                authorized_name, box = self._face.process(frame)
            except Exception:
                log.exception("face.process failed")

            if authorized_name and self._gesture is not None:
                try:
                    gesture = self._gesture.process(frame)
                except Exception:
                    log.exception("gesture.process failed")

        if gesture and authorized_name:
            if dispatch(gesture, self._bridge):
                store.update(
                    last_gesture=gesture,
                    last_command=self._bridge.last_command,
                    last_command_ts=time.time(),
                    authorized=authorized_name,
                    **walking_effect(gesture),
                )
                store.log_line(f"{authorized_name} → {gesture} → {self._bridge.last_command}")

        # Debounced in robot_state.update — only broadcasts when value actually changes.
        store.update(authorized=authorized_name)

        # ---- annotate + encode for streaming ----
        try:
            annotated = self._annotate(frame, authorized_name, box, gesture)
            ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                with self._jpeg_lock:
                    self._latest_jpeg = buf.tobytes()
        except Exception:
            log.exception("annotate/encode failed")

        return True

    def _annotate(self, frame, name, box, gesture):
        out = frame.copy()
        label = name or "no face"
        color = (0, 200, 0) if name else (0, 0, 200)
        if box is not None:
            x1, y1, x2, y2 = box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{label}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if gesture:
            cv2.putText(out, f"gesture: {gesture}", (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        cv2.putText(out, f"mode: {self.mode}", (10, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        return out


_worker: Optional[AiWorker] = None


def get_worker() -> AiWorker:
    global _worker
    if _worker is None:
        _worker = AiWorker()
    return _worker
