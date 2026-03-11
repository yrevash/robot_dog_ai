#!/usr/bin/env python3
"""
Desktop control center for face enrollment + gesture commands for robot dog.

Run:
    python face_control_center.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import threading
import time
import tkinter as tk
import urllib.error
import urllib.request
from collections import Counter, deque
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, Optional, Set, Tuple

import cv2

try:
    from PIL import Image, ImageTk
except Exception as exc:
    raise SystemExit(
        "Missing dependency Pillow. Install with:\n"
        "    python -m pip install pillow\n"
        f"Original error: {exc}"
    )

try:
    import mediapipe as mp
except Exception:
    mp = None

try:
    from mediapipe.tasks import python as mp_tasks_python
    from mediapipe.tasks.python import vision as mp_vision
except Exception:
    mp_tasks_python = None
    mp_vision = None

try:
    import winsound
except Exception:
    winsound = None

sys.path.insert(0, str(Path(__file__).resolve().parent))  # ensure src/ is on path
import face_embedding as fe
from power_state import PowerManager, PowerState


GESTURE_COMMANDS = {
    "FORWARD": "forward",
    "BACKWARD": "backward",
    "LEFT": "left",
    "RIGHT": "right",
    "BARK": "bark",
    "STAND": "stand",
    "TAIL_WAG": "tail_wag",
    "WALK": "walk",
    "SIT": "sit",
    "STOP": "stop",
    "GREET": "greet",
}

HAND_LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
HAND_LANDMARKER_FILE = "hand_landmarker.task"


def _runtime_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent  # project root (src/ → Revo_Robot_AI/)


def _configure_runtime_paths() -> None:
    base = _runtime_base_dir()
    fe.KNOWN_FACES_DIR = base / "data" / "known_faces"
    fe.MODELS_DIR = base / "models"
    fe.DB_FILE = base / "data" / "face_db.npz"

    external_detector = fe.MODELS_DIR / "face_detection_yunet_2023mar.onnx"
    external_recognizer = fe.MODELS_DIR / "face_recognition_sface_2021dec.onnx"
    bundled_models = base / "_internal" / "models"
    bundled_detector = bundled_models / "face_detection_yunet_2023mar.onnx"
    bundled_recognizer = bundled_models / "face_recognition_sface_2021dec.onnx"

    fe.DETECTOR_MODEL = (
        bundled_detector
        if bundled_detector.exists()
        else (external_detector if external_detector.exists() else external_detector)
    )
    fe.RECOGNIZER_MODEL = (
        bundled_recognizer
        if bundled_recognizer.exists()
        else (external_recognizer if external_recognizer.exists() else external_recognizer)
    )
    fe.MODEL_URLS = {
        fe.DETECTOR_MODEL: "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        fe.RECOGNIZER_MODEL: "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    }
    fe.KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)


class FaceControlCenter:
    def __init__(self, root: tk.Tk) -> None:
        _configure_runtime_paths()
        self.root = root
        self.root.title("REVO — Robot Dog Control Center")
        self.root.geometry("1280x860")

        self.camera_index = 0
        self.width = 640
        self.height = 480

        self.threshold = 0.42
        self.margin = 0.06
        self.centroid_threshold = 0.40
        self.min_face_area_ratio = 0.05
        self.track_timeout = 2.0
        self.face_cooldown = 3.0
        self.history = deque(maxlen=6)
        self.stable_count = 4
        self.light_normalize = False

        self.gesture_history = deque(maxlen=4)
        self.gesture_vote_count = 2
        self.detected_gesture_history = deque(maxlen=3)
        self.detected_gesture_vote_count = 2
        self.command_cooldown = 1.2
        self.last_command = ""
        self.last_command_time = 0.0
        self.greet_cooldown = 20.0
        self.last_greet_times: Dict[str, float] = {}
        self.runtime_ready = False
        self.frame_counter = 0
        self.gesture_detect_interval = 1
        self.last_raw_gesture: Optional[str] = None
        self.last_hand_points: Optional[list[Tuple[int, int]]] = None
        self.last_hand_points_time = 0.0
        self.hand_overlay_hold_sec = 0.55
        self.controller_hold_sec = 0.90
        self.last_controller_name: Optional[str] = None
        self.last_controller_face_bbox: Optional[Tuple[int, int, int, int]] = None
        self.last_controller_time = 0.0
        self.hand_max_inference_size = 320

        self.cap: Optional[cv2.VideoCapture] = None
        self.detector = None
        self.recognizer = None
        self.running = False
        self.recognition_on = False
        self.build_thread: Optional[threading.Thread] = None

        self.last_frame = None
        self.last_proc_frame = None
        self.last_faces = []

        self.db_embeddings = None
        self.db_names = None
        self.db_centroids = None
        self.db_centroid_names = None
        self.db_loaded = False

        self.last_face_trigger_times: Dict[str, float] = {}
        self.authorized_until_by_name: Dict[str, float] = {}

        self.hand_model_path = fe.MODELS_DIR / HAND_LANDMARKER_FILE
        self.gesture_backend = None
        if mp is not None and hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self.gesture_backend = "solutions"
        elif mp is not None and mp_tasks_python is not None and mp_vision is not None:
            self.gesture_backend = "tasks"

        self.mp_hands_module = mp.solutions.hands if self.gesture_backend == "solutions" else None
        self.mp_draw_module = mp.solutions.drawing_utils if self.gesture_backend == "solutions" else None
        self.hands_engine = None
        self.task_latest_result = None
        self.task_latest_timestamp_ms = -1
        self.task_result_lock = threading.Lock()
        self.hand_connection_pairs: list[Tuple[int, int]] = []
        if self.gesture_backend == "solutions" and self.mp_hands_module is not None:
            self.hand_connection_pairs = [(int(a), int(b)) for (a, b) in self.mp_hands_module.HAND_CONNECTIONS]
        elif mp_vision is not None:
            self.hand_connection_pairs = [
                (int(c.start), int(c.end)) for c in mp_vision.HandLandmarksConnections.HAND_CONNECTIONS
            ]

        self.person_name_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready")
        self.detect_var = tk.StringVar(value="Faces: 0")
        self.auth_var = tk.StringVar(value="Authorized: None")
        self.gesture_var = tk.StringVar(value="Gesture: None")
        self.command_var = tk.StringVar(value="Last command: None")

        self.multi_face_var = tk.BooleanVar(value=True)
        self.gesture_enabled_var = tk.BooleanVar(value=True)
        self.greet_enabled_var = tk.BooleanVar(value=True)
        self.unmirror_camera_var = tk.BooleanVar(value=True)
        if self.gesture_backend is None:
            self.gesture_enabled_var.set(False)

        default_bark = _runtime_base_dir() / "bark.wav"
        self.bark_audio_var = tk.StringVar(value=str(default_bark))
        self.iot_url_var = tk.StringVar(value="")

        # Power management
        self.power_state_var = tk.StringVar(value="ACTIVE")
        self.fps_var = tk.StringVar(value="FPS: --")
        self.idle_var = tk.StringVar(value="Idle: 0s")
        self.uptime_var = tk.StringVar(value="Uptime: 0s")
        self._start_time = time.time()
        self._fps_frames: deque = deque(maxlen=30)
        self._power_save_frame_skip = 10  # only process every Nth frame in power save

        self.power_manager = PowerManager(
            idle_to_save_sec=900.0,   # 15 minutes
            save_to_off_sec=1800.0,   # 30 minutes
            on_enter_active=self._on_power_active,
            on_enter_power_save=self._on_power_save,
            on_enter_power_off=self._on_power_off,
        )

        self._build_ui()
        self._refresh_person_suggestions()
        self._load_database(show_info=False)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<KeyPress-w>", lambda e: self._wake_button_pressed())
        self.root.bind("<KeyPress-W>", lambda e: self._wake_button_pressed())
        self.root.after(20, self._update_loop)

    # ── Power management callbacks ─────────────────────────────────────────

    def _on_power_active(self) -> None:
        """Resume full operation."""
        self.power_state_var.set("ACTIVE")
        if not self.running:
            self.start_camera()
        if self.db_loaded and not self.recognition_on:
            self.start_recognition()

    def _on_power_save(self) -> None:
        """Reduce frame rate, disable gesture detection. Camera stays open."""
        self.power_state_var.set("POWER SAVE")
        self.gesture_history.clear()
        self.detected_gesture_history.clear()
        self.last_raw_gesture = None

    def _on_power_off(self) -> None:
        """Release camera, stop all inference."""
        self.power_state_var.set("POWER OFF")
        self.stop_camera()

    def _wake_button_pressed(self) -> None:
        """Handle wake from keyboard (W) or button click."""
        self.power_manager.wake()

    def _format_duration(self, seconds: float) -> str:
        """Format seconds as Xh Ym Zs."""
        s = int(seconds)
        if s < 60:
            return f"{s}s"
        m, s = divmod(s, 60)
        if m < 60:
            return f"{m}m {s}s"
        h, m = divmod(m, 60)
        return f"{h}h {m}m"

    # ── UI ─────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # ── Main body: sidebar + video ────────────────────────────────────
        body = ttk.Frame(self.root, padding=5)
        body.pack(fill=tk.BOTH, expand=True)

        # Left sidebar
        sidebar = ttk.Frame(body, width=340)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        sidebar.pack_propagate(False)

        # Right: video feed
        right = ttk.Frame(body)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.video_label = ttk.Label(right)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # ── Power State Section ───────────────────────────────────────────
        power_frame = ttk.LabelFrame(sidebar, text=" Power Management ", padding=8)
        power_frame.pack(fill=tk.X, pady=(0, 6))

        self.power_indicator = tk.Label(
            power_frame, textvariable=self.power_state_var,
            font=("Helvetica", 14, "bold"), fg="white", bg="#2ecc71",
            padx=10, pady=4,
        )
        self.power_indicator.pack(fill=tk.X)

        power_info = ttk.Frame(power_frame)
        power_info.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(power_info, textvariable=self.idle_var).pack(side=tk.LEFT)
        ttk.Label(power_info, textvariable=self.uptime_var).pack(side=tk.RIGHT)

        self.wake_btn = ttk.Button(power_frame, text="Wake (W)", command=self._wake_button_pressed)
        self.wake_btn.pack(fill=tk.X, pady=(4, 0))

        # ── Status Section ────────────────────────────────────────────────
        status_frame = ttk.LabelFrame(sidebar, text=" System Status ", padding=8)
        status_frame.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(status_frame, textvariable=self.fps_var).pack(anchor="w")
        ttk.Label(status_frame, textvariable=self.status_var).pack(anchor="w")
        ttk.Label(status_frame, textvariable=self.detect_var).pack(anchor="w")
        ttk.Label(status_frame, textvariable=self.auth_var).pack(anchor="w")
        ttk.Label(status_frame, textvariable=self.gesture_var).pack(anchor="w")
        ttk.Label(status_frame, textvariable=self.command_var).pack(anchor="w")

        # ── Camera Controls ───────────────────────────────────────────────
        cam_frame = ttk.LabelFrame(sidebar, text=" Camera Controls ", padding=8)
        cam_frame.pack(fill=tk.X, pady=(0, 6))

        cam_btns = ttk.Frame(cam_frame)
        cam_btns.pack(fill=tk.X)
        ttk.Button(cam_btns, text="Start Camera", command=self.start_camera).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(cam_btns, text="Stop Camera", command=self.stop_camera).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)

        rec_btns = ttk.Frame(cam_frame)
        rec_btns.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(rec_btns, text="Start Recognition", command=self.start_recognition).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(rec_btns, text="Pause Recognition", command=self.pause_recognition).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)

        ttk.Checkbutton(cam_frame, text="Multi-face recognition", variable=self.multi_face_var).pack(anchor="w", pady=(4, 0))
        ttk.Checkbutton(cam_frame, text="Fix mirrored camera", variable=self.unmirror_camera_var).pack(anchor="w")
        ttk.Checkbutton(cam_frame, text="Enable gesture commands", variable=self.gesture_enabled_var).pack(anchor="w")
        ttk.Checkbutton(cam_frame, text="Greet authorized person", variable=self.greet_enabled_var).pack(anchor="w")

        # ── Enrollment Section ────────────────────────────────────────────
        enroll_frame = ttk.LabelFrame(sidebar, text=" Enrollment ", padding=8)
        enroll_frame.pack(fill=tk.X, pady=(0, 6))

        name_row = ttk.Frame(enroll_frame)
        name_row.pack(fill=tk.X)
        ttk.Label(name_row, text="Person:").pack(side=tk.LEFT)
        self.person_combo = ttk.Combobox(name_row, width=18, textvariable=self.person_name_var)
        self.person_combo.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)

        enroll_btns = ttk.Frame(enroll_frame)
        enroll_btns.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(enroll_btns, text="Capture", command=self.capture_sample).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(enroll_btns, text="Import", command=self.import_photos).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)

        db_btns = ttk.Frame(enroll_frame)
        db_btns.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(db_btns, text="Build DB", command=self.build_database).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(db_btns, text="Reload DB", command=self._load_database).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)

        # ── IoT / Audio Section ───────────────────────────────────────────
        iot_frame = ttk.LabelFrame(sidebar, text=" IoT / Audio ", padding=8)
        iot_frame.pack(fill=tk.X, pady=(0, 6))

        iot_row = ttk.Frame(iot_frame)
        iot_row.pack(fill=tk.X)
        ttk.Label(iot_row, text="URL:").pack(side=tk.LEFT)
        ttk.Entry(iot_row, textvariable=self.iot_url_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        ttk.Button(iot_frame, text="Select Bark Audio", command=self.choose_bark_audio).pack(fill=tk.X, pady=(4, 0))

    def _refresh_person_suggestions(self) -> None:
        people = sorted(p.name for p in fe.KNOWN_FACES_DIR.glob("*") if p.is_dir())
        self.person_combo["values"] = people

    def _ensure_runtime(self) -> bool:
        if self.runtime_ready:
            return True
        try:
            fe.check_opencv_requirements()
            fe.ensure_models(auto_download=True)
            self.runtime_ready = True
            return True
        except BaseException as exc:
            messagebox.showerror("Runtime Error", str(exc))
            return False

    def _ensure_hand_model(self) -> bool:
        if self.hand_model_path.exists():
            return True
        try:
            self.hand_model_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(HAND_LANDMARKER_URL, str(self.hand_model_path))
            return True
        except Exception as exc:
            self.status_var.set(f"Hand model download failed: {exc}")
            return False

    def choose_bark_audio(self) -> None:
        path = filedialog.askopenfilename(
            title="Select bark audio",
            filetypes=[("Audio Files", "*.wav;*.mp3;*.ogg"), ("All Files", "*.*")],
        )
        if path:
            self.bark_audio_var.set(path)
            self.status_var.set(f"Bark audio set: {path}")

    def start_camera(self) -> None:
        if self.running:
            return
        if not self._ensure_runtime():
            return

        self.status_var.set("Starting camera...")
        self.root.update_idletasks()

        if hasattr(cv2, "CAP_DSHOW"):
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap.release()
                self.cap = cv2.VideoCapture(self.camera_index)
        else:
            self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if not self.cap.isOpened():
            self.cap = None
            messagebox.showerror("Camera Error", "Cannot open camera.")
            return

        if self.detector is None:
            self.detector = fe.create_detector((self.width, self.height), score_threshold=0.9)
        else:
            try:
                self.detector.setInputSize((self.width, self.height))
            except Exception:
                self.detector = fe.create_detector((self.width, self.height), score_threshold=0.9)

        if self.recognizer is None:
            self.recognizer = fe.create_recognizer()

        # Pre-initialize gesture engine so first detected hand does not stall the UI.
        if self.gesture_enabled_var.get():
            if not self._ensure_hands_engine():
                self.status_var.set("Camera started (gesture disabled)")
            else:
                self.status_var.set("Camera started")
        else:
            self.status_var.set("Camera started")

        self.running = True
        self.frame_counter = 0
        self.last_raw_gesture = None

    def stop_camera(self) -> None:
        self.running = False
        self.recognition_on = False
        self.history.clear()
        self.gesture_history.clear()
        self.detected_gesture_history.clear()
        self.last_raw_gesture = None
        self.last_hand_points = None
        self.last_hand_points_time = 0.0
        self.last_controller_name = None
        self.last_controller_face_bbox = None
        self.last_controller_time = 0.0
        with self.task_result_lock:
            self.task_latest_result = None
            self.task_latest_timestamp_ms = -1
        self.authorized_until_by_name.clear()
        self.last_face_trigger_times.clear()
        self.last_greet_times.clear()
        if self.hands_engine is not None:
            try:
                self.hands_engine.close()
            except Exception:
                pass
            self.hands_engine = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.status_var.set("Camera stopped")

    def start_recognition(self) -> None:
        if not self.running:
            self.start_camera()
            if not self.running:
                return
        self._load_database(show_info=False)
        if not self.db_loaded:
            messagebox.showwarning("No Database", "Build database first.")
            return
        self.gesture_history.clear()
        self.detected_gesture_history.clear()
        self.last_raw_gesture = None
        self.last_hand_points = None
        self.last_hand_points_time = 0.0
        self.last_controller_name = None
        self.last_controller_face_bbox = None
        self.last_controller_time = 0.0
        with self.task_result_lock:
            self.task_latest_result = None
            self.task_latest_timestamp_ms = -1
        self.recognition_on = True
        self.status_var.set("Recognition active")

    def pause_recognition(self) -> None:
        self.recognition_on = False
        self.history.clear()
        self.gesture_history.clear()
        self.detected_gesture_history.clear()
        self.last_raw_gesture = None
        self.last_hand_points = None
        self.last_hand_points_time = 0.0
        self.last_controller_name = None
        self.last_controller_face_bbox = None
        self.last_controller_time = 0.0
        with self.task_result_lock:
            self.task_latest_result = None
            self.task_latest_timestamp_ms = -1
        self.authorized_until_by_name.clear()
        self.status_var.set("Recognition paused")

    def _load_database(self, show_info: bool = True) -> None:
        try:
            data = fe.load_db(fe.DB_FILE)
        except (SystemExit, Exception) as exc:
            self.db_loaded = False
            self.db_embeddings = None
            self.db_names = None
            self.db_centroids = None
            self.db_centroid_names = None
            if show_info:
                self.status_var.set(f"Database unavailable: {exc}")
            return

        self.db_embeddings, self.db_names, self.db_centroids, self.db_centroid_names = data
        self.db_loaded = True
        if show_info:
            identities = sorted(set(self.db_names.tolist()))
            self.status_var.set(f"Database loaded ({len(identities)} identities)")

    def _next_sample_path(self, person_dir: Path) -> Path:
        existing = sorted(person_dir.glob("*.jpg"))
        next_idx = len(existing) + 1
        return person_dir / f"{next_idx:03d}.jpg"

    def capture_sample(self) -> None:
        if not self.running or self.recognizer is None or self.last_proc_frame is None:
            messagebox.showwarning("Capture", "Start camera first.")
            return

        person = fe.sanitize_name(self.person_name_var.get())
        if not person:
            messagebox.showwarning("Capture", "Enter person name first.")
            return

        if len(self.last_faces) != 1:
            messagebox.showwarning("Capture", "Need exactly one face in frame.")
            return

        face = self.last_faces[0]
        ok, msg = fe.face_quality_ok(face, self.last_proc_frame.shape, min_area_ratio=0.08)
        if not ok:
            messagebox.showwarning("Capture", f"Face quality check failed: {msg}")
            return

        aligned = self.recognizer.alignCrop(self.last_proc_frame, face)
        person_dir = fe.KNOWN_FACES_DIR / person
        person_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._next_sample_path(person_dir)
        cv2.imwrite(str(out_path), aligned)
        self._refresh_person_suggestions()
        self.status_var.set(f"Saved sample: {out_path}")

    def import_photos(self) -> None:
        person = fe.sanitize_name(self.person_name_var.get())
        if not person:
            messagebox.showwarning("Import", "Enter person name first.")
            return

        paths = filedialog.askopenfilenames(
            title="Select face images",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.webp")],
        )
        if not paths:
            return

        person_dir = fe.KNOWN_FACES_DIR / person
        person_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(person_dir.glob("*.jpg")))

        copied = 0
        for src in paths:
            existing += 1
            dst = person_dir / f"import_{existing:03d}.jpg"
            shutil.copy2(src, dst)
            copied += 1

        self._refresh_person_suggestions()
        self.status_var.set(f"Imported {copied} photos for {person}")

    def build_database(self) -> None:
        if self.build_thread and self.build_thread.is_alive():
            return

        def worker() -> None:
            try:
                args = argparse.Namespace(
                    db=fe.DB_FILE,
                    det_score=0.9,
                    no_download=False,
                    augment_lighting=True,
                )
                fe.run_build(args)
                self.root.after(0, self._on_build_done, None)
            except BaseException as exc:
                self.root.after(0, self._on_build_done, str(exc))

        self.status_var.set("Building database...")
        self.build_thread = threading.Thread(target=worker, daemon=True)
        self.build_thread.start()

    def _on_build_done(self, error: Optional[str]) -> None:
        if error:
            self.status_var.set(f"Build failed: {error}")
            messagebox.showerror("Build Error", error)
            return
        self._load_database(show_info=False)
        identities = sorted(set(self.db_names.tolist())) if self.db_loaded else []
        self.status_var.set(f"Database rebuilt. Identities: {', '.join(identities) if identities else 'none'}")

    def _send_iot_async(self, payload: Dict[str, object]) -> None:
        url = self.iot_url_var.get().strip()
        if not url:
            return

        def worker() -> None:
            try:
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    url=url,
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=2.0):
                    pass
            except urllib.error.URLError as exc:
                self.root.after(0, lambda e=str(exc): self.status_var.set(f"IoT send failed: {e}"))
            except Exception as exc:
                self.root.after(0, lambda e=str(exc): self.status_var.set(f"IoT send failed: {e}"))

        threading.Thread(target=worker, daemon=True).start()

    def _play_bark_audio(self) -> None:
        bark_path = Path(self.bark_audio_var.get().strip())
        if not bark_path.exists():
            self.status_var.set("Bark audio file not found.")
            return
        if winsound is None:
            self.status_var.set("Bark playback unsupported on this OS.")
            return
        try:
            winsound.PlaySound(str(bark_path), winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as exc:
            self.status_var.set(f"Bark playback failed: {exc}")

    def _send_robot_command(self, person: str, command_key: str, source: str) -> None:
        command = GESTURE_COMMANDS.get(command_key, command_key.lower())
        payload = {
            "person": person,
            "command": command,
            "source": source,
            "timestamp": time.time(),
        }
        print(f"[ROBOT CMD] {payload}")
        self._send_iot_async(payload)

        if command_key == "BARK":
            self._play_bark_audio()

        self.command_var.set(f"Last command: {command} (by {person})")

    def _authorize_votes(self, now: float) -> None:
        if len(self.history) < self.history.maxlen:
            return

        vote_counts: Counter[str] = Counter()
        for names_set in self.history:
            vote_counts.update(names_set)

        for person, count in vote_counts.items():
            if count < self.stable_count:
                continue

            newly_authorized = person not in self.authorized_until_by_name or now > self.authorized_until_by_name[person]
            self.authorized_until_by_name[person] = now + self.track_timeout

            last_face_time = self.last_face_trigger_times.get(person, 0.0)
            if (now - last_face_time) > self.face_cooldown:
                self.last_face_trigger_times[person] = now
                fe.robot_action(person)

            if self.greet_enabled_var.get() and newly_authorized:
                last_greet = self.last_greet_times.get(person, 0.0)
                if (now - last_greet) > self.greet_cooldown:
                    self.last_greet_times[person] = now
                    self._send_robot_command(person, "GREET", source="face")

    def _dist2d(self, p1, p2) -> float:
        dx = float(p1.x - p2.x)
        dy = float(p1.y - p2.y)
        return (dx * dx + dy * dy) ** 0.5

    def _palm_orientation(self, lmk, hand_label: str) -> Tuple[bool, bool]:
        # Heuristic orientation test from wrist->index_mcp and wrist->pinky_mcp.
        wx, wy = float(lmk[0].x), float(lmk[0].y)
        ix, iy = float(lmk[5].x), float(lmk[5].y)
        px, py = float(lmk[17].x), float(lmk[17].y)
        cross = (ix - wx) * (py - wy) - (iy - wy) * (px - wx)

        palm_confident = abs(cross) > 0.012
        if hand_label == "Right":
            return (cross < 0), palm_confident
        return (cross > 0), palm_confident

    def _is_palm_facing_camera(self, lmk, hand_label: str) -> bool:
        palm_facing, _ = self._palm_orientation(lmk, hand_label)
        return palm_facing

    def _finger_open_closed(self, lmk, tip_idx: int, pip_idx: int) -> Tuple[bool, bool]:
        # Add a dead zone to reduce flicker when a finger is near a boundary.
        delta = float(lmk[pip_idx].y - lmk[tip_idx].y)
        is_open = delta > 0.026
        is_closed = delta < 0.010
        return is_open, is_closed

    def _finger_states(self, lmk, hand_label: str) -> Dict[str, bool]:
        index_up, index_closed = self._finger_open_closed(lmk, 8, 6)
        middle_up, middle_closed = self._finger_open_closed(lmk, 12, 10)
        ring_up, ring_closed = self._finger_open_closed(lmk, 16, 14)
        pinky_up, pinky_closed = self._finger_open_closed(lmk, 20, 18)

        thumb_dx = float(lmk[4].x - lmk[3].x)
        thumb_dy = float(lmk[4].y - lmk[3].y)
        thumb_min = 0.042

        thumb_up = thumb_dy < (-thumb_min) and abs(thumb_dy) > (abs(thumb_dx) * 1.15)
        thumb_down = thumb_dy > thumb_min and abs(thumb_dy) > (abs(thumb_dx) * 1.15)
        thumb_left = thumb_dx < (-thumb_min) and abs(thumb_dx) > (abs(thumb_dy) * 1.15)
        thumb_right = thumb_dx > thumb_min and abs(thumb_dx) > (abs(thumb_dy) * 1.15)

        hand_size = max(self._dist2d(lmk[0], lmk[9]), 1e-6)
        thumb_folded = (
            (self._dist2d(lmk[4], lmk[5]) / hand_size) < 1.03
            and abs(thumb_dy) < 0.085
            and abs(thumb_dx) < 0.105
        )
        thumb_extended = (self._dist2d(lmk[4], lmk[2]) / hand_size) > 0.74
        thumb_lateral = abs(thumb_dx) > 0.05
        thumb_open = thumb_extended and (not thumb_folded)

        palm_facing, palm_confident = self._palm_orientation(lmk, hand_label)
        open_count = int(index_up) + int(middle_up) + int(ring_up) + int(pinky_up)
        closed_count = int(index_closed) + int(middle_closed) + int(ring_closed) + int(pinky_closed)
        unknown_count = 4 - open_count - closed_count

        all_four_closed = closed_count >= 3 and open_count == 0
        all_four_open = open_count == 4
        all_five_open = all_four_open and (thumb_open or thumb_lateral or thumb_up or thumb_down)
        mostly_closed = closed_count >= 3 and open_count <= 1

        return {
            "index": index_up,
            "middle": middle_up,
            "ring": ring_up,
            "pinky": pinky_up,
            "index_closed": index_closed,
            "middle_closed": middle_closed,
            "ring_closed": ring_closed,
            "pinky_closed": pinky_closed,
            "thumb_up": thumb_up,
            "thumb_down": thumb_down,
            "thumb_left": thumb_left,
            "thumb_right": thumb_right,
            "thumb_folded": thumb_folded,
            "thumb_extended": thumb_extended,
            "thumb_open": thumb_open,
            "thumb_lateral": thumb_lateral,
            "all_four_closed": all_four_closed,
            "all_four_open": all_four_open,
            "all_five_open": all_five_open,
            "mostly_closed": mostly_closed,
            "open_count": open_count,
            "closed_count": closed_count,
            "unknown_count": unknown_count,
            "palm_facing": palm_facing,
            "palm_confident": palm_confident,
        }

    def _classify_hand_gesture(self, lmk, hand_label: str) -> Optional[str]:
        s = self._finger_states(lmk, hand_label)
        # Index lean normalised by palm width (index MCP x − pinky MCP x).
        # dx5_norm < -0.07 → finger tips to the left of its own MCP → LEFT
        # dx5_norm > -0.07 → finger tips to the right of or at its MCP → RIGHT
        _palm_w = abs(float(lmk[5].x - lmk[17].x)) + 1e-6
        _idx_dx_norm = (float(lmk[8].x) - float(lmk[5].x)) / _palm_w
        index_pointing_left = _idx_dx_norm < -0.07
        index_pointing_right = _idx_dx_norm > -0.07

        # Strict mode: reject rotated/back-side palm to avoid false commands.
        if (not s["palm_confident"]) or (not s["palm_facing"]):
            return None

        # Default (non-trained) gesture map.
        # STOP: fist.
        if s["closed_count"] >= 4 and s["thumb_folded"]:
            return "STOP"

        # BARK: thumb-index pinch with remaining fingers up.
        hand_size = max(self._dist2d(lmk[0], lmk[9]), 1e-6)
        pinch = self._dist2d(lmk[4], lmk[8]) / hand_size
        if pinch < 0.40 and s["middle"] and s["ring"] and s["pinky"]:
            return "BARK"

        # BACKWARD: thumbs down, other fingers closed.
        if s["thumb_down"] and s["index_closed"] and s["middle_closed"] and s["ring_closed"] and s["pinky_closed"]:
            return "BACKWARD"

        # FORWARD: open palm (all 5 fingers up).
        if s["all_five_open"]:
            return "FORWARD"

        # WALK: 4 fingers up, thumb folded.
        if s["all_four_open"] and s["thumb_folded"]:
            return "WALK"

        # SIT: V sign (index + middle only).
        if s["index"] and s["middle"] and s["ring_closed"] and s["pinky_closed"]:
            return "SIT"

        # STAND: index + middle + ring up, pinky down.
        if s["index"] and s["middle"] and s["ring"] and s["pinky_closed"]:
            return "STAND"

        # TAIL_WAG: only pinky up.
        if (
            s["pinky"]
            and s["index_closed"]
            and s["middle_closed"]
            and s["ring_closed"]
            and s["thumb_folded"]
        ):
            return "TAIL_WAG"

        # LEFT/RIGHT: only index up, pointing direction decides command.
        if s["index"] and s["middle_closed"] and s["ring_closed"] and s["pinky_closed"]:
            if index_pointing_left:
                return "LEFT"
            if index_pointing_right:
                return "RIGHT"

        return None

    def _draw_task_landmarks(self, view: cv2.Mat, landmarks) -> None:
        h, w = view.shape[:2]
        for c in mp_vision.HandLandmarksConnections.HAND_CONNECTIONS:
            p1 = landmarks[c.start]
            p2 = landmarks[c.end]
            x1, y1 = int(p1.x * w), int(p1.y * h)
            x2, y2 = int(p2.x * w), int(p2.y * h)
            cv2.line(view, (x1, y1), (x2, y2), (255, 210, 40), 2, cv2.LINE_AA)
        for p in landmarks:
            x, y = int(p.x * w), int(p.y * h)
            cv2.circle(view, (x, y), 3, (255, 255, 255), -1, cv2.LINE_AA)

    def _cache_hand_overlay(self, points: list[Tuple[int, int]], now: float) -> None:
        self.last_hand_points = points
        self.last_hand_points_time = now

    def _draw_hand_overlay(self, view: cv2.Mat, points: list[Tuple[int, int]]) -> None:
        for a, b in self.hand_connection_pairs:
            if a < len(points) and b < len(points):
                cv2.line(view, points[a], points[b], (255, 210, 40), 2, cv2.LINE_AA)
        for px, py in points:
            cv2.circle(view, (px, py), 3, (255, 255, 255), -1, cv2.LINE_AA)
        if points:
            cv2.circle(view, points[0], 6, (0, 220, 255), 2)

    def _draw_cached_hand_overlay(self, view: cv2.Mat, now: float) -> None:
        if self.last_hand_points is None:
            return
        if (now - self.last_hand_points_time) > self.hand_overlay_hold_sec:
            self.last_hand_points = None
            return
        self._draw_hand_overlay(view, self.last_hand_points)

    def _draw_gesture_hud(self, view: cv2.Mat, gesture_label: str) -> None:
        h, _ = view.shape[:2]
        text = f"Gesture: {gesture_label}"
        org = (12, max(30, h - 20))
        cv2.rectangle(view, (8, org[1] - 24), (290, org[1] + 8), (0, 0, 0), -1)
        cv2.putText(
            view,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2,
        )

    def _on_task_result(self, result, _output_image, timestamp_ms: int) -> None:
        with self.task_result_lock:
            self.task_latest_result = result
            self.task_latest_timestamp_ms = int(timestamp_ms)

    def _gesture_roi(
        self,
        shape: Tuple[int, int, int],
        controller_face_bbox: Optional[Tuple[int, int, int, int]],
    ) -> Tuple[int, int, int, int]:
        h, w = shape[:2]
        if controller_face_bbox is None:
            return 0, 0, w, h

        x1, y1, x2, y2 = controller_face_bbox
        fw = max(1, x2 - x1)
        fh = max(1, y2 - y1)
        rx1 = max(0, int(x1 - 1.00 * fw))
        rx2 = min(w, int(x2 + 1.00 * fw))
        ry1 = max(0, int(y1 - 0.20 * fh))
        ry2 = min(h, int(y2 + 2.00 * fh))
        if rx2 <= rx1 or ry2 <= ry1:
            return 0, 0, w, h
        return rx1, ry1, rx2, ry2

    def _ensure_hands_engine(self) -> bool:
        if self.hands_engine is not None:
            return True
        if not self.gesture_enabled_var.get():
            return False

        if self.gesture_backend == "solutions" and self.mp_hands_module is not None:
            self.hands_engine = self.mp_hands_module.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.55,
                min_tracking_confidence=0.45,
            )
            return True

        if self.gesture_backend == "tasks":
            if not self._ensure_hand_model():
                messagebox.showwarning("Gesture Disabled", "Could not download hand gesture model.")
                self.gesture_enabled_var.set(False)
                return False
            try:
                base_options = mp_tasks_python.BaseOptions(model_asset_path=str(self.hand_model_path))
                options = mp_vision.HandLandmarkerOptions(
                    base_options=base_options,
                    running_mode=mp_vision.RunningMode.LIVE_STREAM,
                    result_callback=self._on_task_result,
                    num_hands=1,
                    min_hand_detection_confidence=0.55,
                    min_hand_presence_confidence=0.45,
                    min_tracking_confidence=0.45,
                )
                self.hands_engine = mp_vision.HandLandmarker.create_from_options(options)
                with self.task_result_lock:
                    self.task_latest_result = None
                    self.task_latest_timestamp_ms = -1
                return True
            except Exception as exc:
                messagebox.showwarning("Gesture Disabled", f"HandLandmarker init failed: {exc}")
                self.gesture_enabled_var.set(False)
                return False

        messagebox.showwarning(
            "Gesture Disabled",
            "MediaPipe gesture backend not available.\nInstall/update with:\npython -m pip install --upgrade mediapipe",
        )
        self.gesture_enabled_var.set(False)
        return False

    def _pick_hand_for_controller(
        self,
        hand_wrist_points: list[Tuple[int, int]],
        controller_face_bbox: Optional[Tuple[int, int, int, int]],
    ) -> Optional[int]:
        if not hand_wrist_points:
            return None
        if controller_face_bbox is None:
            return 0

        x1, y1, x2, y2 = controller_face_bbox
        fw = max(float(x2 - x1), 1.0)
        fh = max(float(y2 - y1), 1.0)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        face_size = max(fw, fh, 1.0)
        max_assoc_dist = face_size * 1.8
        max_assoc_dist_sq = max_assoc_dist * max_assoc_dist

        gate_x1 = x1 - (1.2 * fw)
        gate_x2 = x2 + (1.2 * fw)
        gate_y1 = y1 - (0.35 * fh)
        gate_y2 = y2 + (2.2 * fh)

        best_idx = None
        best_dist_sq = None
        for idx, (hx, hy) in enumerate(hand_wrist_points):
            if hx < gate_x1 or hx > gate_x2 or hy < gate_y1 or hy > gate_y2:
                continue
            dx = float(hx) - cx
            dy = float(hy) - cy
            dist_sq = dx * dx + dy * dy
            if best_dist_sq is None or dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_idx = idx

        if best_dist_sq is None or best_dist_sq > max_assoc_dist_sq:
            return None
        return best_idx

    def _stabilize_detected_gesture(self, raw_gesture: Optional[str]) -> Optional[str]:
        self.detected_gesture_history.append(raw_gesture)
        if raw_gesture is None:
            return None
        if len(self.detected_gesture_history) < 2:
            return raw_gesture

        counts = Counter(g for g in self.detected_gesture_history if g is not None and g != "UNKNOWN")
        if not counts:
            return raw_gesture
        best, count = counts.most_common(1)[0]
        if count < self.detected_gesture_vote_count:
            return "UNKNOWN"
        return best

    def _detect_gesture(
        self,
        frame_bgr: cv2.Mat,
        view: Optional[cv2.Mat] = None,
        controller_face_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[str]:
        if not self.gesture_enabled_var.get():
            return None
        if self.hands_engine is None:
            return None

        rx1, ry1, rx2, ry2 = self._gesture_roi(frame_bgr.shape, controller_face_bbox)
        roi_bgr = frame_bgr[ry1:ry2, rx1:rx2]
        if roi_bgr.size == 0:
            return None
        roi_h, roi_w = roi_bgr.shape[:2]
        infer_bgr = roi_bgr
        max_dim = max(roi_w, roi_h)
        if max_dim > self.hand_max_inference_size:
            scale = self.hand_max_inference_size / float(max_dim)
            infer_w = max(1, int(roi_w * scale))
            infer_h = max(1, int(roi_h * scale))
            infer_bgr = cv2.resize(roi_bgr, (infer_w, infer_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(infer_bgr, cv2.COLOR_BGR2RGB)

        if self.gesture_backend == "solutions":
            try:
                result = self.hands_engine.process(rgb)
            except Exception:
                return None
            if not result.multi_hand_landmarks:
                return None

            hands_data = []
            hand_wrist_points = []
            for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                hand_label = "Right"
                if result.multi_handedness and idx < len(result.multi_handedness):
                    hand_label = result.multi_handedness[idx].classification[0].label

                lmk = hand_landmarks.landmark
                gesture = self._classify_hand_gesture(lmk, hand_label)
                x = rx1 + int(lmk[0].x * roi_w)
                y = ry1 + int(lmk[0].y * roi_h)
                hands_data.append((hand_landmarks, gesture, x, y))
                hand_wrist_points.append((x, y))

            chosen_idx = self._pick_hand_for_controller(hand_wrist_points, controller_face_bbox)
            if chosen_idx is None:
                return None

            chosen_landmarks, chosen_gesture, wx, wy = hands_data[chosen_idx]
            points_full = [(rx1 + int(p.x * roi_w), ry1 + int(p.y * roi_h)) for p in chosen_landmarks.landmark]
            self._cache_hand_overlay(points_full, time.time())
            if view is not None and self.mp_draw_module is not None:
                self._draw_hand_overlay(view, points_full)
            return chosen_gesture if chosen_gesture else "UNKNOWN"
        elif self.gesture_backend == "tasks":
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            now_ms = int(time.time() * 1000)
            try:
                self.hands_engine.detect_async(mp_image, now_ms)
            except Exception:
                return None

            with self.task_result_lock:
                result = self.task_latest_result
                result_ts = self.task_latest_timestamp_ms
            if result is None:
                return None
            # Ignore very old async results.
            if result_ts >= 0 and (now_ms - result_ts) > 350:
                return None
            if not result.hand_landmarks:
                return None

            hands_data = []
            hand_wrist_points = []
            for idx, lmk in enumerate(result.hand_landmarks):
                hand_label = "Right"
                if result.handedness and idx < len(result.handedness) and result.handedness[idx]:
                    hand_label = result.handedness[idx][0].category_name

                gesture = self._classify_hand_gesture(lmk, hand_label)
                x = rx1 + int(lmk[0].x * roi_w)
                y = ry1 + int(lmk[0].y * roi_h)
                hands_data.append((lmk, gesture, x, y))
                hand_wrist_points.append((x, y))

            chosen_idx = self._pick_hand_for_controller(hand_wrist_points, controller_face_bbox)
            if chosen_idx is None:
                return None

            chosen_lmk, chosen_gesture, wx, wy = hands_data[chosen_idx]
            points_full = [(rx1 + int(p.x * roi_w), ry1 + int(p.y * roi_h)) for p in chosen_lmk]
            self._cache_hand_overlay(points_full, time.time())
            if view is not None:
                self._draw_hand_overlay(view, points_full)
            return chosen_gesture if chosen_gesture else "UNKNOWN"
        else:
            return None

    def _controller_name(self, authorized: Set[str], visible_area_by_name: Dict[str, float]) -> Optional[str]:
        if not authorized:
            return None

        visible_authorized = {n: a for n, a in visible_area_by_name.items() if n in authorized}
        if visible_authorized:
            return max(visible_authorized, key=visible_authorized.get)

        return sorted(authorized)[0]

    def _apply_gesture_command(
        self,
        now: float,
        gesture: Optional[str],
        controller: Optional[str],
    ) -> None:
        if gesture is None or gesture == "UNKNOWN" or controller is None:
            self.gesture_history.append(None)
            return

        self.gesture_history.append(gesture)
        if len(self.gesture_history) < self.gesture_history.maxlen:
            return

        counts = Counter(g for g in self.gesture_history if g is not None)
        if not counts:
            return

        cmd, count = counts.most_common(1)[0]
        if count < self.gesture_vote_count:
            return

        if cmd != self.last_command or (now - self.last_command_time) > self.command_cooldown:
            self.last_command = cmd
            self.last_command_time = now
            self.power_manager.report_activity()
            self._send_robot_command(controller, cmd, source="gesture")

    def _update_power_indicator(self) -> None:
        """Update the power state indicator color."""
        state = self.power_manager.state
        colors = {
            PowerState.ACTIVE: "#2ecc71",      # green
            PowerState.POWER_SAVE: "#f39c12",  # amber
            PowerState.POWER_OFF: "#e74c3c",   # red
        }
        self.power_indicator.configure(bg=colors.get(state, "#2ecc71"))
        self.idle_var.set(f"Idle: {self._format_duration(self.power_manager.idle_seconds)}")
        self.uptime_var.set(f"Up: {self._format_duration(time.time() - self._start_time)}")

    def _update_fps(self) -> None:
        """Compute rolling FPS."""
        now = time.monotonic()
        self._fps_frames.append(now)
        if len(self._fps_frames) >= 2:
            elapsed = self._fps_frames[-1] - self._fps_frames[0]
            if elapsed > 0:
                fps = (len(self._fps_frames) - 1) / elapsed
                self.fps_var.set(f"FPS: {fps:.1f}")

    def _render_power_overlay(self, view) -> None:
        """Draw overlay text on video when in power save or power off."""
        h, w = view.shape[:2]
        state = self.power_manager.state
        if state == PowerState.POWER_SAVE:
            overlay = view.copy()
            cv2.rectangle(overlay, (0, 0), (w, 50), (0, 180, 230), -1)
            cv2.addWeighted(overlay, 0.6, view, 0.4, 0, view)
            cv2.putText(view, "POWER SAVE — Monitoring for faces",
                        (15, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def _render_sleep_screen(self) -> None:
        """Show a static dark frame when in POWER_OFF."""
        dark = Image.new("RGB", (640, 480), (30, 30, 30))
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(dark)
        try:
            font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
            font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        except Exception:
            font_large = ImageFont.load_default()
            font_small = font_large
        draw.text((180, 190), "REVO — SLEEPING", fill=(200, 200, 200), font=font_large)
        draw.text((190, 240), "Press  W  or click Wake to resume", fill=(150, 150, 150), font=font_small)
        idle_text = f"Idle for {self._format_duration(self.power_manager.idle_seconds)}"
        draw.text((240, 280), idle_text, fill=(100, 100, 100), font=font_small)
        imgtk = ImageTk.PhotoImage(image=dark)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def _update_loop(self) -> None:
        try:
            # ── Power management tick ─────────────────────────────────────
            self.power_manager.tick()
            self._update_power_indicator()
            power_state = self.power_manager.state

            # In POWER_OFF: just show sleep screen, slow poll
            if power_state == PowerState.POWER_OFF:
                self._render_sleep_screen()
                self.status_var.set("POWER OFF — Press W to wake")
                self.root.after(500, self._update_loop)
                return

            # In POWER_SAVE: skip most frames
            if power_state == PowerState.POWER_SAVE and self.running:
                self.frame_counter += 1
                if (self.frame_counter % self._power_save_frame_skip) != 0:
                    self.root.after(20, self._update_loop)
                    return

            if self.running and self.cap is not None:
                ok, frame = self.cap.read()
                if ok:
                    self.frame_counter += 1
                    self._update_fps()
                    if self.unmirror_camera_var.get():
                        frame = cv2.flip(frame, 1)
                    proc = fe.normalize_lighting(frame) if self.light_normalize else frame
                    faces = fe.detect_faces(self.detector, proc)
                    self.last_frame = frame
                    self.last_proc_frame = proc
                    self.last_faces = faces

                    now = time.time()
                    view = frame.copy()
                    recognized_names: Set[str] = set()
                    visible_area_by_name: Dict[str, float] = {}
                    visible_face_box_by_name: Dict[str, Tuple[int, int, int, int]] = {}

                    # In POWER_SAVE: if any face detected, wake up
                    if power_state == PowerState.POWER_SAVE:
                        if faces:
                            self.power_manager.report_activity()
                        self._render_power_overlay(view)
                        rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(rgb)
                        imgtk = ImageTk.PhotoImage(image=img)
                        self.video_label.imgtk = imgtk
                        self.video_label.configure(image=imgtk)
                        self.status_var.set("POWER SAVE — Watching for faces")
                        self.root.after(20, self._update_loop)
                        return

                    if self.recognition_on and self.db_loaded:
                        if faces and (not self.multi_face_var.get()) and len(faces) != 1:
                            cv2.putText(
                                view,
                                "ONE FACE REQUIRED",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9,
                                (0, 0, 255),
                                2,
                            )
                        else:
                            for face in faces:
                                if fe.face_area_ratio(face, proc.shape) < self.min_face_area_ratio:
                                    continue

                                emb = fe.embedding_from_face(proc, face, self.recognizer)
                                if emb is None:
                                    continue

                                name, sample_score, centroid_score = fe.match_identity(
                                    emb,
                                    self.db_embeddings,
                                    self.db_names,
                                    self.db_centroids,
                                    self.db_centroid_names,
                                    threshold=self.threshold,
                                    margin=self.margin,
                                    centroid_threshold=self.centroid_threshold,
                                )
                                x1, y1, x2, y2 = fe.bbox(face)
                                area = max(0, (x2 - x1) * (y2 - y1))

                                if name != "Unknown":
                                    recognized_names.add(name)
                                    self.power_manager.report_activity()
                                    if area > visible_area_by_name.get(name, 0.0):
                                        visible_area_by_name[name] = float(area)
                                        visible_face_box_by_name[name] = (x1, y1, x2, y2)

                                color = (0, 200, 0) if name != "Unknown" else (0, 0, 255)
                                cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
                                cv2.putText(
                                    view,
                                    f"{name} {sample_score:.2f}/{centroid_score:.2f}",
                                    (x1, max(20, y1 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55,
                                    (255, 255, 255),
                                    2,
                                )

                            self.history.append(recognized_names)
                            self._authorize_votes(now)
                    else:
                        for face in faces:
                            x1, y1, x2, y2 = fe.bbox(face)
                            cv2.rectangle(view, (x1, y1), (x2, y2), (30, 144, 255), 2)

                    expired = [name for name, until in self.authorized_until_by_name.items() if now > until]
                    for name in expired:
                        self.authorized_until_by_name.pop(name, None)

                    authorized = set(self.authorized_until_by_name.keys())
                    controller = self._controller_name(authorized, visible_area_by_name)
                    controller_face_bbox = visible_face_box_by_name.get(controller) if controller else None
                    if controller and controller_face_bbox is not None:
                        self.last_controller_name = controller
                        self.last_controller_face_bbox = controller_face_bbox
                        self.last_controller_time = now
                    elif (
                        self.last_controller_name is not None
                        and self.last_controller_face_bbox is not None
                        and (now - self.last_controller_time) <= self.controller_hold_sec
                    ):
                        controller = self.last_controller_name
                        controller_face_bbox = self.last_controller_face_bbox
                    else:
                        self.last_controller_name = None
                        self.last_controller_face_bbox = None
                        self.last_controller_time = 0.0

                    gesture = None
                    gesture_display = "None"
                    if self.recognition_on and controller and controller_face_bbox and self.gesture_enabled_var.get():
                        if self.hands_engine is None:
                            self._ensure_hands_engine()

                        if self.hands_engine is not None and (self.frame_counter % self.gesture_detect_interval) == 0:
                            self.last_raw_gesture = self._detect_gesture(
                                proc,
                                view=view,
                                controller_face_bbox=controller_face_bbox,
                            )

                        raw_gesture = self.last_raw_gesture
                        gesture = self._stabilize_detected_gesture(raw_gesture)
                        if gesture is None:
                            gesture_display = "UNKNOWN"
                        else:
                            gesture_display = gesture
                        self._apply_gesture_command(now, gesture, controller)
                        self._draw_cached_hand_overlay(view, now)
                    else:
                        self.gesture_history.append(None)
                        self.detected_gesture_history.clear()
                        self.last_raw_gesture = None
                        self.last_hand_points = None
                        self.last_hand_points_time = 0.0
                        self.last_controller_name = None
                        self.last_controller_face_bbox = None
                        self.last_controller_time = 0.0
                        if self.recognition_on and self.gesture_enabled_var.get():
                            if not controller or not controller_face_bbox:
                                gesture_display = "WAITING FOR AUTHORIZED FACE"
                            else:
                                gesture_display = "NONE"

                    if self.recognition_on and self.gesture_enabled_var.get():
                        self._draw_gesture_hud(view, gesture_display)

                    self.detect_var.set(
                        f"Faces: {len(faces)} | Recognized now: {', '.join(sorted(recognized_names)) if recognized_names else 'None'}"
                    )
                    self.auth_var.set(f"Authorized: {', '.join(sorted(authorized)) if authorized else 'None'}")
                    if self.recognition_on and self.gesture_enabled_var.get():
                        if gesture_display == "UNKNOWN":
                            self.gesture_var.set("Gesture: UNKNOWN (not defined / unclear angle)")
                        else:
                            self.gesture_var.set(f"Gesture: {gesture_display}")
                    elif gesture == "UNKNOWN":
                        self.gesture_var.set("Gesture: UNKNOWN (not defined / unclear angle)")
                    else:
                        self.gesture_var.set(f"Gesture: {gesture if gesture else 'None'}")

                    if self.recognition_on:
                        status = "RECOGNIZING"
                        if authorized:
                            status = "AUTHORIZED: " + ", ".join(sorted(authorized)[:3])
                    else:
                        status = "CAMERA ON (recognition paused)"
                    self.status_var.set(status)

                    rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.video_label.imgtk = imgtk
                    self.video_label.configure(image=imgtk)
        except Exception:
            # Keep UI loop alive even if a frame/gesture operation fails.
            pass
        finally:
            self.root.after(20, self._update_loop)

    def _on_close(self) -> None:
        self.stop_camera()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    FaceControlCenter(root)
    root.mainloop()


if __name__ == "__main__":
    main()
