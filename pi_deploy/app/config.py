from dataclasses import dataclass, field
from pathlib import Path
import os

PKG_DIR = Path(__file__).resolve().parent
DEPLOY_DIR = PKG_DIR.parent
REPO_ROOT = DEPLOY_DIR.parent

@dataclass
class Settings:
    host: str = "0.0.0.0"
    port: int = 8080

    # Pi hardware UART to ESP32 GPIO 16/17 (Serial2). On Pi 5 this is /dev/serial0
    # after enabling hardware serial in raspi-config. Falls back to mock if missing.
    serial_device: str = os.getenv("REVO_SERIAL", "/dev/serial0")
    serial_baud: int = 115200
    serial_timeout_s: float = 1.0
    force_mock_uart: bool = os.getenv("REVO_MOCK_UART", "0") == "1"

    # Where models and face_db live (reuse existing repo layout)
    models_dir: Path = field(default_factory=lambda: REPO_ROOT / "models")
    known_faces_dir: Path = field(default_factory=lambda: REPO_ROOT / "data" / "known_faces")
    face_db_file: Path = field(default_factory=lambda: REPO_ROOT / "data" / "face_db.npz")

    # Recognition tuning (matches src/face_embedding.py defaults)
    recog_threshold: float = 0.42
    recog_margin: float = 0.06
    recog_centroid_threshold: float = 0.40
    vote_history: int = 6
    vote_required: int = 4

    # Gesture stability (seconds a sign must be held)
    gesture_hold_s: float = 0.5
    gesture_cooldown_s: float = 0.8  # min gap between sending two commands

    # Camera
    pi_camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    target_fps: int = 15

    # Modes: "pi_camera" | "laptop_client" | "laptop_stream"
    default_mode: str = "pi_camera"

    web_dir: Path = field(default_factory=lambda: DEPLOY_DIR / "web")


settings = Settings()
