#!/usr/bin/env python3
"""
Laptop-side AI runner (mode "laptop_client").

Runs on your laptop when the Pi is in laptop_client mode. Grabs the laptop's
webcam, does face + gesture locally, and POSTs recognised gestures to the Pi,
which forwards them to the ESP32 over UART.

Usage:
    python laptop_client.py --pi-url http://<pi-ip>:8080

Requires the same requirements.txt as the Pi app.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import requests

# Reuse the pi_deploy AI code. This script must be run from repo root so
# `app.*` imports resolve.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pi_deploy.app.ai.face import FaceRecognizer
from pi_deploy.app.ai.gesture import GestureClassifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pi-url", required=True, help="e.g. http://192.168.1.42:8080")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--preview", action="store_true", help="Show OpenCV preview window")
    args = ap.parse_args()

    pi = args.pi_url.rstrip("/")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera index {args.camera}")

    face = FaceRecognizer()
    gest = GestureClassifier()

    # Tell the Pi we're driving it
    try:
        requests.post(f"{pi}/api/mode", json={"mode": "laptop_client"}, timeout=2)
    except Exception as e:
        print(f"[warn] could not set mode on Pi: {e}")

    print(f"[laptop_client] streaming gestures → {pi}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02); continue

            name, _box = face.process(frame)
            g = gest.process(frame) if name else None

            if g and name:
                try:
                    r = requests.post(f"{pi}/api/gesture",
                                      json={"gesture": g, "person": name}, timeout=2)
                    print(f"  {name} → {g} → {r.json().get('sent')}")
                except Exception as e:
                    print(f"[err] POST /api/gesture: {e}")

            if args.preview:
                label = f"{name or 'no face'} | {g or ''}"
                cv2.putText(frame, label, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("laptop_client", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        cap.release()
        if args.preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
