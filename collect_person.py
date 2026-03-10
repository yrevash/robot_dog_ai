#!/usr/bin/env python3
"""
collect_person.py
=================
Single command to enroll a new person (face + gestures) into REVO.

Usage
-----
    python collect_person.py --name Alice
    python collect_person.py --name Alice --face-samples 25 --gesture-samples 30
    python collect_person.py --name Alice --camera 1   # if webcam index != 0
    python collect_person.py --name Alice --gestures-only   # skip face step
    python collect_person.py --name Alice --face-only       # skip gesture step

Gestures collected (5 key commands):
    WALK   – 4 fingers up, thumb folded → walk forward
    SIT    – V sign (index + middle up) → sit
    STOP   – Fist, all fingers closed   → stop / stay
    STAND  – 3 fingers (index+middle+ring up) → stand up
    BARK   – Pinch thumb+index, 3 fingers up → bark

After collecting from ALL people, run:
    python src/face_embedding.py build          # rebuild face DB
    python experiments/run_all.py               # re-run all evaluations
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ── 5 gestures we care about ──────────────────────────────────────────────────
GESTURES = ["WALK", "SIT", "STOP", "STAND", "BARK"]


def run(cmd: list[str]) -> int:
    """Run a subprocess command, streaming output to the terminal."""
    print(f"\n>>> {' '.join(cmd)}\n", flush=True)
    result = subprocess.run(cmd)
    return result.returncode


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Enroll a person into REVO (face + gesture data collection).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--name", "-n", required=True,
                   help="Person's name (used as folder name, e.g. Alice).")
    p.add_argument("--face-samples", type=int, default=25,
                   help="Number of face photos to capture (default: 25).")
    p.add_argument("--gesture-samples", type=int, default=30,
                   help="Number of gesture frames per gesture class (default: 30).")
    p.add_argument("--camera", "-c", type=int, default=0,
                   help="Camera index (default: 0).")
    p.add_argument("--face-only", action="store_true",
                   help="Collect face images only, skip gestures.")
    p.add_argument("--gestures-only", action="store_true",
                   help="Collect gestures only, skip face enrollment.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    name = args.name.strip()
    if not name:
        print("ERROR: --name cannot be empty.", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print(f"  REVO Data Collection — Subject: {name}")
    print("=" * 60)
    print(f"  Gestures : {', '.join(GESTURES)}")
    print(f"  Face samples    : {args.face_samples}")
    print(f"  Gesture samples : {args.gesture_samples} per gesture")
    print(f"  Camera index    : {args.camera}")
    print("=" * 60)

    # ── Step 1: Face enrollment ───────────────────────────────────────────────
    if not args.gestures_only:
        print("\n[STEP 1/2] Face Capture")
        print("  Hold still, look at the camera.")
        print("  The system will capture your face automatically.\n")
        rc = run([
            sys.executable,
            str(ROOT / "src" / "face_embedding.py"),
            "capture",
            "--name", name,
            "--samples", str(args.face_samples),
            "--camera", str(args.camera),
        ])
        if rc != 0:
            print(f"\nERROR: Face capture failed (exit code {rc}).", file=sys.stderr)
            sys.exit(rc)
        print(f"\n✓ Face images saved to known_faces/{name}/")
    else:
        print("\n[STEP 1/2] Face capture — SKIPPED (--gestures-only)")

    # ── Step 2: Gesture collection ────────────────────────────────────────────
    if not args.face_only:
        print("\n[STEP 2/2] Gesture Collection")
        print(f"  You will perform {len(GESTURES)} gestures.")
        print("  For each gesture:")
        print("    - Read the instruction on screen")
        print("    - Press SPACE to start a 3-second countdown")
        print("    - Hold the gesture steady while recording")
        print("    - Press S to skip a gesture, Q to quit early\n")
        rc = run([
            sys.executable,
            str(ROOT / "experiments" / "collect_gesture_dataset.py"),
            "--name", name,
            "--samples-per-gesture", str(args.gesture_samples),
            "--gestures", *GESTURES,
            "--camera", str(args.camera),
        ])
        if rc != 0:
            print(f"\nERROR: Gesture collection failed (exit code {rc}).", file=sys.stderr)
            sys.exit(rc)
        print(f"\n✓ Gesture data saved to data/gesture_dataset/{name}/")
    else:
        print("\n[STEP 2/2] Gesture collection — SKIPPED (--face-only)")

    # ── Done ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Collection complete for: {name}")
    print("=" * 60)
    print("\nNEXT STEPS (run these after collecting ALL people):")
    print("  1. Rebuild face DB:")
    print("       python src/face_embedding.py build")
    print("  2. Re-run experiments:")
    print("       python experiments/run_all.py")
    print()


if __name__ == "__main__":
    main()
