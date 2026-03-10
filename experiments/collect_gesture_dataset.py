#!/usr/bin/env python3
"""
experiments/collect_gesture_dataset.py
=======================================
Phase 1 helper — guided interactive tool to collect a labeled gesture dataset
from a webcam.

For each of 10 gesture classes the operator is shown:
  - The gesture name (large overlay)
  - A hand-sign description
  - Keyboard hints (SPACE = start, S = skip, Q = quit)

Once SPACE is pressed a 3-2-1 countdown begins, then the tool auto-captures
--samples-per-gesture frames while the subject holds the pose.  Frames are
saved under::

    gesture_dataset/<subject_name>/<GESTURE_CLASS>/<NNN>.jpg

After every session a CSV is appended to::

    gesture_dataset/ground_truth.csv

with columns: image_path, subject, gesture_label

Usage
-----
    python experiments/collect_gesture_dataset.py --name Alice
    python experiments/collect_gesture_dataset.py --name Bob --samples-per-gesture 30
    python experiments/collect_gesture_dataset.py --name Alice --gestures FORWARD SIT STOP
    python experiments/collect_gesture_dataset.py --name Alice --camera 1
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

# ── Bootstrap project path so utils.py is importable ─────────────────────────
_EXPERIMENTS = Path(__file__).resolve().parent
_PROJECT_ROOT = _EXPERIMENTS.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

from utils import GESTURE_DATA, PROJECT_ROOT, setup_logging  # noqa: E402

import cv2  # noqa: E402

# ── Gesture definitions ───────────────────────────────────────────────────────
# Order matters: it is the default capture order shown to the operator.
ALL_GESTURES: list[tuple[str, str]] = [
    ("FORWARD",  "Open palm — all 5 fingers up"),
    ("BACKWARD", "Thumb down — fist closed"),
    ("LEFT",     "Index up — lean left"),
    ("RIGHT",    "Index up — lean right"),
    ("SIT",      "V sign — index + middle up"),
    ("STAND",    "3 fingers — index + middle + ring"),
    ("WALK",     "4 fingers up — thumb folded"),
    ("TAIL_WAG", "Only pinky up"),
    ("STOP",     "Fist — all fingers closed"),
    ("BARK",     "Pinch thumb+index + 3 fingers up"),
]

GROUND_TRUTH_CSV = GESTURE_DATA / "ground_truth.csv"
CSV_FIELDNAMES   = ["image_path", "subject", "gesture_label"]

# ── Drawing constants ─────────────────────────────────────────────────────────
_FONT       = cv2.FONT_HERSHEY_DUPLEX
_FONT_BOLD  = cv2.FONT_HERSHEY_SIMPLEX
_GREEN      = (50, 220, 50)
_RED        = (30, 30, 220)
_YELLOW     = (30, 220, 220)
_WHITE      = (255, 255, 255)
_BLACK      = (0, 0, 0)
_DARK_GRAY  = (30, 30, 30)
_BORDER_W   = 6   # px


# ── Helpers ───────────────────────────────────────────────────────────────────

def _text_with_shadow(
    frame,
    text: str,
    origin: tuple[int, int],
    font,
    scale: float,
    color: tuple,
    thickness: int = 2,
) -> None:
    """Draw text with a dark shadow for readability over any background."""
    ox, oy = origin
    cv2.putText(frame, text, (ox + 2, oy + 2), font, scale, _BLACK, thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, origin, font, scale, color, thickness, cv2.LINE_AA)


def _draw_progress_bar(
    frame,
    captured: int,
    total: int,
    bar_h: int = 22,
    margin: int = 16,
) -> None:
    """Green progress bar at the bottom of the frame."""
    h, w = frame.shape[:2]
    y0 = h - bar_h - margin
    y1 = h - margin
    x0 = margin
    x1 = w - margin

    # Background track
    cv2.rectangle(frame, (x0, y0), (x1, y1), _DARK_GRAY, -1)

    # Filled portion
    if total > 0:
        fill_x = int(x0 + (x1 - x0) * (captured / total))
        cv2.rectangle(frame, (x0, y0), (fill_x, y1), _GREEN, -1)

    # Border
    cv2.rectangle(frame, (x0, y0), (x1, y1), _WHITE, 1)

    # Label centred in bar
    label = f"Capturing: {captured}/{total}"
    (tw, th), _ = cv2.getTextSize(label, _FONT, 0.55, 1)
    tx = x0 + ((x1 - x0) - tw) // 2
    ty = y0 + (bar_h + th) // 2
    cv2.putText(frame, label, (tx, ty), _FONT, 0.55, _WHITE, 1, cv2.LINE_AA)


def _draw_idle_overlay(
    frame,
    gesture_name: str,
    description: str,
    gesture_index: int,
    gesture_total: int,
    hint: str = "SPACE = start   S = skip   Q = quit",
) -> None:
    """Overlay shown while waiting for the operator to press SPACE."""
    h, w = frame.shape[:2]

    # Semi-transparent top banner
    banner_h = 120
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), _DARK_GRAY, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Gesture counter  "3 / 10"
    counter_text = f"Gesture {gesture_index} / {gesture_total}"
    _text_with_shadow(frame, counter_text, (16, 26), _FONT, 0.6, _YELLOW, 1)

    # Large gesture name
    (tw, _), _ = cv2.getTextSize(gesture_name, _FONT_BOLD, 1.6, 3)
    gx = max(16, (w - tw) // 2)
    _text_with_shadow(frame, gesture_name, (gx, 72), _FONT_BOLD, 1.6, _WHITE, 3)

    # Description
    (tw2, _), _ = cv2.getTextSize(description, _FONT, 0.75, 1)
    dx = max(16, (w - tw2) // 2)
    _text_with_shadow(frame, description, (dx, 108), _FONT, 0.75, _GREEN, 1)

    # Bottom hint bar
    hint_h = 36
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - hint_h), (w, h), _DARK_GRAY, -1)
    cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)
    (tw3, _), _ = cv2.getTextSize(hint, _FONT, 0.55, 1)
    hx = max(16, (w - tw3) // 2)
    cv2.putText(frame, hint, (hx, h - 10), _FONT, 0.55, _YELLOW, 1, cv2.LINE_AA)


def _draw_capture_overlay(
    frame,
    gesture_name: str,
    description: str,
    captured: int,
    total: int,
) -> None:
    """Overlay shown while actively capturing frames."""
    h, w = frame.shape[:2]

    # Red border to signal active recording
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), _RED, _BORDER_W)

    # Top banner
    banner_h = 80
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), _DARK_GRAY, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Gesture name
    (tw, _), _ = cv2.getTextSize(gesture_name, _FONT_BOLD, 1.4, 2)
    gx = max(16, (w - tw) // 2)
    _text_with_shadow(frame, gesture_name, (gx, 52), _FONT_BOLD, 1.4, _WHITE, 2)

    # Description
    (tw2, _), _ = cv2.getTextSize(description, _FONT, 0.65, 1)
    dx = max(16, (w - tw2) // 2)
    _text_with_shadow(frame, description, (dx, 76), _FONT, 0.65, _GREEN, 1)

    # REC dot + label top-right
    rec_x = w - 110
    cv2.circle(frame, (rec_x, 20), 9, _RED, -1)
    cv2.putText(frame, "REC", (rec_x + 16, 26), _FONT, 0.6, _RED, 1, cv2.LINE_AA)

    # Progress bar at bottom
    _draw_progress_bar(frame, captured, total)


def _draw_countdown(frame, count: int) -> None:
    """Full-frame countdown overlay (3, 2, 1, GO!)."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), _DARK_GRAY, -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    label = str(count) if count > 0 else "GO!"
    color = _YELLOW if count > 0 else _GREEN
    scale = 5.0 if count > 0 else 3.5
    thick = 10 if count > 0 else 6
    (tw, th), _ = cv2.getTextSize(label, _FONT_BOLD, scale, thick)
    cx = (w - tw) // 2
    cy = (h + th) // 2
    _text_with_shadow(frame, label, (cx, cy), _FONT_BOLD, scale, color, thick)


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _ensure_csv_header() -> None:
    """Write the CSV header if the file is new."""
    if not GROUND_TRUTH_CSV.exists():
        GROUND_TRUTH_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(GROUND_TRUTH_CSV, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDNAMES).writeheader()


def _append_rows(rows: list[dict]) -> None:
    with open(GROUND_TRUTH_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        w.writerows(rows)


# ── Core collection loop ──────────────────────────────────────────────────────

def run_collection(
    cap: cv2.VideoCapture,
    subject: str,
    gestures: list[tuple[str, str]],
    samples_per_gesture: int,
    log,
) -> dict[str, int]:
    """
    Interactive collection loop.

    Returns a dict mapping gesture_name -> number of samples captured.
    """
    summary: dict[str, int] = {}
    new_csv_rows: list[dict] = []
    gesture_total = len(gestures)

    for g_idx, (gesture_name, description) in enumerate(gestures, start=1):
        out_dir = GESTURE_DATA / subject / gesture_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Count existing samples so we can continue numbering correctly.
        # Use max existing stem number (not count) to avoid collisions when
        # files have been deleted and the sequence has gaps.
        existing = sorted(out_dir.glob("*.jpg"))
        if existing:
            try:
                next_num = max(int(p.stem) for p in existing) + 1
            except ValueError:
                next_num = len(existing) + 1
        else:
            next_num = 1

        log.info(f"[{g_idx}/{gesture_total}] Gesture: {gesture_name} — {description}")
        log.info(f"  Output dir: {out_dir}  (existing: {len(existing)})")

        # ── Idle state: wait for SPACE / S / Q ────────────────────────────────
        waiting = True
        while waiting:
            ret, frame = cap.read()
            if not ret:
                log.error("Camera read failed.")
                break

            display = frame.copy()
            _draw_idle_overlay(display, gesture_name, description, g_idx, gesture_total)
            cv2.imshow("REVO Gesture Collection", display)

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q") or key == 27:   # Q or ESC
                log.info("User quit.")
                _append_rows(new_csv_rows)
                return summary
            elif key == ord("s"):
                log.info(f"  Skipped: {gesture_name}")
                summary[gesture_name] = 0
                waiting = False
                break
            elif key == ord(" "):
                waiting = False
                # ── Countdown 3-2-1-GO ────────────────────────────────────────
                for count in (3, 2, 1, 0):
                    t_start = time.monotonic()
                    while time.monotonic() - t_start < 1.0:
                        ret2, f2 = cap.read()
                        if not ret2:
                            break
                        cd = f2.copy()
                        _draw_countdown(cd, count)
                        cv2.imshow("REVO Gesture Collection", cd)
                        cv2.waitKey(30)

        if summary.get(gesture_name) == 0:
            continue  # was skipped

        # ── Capture state ─────────────────────────────────────────────────────
        captured = 0
        log.info(f"  Capturing {samples_per_gesture} samples …")

        while captured < samples_per_gesture:
            ret, frame = cap.read()
            if not ret:
                log.warning("  Frame grab failed, retrying …")
                time.sleep(0.05)
                continue

            # Save frame
            img_num   = next_num + captured
            img_fname = f"{img_num:03d}.jpg"
            img_path  = out_dir / img_fname
            cv2.imwrite(str(img_path), frame)

            # Record for CSV
            new_csv_rows.append({
                "image_path":    str(img_path.relative_to(PROJECT_ROOT)),
                "subject":       subject,
                "gesture_label": gesture_name,
            })
            captured += 1

            # Draw capture overlay and show live
            display = frame.copy()
            _draw_capture_overlay(display, gesture_name, description, captured, samples_per_gesture)
            cv2.imshow("REVO Gesture Collection", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                log.info("  User quit during capture.")
                summary[gesture_name] = captured
                _append_rows(new_csv_rows)
                return summary

        summary[gesture_name] = captured
        log.info(f"  Done: {captured} samples saved to {out_dir}")

        # Brief pause before next gesture
        pause_end = time.monotonic() + 1.0
        while time.monotonic() < pause_end:
            ret, frame = cap.read()
            if ret:
                cv2.imshow("REVO Gesture Collection", frame)
            cv2.waitKey(30)

    # Flush CSV rows
    _append_rows(new_csv_rows)
    return summary


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect labeled gesture dataset for REVO gesture experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--name", "-n",
        default="",
        help="Subject name (used as sub-directory name). Prompted if omitted.",
    )
    parser.add_argument(
        "--samples-per-gesture", "-s",
        type=int,
        default=50,
        metavar="N",
        help="Number of frames to capture per gesture class (default: 50).",
    )
    parser.add_argument(
        "--gestures", "-g",
        nargs="+",
        metavar="GESTURE",
        default=None,
        help=(
            "Subset of gestures to collect. "
            "Choices: " + ", ".join(g for g, _ in ALL_GESTURES) + ". "
            "Default: all 10."
        ),
    )
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=0,
        metavar="INDEX",
        help="Camera device index (default: 0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log  = setup_logging("phase1_gesture_collection")

    # ── Resolve subject name ──────────────────────────────────────────────────
    subject = args.name.strip()
    if not subject:
        subject = input("Enter subject name: ").strip()
    if not subject:
        log.error("Subject name is required.")
        sys.exit(1)
    log.info(f"Subject: {subject}")

    # ── Resolve gesture list ──────────────────────────────────────────────────
    all_names = [g for g, _ in ALL_GESTURES]
    if args.gestures:
        bad = [g for g in args.gestures if g not in all_names]
        if bad:
            log.error(f"Unknown gesture(s): {bad}. Valid names: {all_names}")
            sys.exit(1)
        # Preserve canonical order
        order = {g: i for i, g in enumerate(all_names)}
        selected_names = sorted(set(args.gestures), key=lambda g: order[g])
        gestures = [(g, d) for g, d in ALL_GESTURES if g in selected_names]
    else:
        gestures = list(ALL_GESTURES)

    log.info(f"Gestures to collect ({len(gestures)}): {[g for g, _ in gestures]}")
    log.info(f"Samples per gesture: {args.samples_per_gesture}")
    log.info(f"Output root: {GESTURE_DATA / subject}")

    # ── Open camera ───────────────────────────────────────────────────────────
    log.info(f"Opening camera index {args.camera} …")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        log.error(f"Cannot open camera {args.camera}.")
        sys.exit(1)

    # Prefer a reasonable resolution; camera may ignore this.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info(f"Camera resolution: {actual_w}×{actual_h}")

    # ── Ensure CSV header exists ───────────────────────────────────────────────
    _ensure_csv_header()

    # ── Run collection ────────────────────────────────────────────────────────
    try:
        summary = run_collection(cap, subject, gestures, args.samples_per_gesture, log)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    # ── Print summary ─────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 50)
    log.info("COLLECTION SUMMARY")
    log.info("=" * 50)
    log.info(f"{'Gesture':<16}  {'Captured':>8}")
    log.info("-" * 30)
    total_captured = 0
    for gesture_name, _ in gestures:
        n = summary.get(gesture_name, 0)
        total_captured += n
        status = str(n) if n > 0 else "skipped"
        log.info(f"  {gesture_name:<14}  {status:>8}")
    log.info("-" * 30)
    log.info(f"  {'TOTAL':<14}  {total_captured:>8}")
    log.info("=" * 50)
    log.info(f"ground_truth.csv updated: {GROUND_TRUTH_CSV}")
    log.info("Done.")


if __name__ == "__main__":
    main()
