#!/usr/bin/env python3
"""
experiments/bench_power.py
==========================
Phase 8 — Power Management Benchmark

Measures CPU usage, throughput, wake-up latency, thread safety, and
projected power savings across the three power states
(ACTIVE, POWER_SAVE, POWER_OFF).

Experiments
-----------
E1  Per-state resource usage
    - CPU % (via psutil)
    - Frames processed per second
    - Inference calls per second

E2  Wake-up latency
    - POWER_SAVE -> ACTIVE  (face-triggered wake)
    - POWER_OFF  -> ACTIVE  (button/signal wake)

E3  State transition timeline
    - Simulate a realistic usage session (bursts of activity, idle periods)
    - Log every transition with timestamp

E4  Idle timer accuracy
    - Verify transitions happen at the correct times (tolerance < 100ms)

E5  Thread safety stress test
    - Concurrent tick() / report_activity() / wake() from multiple threads
    - Verify no crashes, no invalid states, state consistency

E6  Power savings projection
    - Model cumulative CPU savings over 8-hour session
    - Compare always-on vs power-managed for varying activity ratios

Usage
-----
    python experiments/bench_power.py
    python experiments/bench_power.py --idle-save 3 --idle-off 6   # fast test
"""

from __future__ import annotations

import argparse
import csv
import sys
import threading
import time
from pathlib import Path

_EXPERIMENTS = Path(__file__).resolve().parent
_PROJECT_ROOT = _EXPERIMENTS.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

from utils import RESULTS_DIR, setup_logging, apply_paper_style  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(_PROJECT_ROOT / "src"))
from power_state import PowerManager, PowerState  # noqa: E402
import face_embedding as fe  # noqa: E402

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

PHASE_DIR = RESULTS_DIR / "phase8"


def _ensure_dirs() -> None:
    PHASE_DIR.mkdir(parents=True, exist_ok=True)


def _create_synthetic_frame(w: int = 640, h: int = 480) -> np.ndarray:
    """Create a random synthetic frame for benchmarking."""
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# E1: Per-state resource usage
# ══════════════════════════════════════════════════════════════════════════════

def experiment_resource_usage(log, n_frames: int = 200) -> list[dict]:
    """Measure CPU time and throughput in each power state."""
    log.info("=" * 60)
    log.info("E1: Per-state resource usage (n_frames=%d)", n_frames)
    log.info("=" * 60)

    # Load models
    fe.KNOWN_FACES_DIR = _PROJECT_ROOT / "data" / "known_faces"
    fe.MODELS_DIR = _PROJECT_ROOT / "models"
    fe.DB_FILE = _PROJECT_ROOT / "data" / "face_db.npz"
    fe.check_opencv_requirements()
    fe.ensure_models(auto_download=True)

    detector = fe.create_detector((640, 480), score_threshold=0.9)
    recognizer = fe.create_recognizer()

    try:
        db_embs, db_names, centroids, centroid_names = fe.load_db(fe.DB_FILE)
        db_loaded = True
    except Exception:
        db_loaded = False
        db_embs = db_names = centroids = centroid_names = None

    # Try to init mediapipe
    mp_available = False
    hand_landmarker = None
    mp_lib = None
    try:
        import mediapipe as _mp_lib
        mp_lib = _mp_lib
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
        model_path = _PROJECT_ROOT / "models" / "hand_landmarker.task"
        if model_path.exists():
            base_opts = mp_tasks.BaseOptions(model_asset_path=str(model_path))
            opts = mp_vision.HandLandmarkerOptions(
                base_options=base_opts,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=1,
            )
            hand_landmarker = mp_vision.HandLandmarker.create_from_options(opts)
            mp_available = True
    except Exception:
        pass

    results = []

    for state_name, do_detect, do_embed, do_gesture in [
        ("ACTIVE", True, True, True),
        ("POWER_SAVE", True, False, False),  # face detect only, every 10th frame
        ("POWER_OFF", False, False, False),   # no processing
    ]:
        log.info("\n--- State: %s ---", state_name)

        frame_skip = 10 if state_name == "POWER_SAVE" else 1
        frames_processed = 0
        detect_calls = 0
        embed_calls = 0
        gesture_calls = 0

        process = psutil.Process() if _HAS_PSUTIL else None
        cpu_start = process.cpu_times() if process else None
        wall_start = time.monotonic()

        for i in range(n_frames):
            frame = _create_synthetic_frame()

            if state_name == "POWER_OFF":
                time.sleep(0.001)
                continue

            if (i % frame_skip) != 0:
                continue

            frames_processed += 1

            if do_detect:
                faces = fe.detect_faces(detector, frame)
                detect_calls += 1

                if do_embed and db_loaded and faces:
                    for face in faces[:1]:
                        try:
                            emb = fe.embedding_from_face(frame, face, recognizer)
                            if emb is not None:
                                fe.match_identity(
                                    emb, db_embs, db_names,
                                    centroids, centroid_names,
                                )
                            embed_calls += 1
                        except Exception:
                            pass

                if do_gesture and mp_available and hand_landmarker is not None:
                    try:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_img = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=rgb)
                        hand_landmarker.detect(mp_img)
                        gesture_calls += 1
                    except Exception:
                        pass

        wall_elapsed = time.monotonic() - wall_start

        cpu_pct = 0.0
        if process and cpu_start:
            cpu_end = process.cpu_times()
            cpu_used = (cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)
            cpu_pct = (cpu_used / wall_elapsed) * 100.0 if wall_elapsed > 0 else 0.0

        fps = frames_processed / wall_elapsed if wall_elapsed > 0 else 0.0

        row = {
            "state": state_name,
            "n_input_frames": n_frames,
            "frames_processed": frames_processed,
            "detect_calls": detect_calls,
            "embed_calls": embed_calls,
            "gesture_calls": gesture_calls,
            "wall_time_s": round(wall_elapsed, 4),
            "fps": round(fps, 2),
            "cpu_percent": round(cpu_pct, 1),
            "frame_skip": frame_skip,
        }
        results.append(row)
        log.info("  Processed: %d/%d frames (skip=%d)", frames_processed, n_frames, frame_skip)
        log.info("  FPS: %.1f | CPU: %.1f%% | Wall: %.2fs", fps, cpu_pct, wall_elapsed)
        log.info("  Detect: %d | Embed: %d | Gesture: %d", detect_calls, embed_calls, gesture_calls)

    if hand_landmarker:
        hand_landmarker.close()

    return results


# ══════════════════════════════════════════════════════════════════════════════
# E2: Wake-up latency
# ══════════════════════════════════════════════════════════════════════════════

def experiment_wake_latency(log, n_trials: int = 10) -> list[dict]:
    """Measure time from wake trigger to state transition completion."""
    log.info("\n" + "=" * 60)
    log.info("E2: Wake-up latency (n_trials=%d)", n_trials)
    log.info("=" * 60)

    results = []

    for trial in range(n_trials):
        # Test 1: POWER_SAVE -> ACTIVE (via report_activity)
        transitions = []
        pm = PowerManager(
            idle_to_save_sec=0.01,
            save_to_off_sec=10.0,
            on_enter_active=lambda: transitions.append(("ACTIVE", time.monotonic())),
            on_enter_power_save=lambda: transitions.append(("POWER_SAVE", time.monotonic())),
            on_enter_power_off=lambda: transitions.append(("POWER_OFF", time.monotonic())),
        )

        # Enter POWER_SAVE
        time.sleep(0.02)
        pm.tick()
        assert pm.state == PowerState.POWER_SAVE

        # Measure wake latency
        t_wake_start = time.monotonic()
        pm.report_activity()
        t_wake_end = time.monotonic()
        wake_save_ms = (t_wake_end - t_wake_start) * 1000.0

        results.append({
            "trial": trial + 1,
            "transition": "POWER_SAVE_to_ACTIVE",
            "method": "report_activity",
            "latency_ms": round(wake_save_ms, 4),
        })

        # Test 2: POWER_OFF -> ACTIVE (via wake())
        transitions.clear()
        pm2 = PowerManager(
            idle_to_save_sec=0.01,
            save_to_off_sec=0.02,
            on_enter_active=lambda: transitions.append(("ACTIVE", time.monotonic())),
            on_enter_power_save=lambda: transitions.append(("POWER_SAVE", time.monotonic())),
            on_enter_power_off=lambda: transitions.append(("POWER_OFF", time.monotonic())),
        )
        time.sleep(0.03)
        pm2.tick()
        pm2.tick()  # ensure POWER_OFF
        assert pm2.state == PowerState.POWER_OFF

        t_wake_start = time.monotonic()
        pm2.wake()
        t_wake_end = time.monotonic()
        wake_off_ms = (t_wake_end - t_wake_start) * 1000.0

        results.append({
            "trial": trial + 1,
            "transition": "POWER_OFF_to_ACTIVE",
            "method": "wake()",
            "latency_ms": round(wake_off_ms, 4),
        })

    save_latencies = [r["latency_ms"] for r in results if r["transition"] == "POWER_SAVE_to_ACTIVE"]
    off_latencies = [r["latency_ms"] for r in results if r["transition"] == "POWER_OFF_to_ACTIVE"]

    log.info("\n  POWER_SAVE -> ACTIVE: mean=%.4f ms, max=%.4f ms",
             np.mean(save_latencies), np.max(save_latencies))
    log.info("  POWER_OFF  -> ACTIVE: mean=%.4f ms, max=%.4f ms",
             np.mean(off_latencies), np.max(off_latencies))

    return results


# ══════════════════════════════════════════════════════════════════════════════
# E3: State transition timeline
# ══════════════════════════════════════════════════════════════════════════════

def experiment_transition_timeline(log, idle_save: float = 5.0, idle_off: float = 10.0) -> list[dict]:
    """Simulate a realistic usage session with activity bursts and idle periods."""
    log.info("\n" + "=" * 60)
    log.info("E3: State transition timeline (save=%ds, off=%ds)", int(idle_save), int(idle_off))
    log.info("=" * 60)

    transitions = []
    t0 = time.monotonic()

    def record(state_name: str) -> None:
        transitions.append({
            "time_s": round(time.monotonic() - t0, 3),
            "state": state_name,
        })

    pm = PowerManager(
        idle_to_save_sec=idle_save,
        save_to_off_sec=idle_off,
        on_enter_active=lambda: record("ACTIVE"),
        on_enter_power_save=lambda: record("POWER_SAVE"),
        on_enter_power_off=lambda: record("POWER_OFF"),
    )

    # Record initial state
    transitions.append({"time_s": 0.0, "state": "ACTIVE"})

    # Phase 1: Active usage for 2 seconds (activity every 0.5s)
    log.info("  Phase 1: Active usage (2s with activity)")
    end = time.monotonic() + 2.0
    while time.monotonic() < end:
        pm.report_activity()
        pm.tick()
        time.sleep(0.5)

    # Phase 2: Go idle, wait for POWER_SAVE
    log.info("  Phase 2: Idle, waiting for POWER_SAVE (%.1fs)", idle_save)
    end = time.monotonic() + idle_save + 1.0
    while time.monotonic() < end:
        pm.tick()
        time.sleep(0.5)
        if pm.state == PowerState.POWER_SAVE:
            break

    # Phase 3: Brief wake via activity, then idle again
    log.info("  Phase 3: Brief wake, then idle again")
    pm.report_activity()
    pm.tick()
    time.sleep(1.0)

    # Phase 4: Wait for POWER_SAVE then POWER_OFF
    log.info("  Phase 4: Waiting for POWER_SAVE -> POWER_OFF")
    end = time.monotonic() + idle_off + 2.0
    while time.monotonic() < end:
        pm.tick()
        time.sleep(0.5)
        if pm.state == PowerState.POWER_OFF:
            break

    # Phase 5: Wake from POWER_OFF
    log.info("  Phase 5: Wake from POWER_OFF")
    time.sleep(0.5)
    pm.wake()
    pm.tick()

    # Phase 6: Brief activity then final idle
    log.info("  Phase 6: Brief activity, then idle to POWER_SAVE")
    pm.report_activity()
    end = time.monotonic() + idle_save + 1.0
    while time.monotonic() < end:
        pm.tick()
        time.sleep(0.5)
        if pm.state == PowerState.POWER_SAVE:
            break

    log.info("  Timeline complete: %d transitions recorded", len(transitions))
    for t in transitions:
        log.info("    t=%.1fs -> %s", t["time_s"], t["state"])

    return transitions


# ══════════════════════════════════════════════════════════════════════════════
# E4: Idle timer accuracy
# ══════════════════════════════════════════════════════════════════════════════

def experiment_timer_accuracy(log, n_trials: int = 10) -> list[dict]:
    """Verify state transitions happen at the correct times."""
    log.info("\n" + "=" * 60)
    log.info("E4: Idle timer accuracy (n_trials=%d)", n_trials)
    log.info("=" * 60)

    target_save = 2.0   # 2 seconds for fast testing
    target_off = 4.0    # 4 seconds total
    tolerance = 0.1     # 100ms tolerance (tighter than before)

    results = []

    for trial in range(n_trials):
        transitions = {}

        def on_save():
            transitions["POWER_SAVE"] = time.monotonic()

        def on_off():
            transitions["POWER_OFF"] = time.monotonic()

        pm = PowerManager(
            idle_to_save_sec=target_save,
            save_to_off_sec=target_off,
            on_enter_power_save=on_save,
            on_enter_power_off=on_off,
        )

        t_start = time.monotonic()

        # Tick rapidly until POWER_OFF
        while pm.state != PowerState.POWER_OFF:
            pm.tick()
            time.sleep(0.01)  # 10ms tick — tighter polling
            if time.monotonic() - t_start > target_off + 3.0:
                break

        save_at = transitions.get("POWER_SAVE")
        off_at = transitions.get("POWER_OFF")

        save_elapsed = (save_at - t_start) if save_at else None
        off_elapsed = (off_at - t_start) if off_at else None

        save_error = abs(save_elapsed - target_save) if save_elapsed else None
        off_error = abs(off_elapsed - target_off) if off_elapsed else None

        save_ok = save_error is not None and save_error < tolerance
        off_ok = off_error is not None and off_error < tolerance

        row = {
            "trial": trial + 1,
            "target_save_s": target_save,
            "actual_save_s": round(save_elapsed, 4) if save_elapsed else None,
            "save_error_ms": round(save_error * 1000, 1) if save_error else None,
            "save_pass": save_ok,
            "target_off_s": target_off,
            "actual_off_s": round(off_elapsed, 4) if off_elapsed else None,
            "off_error_ms": round(off_error * 1000, 1) if off_error else None,
            "off_pass": off_ok,
        }
        results.append(row)
        log.info("  Trial %d: SAVE at %.4fs (err=%.1fms %s) | OFF at %.4fs (err=%.1fms %s)",
                 trial + 1,
                 save_elapsed or 0, (save_error or 0) * 1000, "PASS" if save_ok else "FAIL",
                 off_elapsed or 0, (off_error or 0) * 1000, "PASS" if off_ok else "FAIL")

    all_pass = all(r["save_pass"] and r["off_pass"] for r in results)
    log.info("  Overall: %s (%d/%d trials passed)", "PASS" if all_pass else "FAIL",
             sum(1 for r in results if r["save_pass"] and r["off_pass"]), len(results))

    return results


# ══════════════════════════════════════════════════════════════════════════════
# E5: Thread safety stress test
# ══════════════════════════════════════════════════════════════════════════════

def experiment_thread_safety(log, n_threads: int = 8, ops_per_thread: int = 5000) -> list[dict]:
    """Stress-test PowerManager under concurrent access from multiple threads."""
    log.info("\n" + "=" * 60)
    log.info("E5: Thread safety stress test (%d threads, %d ops each)", n_threads, ops_per_thread)
    log.info("=" * 60)

    VALID_STATES = {PowerState.ACTIVE, PowerState.POWER_SAVE, PowerState.POWER_OFF}
    transition_count = {"to_active": 0, "to_save": 0, "to_off": 0}
    lock = threading.Lock()

    def on_active():
        with lock:
            transition_count["to_active"] += 1

    def on_save():
        with lock:
            transition_count["to_save"] += 1

    def on_off():
        with lock:
            transition_count["to_off"] += 1

    pm = PowerManager(
        idle_to_save_sec=0.001,   # very short for stress testing
        save_to_off_sec=0.002,
        on_enter_active=on_active,
        on_enter_power_save=on_save,
        on_enter_power_off=on_off,
    )

    errors = []
    invalid_states = []

    def worker(thread_id: int) -> None:
        rng = np.random.RandomState(thread_id)
        for i in range(ops_per_thread):
            op = rng.randint(0, 3)
            try:
                if op == 0:
                    pm.tick()
                elif op == 1:
                    pm.report_activity()
                else:
                    pm.wake()

                # Verify state is always valid
                s = pm.state
                if s not in VALID_STATES:
                    invalid_states.append((thread_id, i, str(s)))
            except Exception as exc:
                errors.append((thread_id, i, str(exc)))

    threads = []
    t_start = time.monotonic()
    for tid in range(n_threads):
        t = threading.Thread(target=worker, args=(tid,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
    wall_time = time.monotonic() - t_start

    total_ops = n_threads * ops_per_thread
    final_state = pm.state

    log.info("  Total operations: %d", total_ops)
    log.info("  Wall time: %.3fs (%.0f ops/s)", wall_time, total_ops / wall_time)
    log.info("  Errors: %d", len(errors))
    log.info("  Invalid states: %d", len(invalid_states))
    log.info("  Final state: %s", final_state.name)
    log.info("  Transitions: active=%d, save=%d, off=%d",
             transition_count["to_active"], transition_count["to_save"], transition_count["to_off"])
    log.info("  Result: %s", "PASS" if len(errors) == 0 and len(invalid_states) == 0 else "FAIL")

    if errors:
        for tid, i, exc in errors[:5]:
            log.error("    Thread %d op %d: %s", tid, i, exc)

    results = [{
        "n_threads": n_threads,
        "ops_per_thread": ops_per_thread,
        "total_ops": total_ops,
        "wall_time_s": round(wall_time, 4),
        "ops_per_sec": round(total_ops / wall_time, 0),
        "errors": len(errors),
        "invalid_states": len(invalid_states),
        "transitions_to_active": transition_count["to_active"],
        "transitions_to_save": transition_count["to_save"],
        "transitions_to_off": transition_count["to_off"],
        "final_state": final_state.name,
        "pass": len(errors) == 0 and len(invalid_states) == 0,
    }]

    return results


# ══════════════════════════════════════════════════════════════════════════════
# E6: Power savings projection
# ══════════════════════════════════════════════════════════════════════════════

def experiment_power_projection(log, e1_results: list[dict]) -> list[dict]:
    """Project cumulative CPU savings over an 8-hour session at varying activity ratios."""
    log.info("\n" + "=" * 60)
    log.info("E6: Power savings projection (8-hour session)")
    log.info("=" * 60)

    # Extract per-state CPU% from E1
    cpu_by_state = {}
    for r in e1_results:
        cpu_by_state[r["state"]] = r["cpu_percent"]

    cpu_active = cpu_by_state.get("ACTIVE", 100.0)
    cpu_save = cpu_by_state.get("POWER_SAVE", 50.0)
    cpu_off = cpu_by_state.get("POWER_OFF", 30.0)

    session_hours = 8.0
    # idle_to_save = 15 min, save_to_off = 30 min total (so 15 min in save before off)
    save_delay_min = 15.0
    off_delay_min = 15.0  # additional time in POWER_SAVE before POWER_OFF

    results = []

    # Vary active ratio from 10% to 100%
    for active_pct in range(10, 101, 10):
        active_frac = active_pct / 100.0
        active_hours = session_hours * active_frac
        idle_hours = session_hours - active_hours

        # Always-on baseline: CPU at ACTIVE rate for full session
        baseline_cpu_hours = session_hours * cpu_active

        # With power management:
        # Active time at full CPU
        managed_active = active_hours * cpu_active

        # Idle time split: first 15 min at ACTIVE (waiting for timeout),
        # then 15 min at POWER_SAVE, then rest at POWER_OFF
        # But idle periods may be shorter than the timeout, so cap
        idle_min = idle_hours * 60.0

        if idle_min <= save_delay_min:
            # Never reaches POWER_SAVE — all idle time at ACTIVE rate
            managed_idle = idle_hours * cpu_active
        elif idle_min <= save_delay_min + off_delay_min:
            # Reaches POWER_SAVE but not POWER_OFF
            save_wait_h = save_delay_min / 60.0
            in_save_h = idle_hours - save_wait_h
            managed_idle = save_wait_h * cpu_active + in_save_h * cpu_save
        else:
            # Reaches POWER_OFF
            save_wait_h = save_delay_min / 60.0
            in_save_h = off_delay_min / 60.0
            in_off_h = idle_hours - save_wait_h - in_save_h
            managed_idle = (save_wait_h * cpu_active +
                            in_save_h * cpu_save +
                            in_off_h * cpu_off)

        managed_cpu_hours = managed_active + managed_idle
        savings_pct = ((baseline_cpu_hours - managed_cpu_hours) / baseline_cpu_hours) * 100.0

        row = {
            "active_pct": active_pct,
            "session_hours": session_hours,
            "baseline_cpu_hours": round(baseline_cpu_hours, 1),
            "managed_cpu_hours": round(managed_cpu_hours, 1),
            "savings_pct": round(savings_pct, 1),
            "cpu_active": cpu_active,
            "cpu_save": cpu_save,
            "cpu_off": cpu_off,
        }
        results.append(row)
        log.info("  Active=%3d%%: baseline=%.1f CPU-h, managed=%.1f CPU-h, savings=%.1f%%",
                 active_pct, baseline_cpu_hours, managed_cpu_hours, savings_pct)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Plot generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_plots(log, e1, e2, e3, e4, e5, e6) -> None:
    """Generate all Phase 8 publication-quality plots."""
    apply_paper_style()
    import matplotlib.pyplot as plt

    # ── Plot 1: Resource usage bar chart ──────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    states = [r["state"] for r in e1]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]

    # CPU%
    cpus = [r["cpu_percent"] for r in e1]
    axes[0].bar(states, cpus, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_ylabel("CPU Usage (%)")
    axes[0].set_title("CPU Usage by State")
    for i, v in enumerate(cpus):
        axes[0].text(i, v + 3, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")

    # Frames processed
    frames = [r["frames_processed"] for r in e1]
    axes[1].bar(states, frames, color=colors, edgecolor="black", linewidth=0.5)
    axes[1].set_ylabel("Frames Processed")
    axes[1].set_title("Frames Processed (of 200)")
    for i, v in enumerate(frames):
        axes[1].text(i, v + 3, str(v), ha="center", fontsize=9, fontweight="bold")

    # Inference calls
    detect = [r["detect_calls"] for r in e1]
    gesture = [r["gesture_calls"] for r in e1]
    x = np.arange(len(states))
    w = 0.35
    axes[2].bar(x - w / 2, detect, w, label="Face Detection", color="#3498db", edgecolor="black", linewidth=0.5)
    axes[2].bar(x + w / 2, gesture, w, label="Gesture", color="#9b59b6", edgecolor="black", linewidth=0.5)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(states)
    axes[2].set_ylabel("Inference Calls")
    axes[2].set_title("Inference Calls by State")
    axes[2].legend()

    plt.tight_layout()
    fig.savefig(PHASE_DIR / "resource_usage_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved: resource_usage_bar.png")

    # ── Plot 2: Wake latency box plot ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))

    save_lats = [r["latency_ms"] for r in e2 if r["transition"] == "POWER_SAVE_to_ACTIVE"]
    off_lats = [r["latency_ms"] for r in e2 if r["transition"] == "POWER_OFF_to_ACTIVE"]

    bp = ax.boxplot([save_lats, off_lats],
                    labels=["POWER_SAVE\n-> ACTIVE", "POWER_OFF\n-> ACTIVE"],
                    patch_artist=True,
                    boxprops=dict(linewidth=1.5),
                    medianprops=dict(color="red", linewidth=2))
    bp["boxes"][0].set_facecolor("#f39c12")
    bp["boxes"][1].set_facecolor("#e74c3c")

    ax.set_ylabel("Latency (ms)")
    ax.set_title("Wake-Up Latency (State Machine Only)")
    ax.text(0.98, 0.95, f"N = {len(save_lats)} trials each",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, style="italic")

    fig.savefig(PHASE_DIR / "wake_latency_box.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved: wake_latency_box.png")

    # ── Plot 3: State transition timeline ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 3))

    state_map = {"ACTIVE": 2, "POWER_SAVE": 1, "POWER_OFF": 0}
    state_colors = {"ACTIVE": "#2ecc71", "POWER_SAVE": "#f39c12", "POWER_OFF": "#e74c3c"}

    times = [t["time_s"] for t in e3]
    levels = [state_map[t["state"]] for t in e3]

    # Draw step function
    for i in range(len(times) - 1):
        ax.fill_between([times[i], times[i + 1]], 0, levels[i],
                        color=state_colors[e3[i]["state"]], alpha=0.3, step="post")
        ax.plot([times[i], times[i + 1]], [levels[i], levels[i]], color=state_colors[e3[i]["state"]], linewidth=2)
        if i < len(times) - 1:
            ax.plot([times[i + 1], times[i + 1]], [levels[i], levels[i + 1]], color="black", linewidth=1, linestyle="--")

    # Mark transitions
    for i, t in enumerate(e3):
        if i > 0:  # skip initial
            ax.plot(t["time_s"], state_map[t["state"]], "ko", markersize=6)

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["POWER_OFF", "POWER_SAVE", "ACTIVE"])
    ax.set_xlabel("Time (seconds)")
    ax.set_title("State Transition Timeline (Simulated Session)")
    ax.set_ylim(-0.3, 2.5)

    fig.savefig(PHASE_DIR / "transition_timeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved: transition_timeline.png")

    # ── Plot 4: Timer accuracy scatter ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))

    save_errors = [r["save_error_ms"] for r in e4 if r["save_error_ms"] is not None]
    off_errors = [r["off_error_ms"] for r in e4 if r["off_error_ms"] is not None]
    trials = list(range(1, len(save_errors) + 1))

    ax.scatter(trials, save_errors, label="ACTIVE -> POWER_SAVE", marker="o", s=60, color="#f39c12", edgecolors="black", zorder=3)
    ax.scatter(trials, off_errors, label="POWER_SAVE -> POWER_OFF", marker="s", s=60, color="#e74c3c", edgecolors="black", zorder=3)
    ax.axhline(y=100, color="red", linestyle="--", alpha=0.5, label="100ms tolerance")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Timer Error (ms)")
    ax.set_title("Idle Timer Accuracy")
    ax.legend()

    fig.savefig(PHASE_DIR / "timer_accuracy_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved: timer_accuracy_scatter.png")

    # ── Plot 5: Power savings projection ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))

    active_pcts = [r["active_pct"] for r in e6]
    savings = [r["savings_pct"] for r in e6]

    ax.bar(active_pcts, savings, width=8, color="#3498db", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Active Usage (%)")
    ax.set_ylabel("CPU Savings (%)")
    ax.set_title("Projected CPU Savings (8-Hour Session, 15-min Timeouts)")
    ax.set_xticks(active_pcts)

    for i, (pct, sav) in enumerate(zip(active_pcts, savings)):
        if sav > 0:
            ax.text(pct, sav + 0.3, f"{sav:.1f}%", ha="center", fontsize=8, fontweight="bold")

    fig.savefig(PHASE_DIR / "power_savings_projection.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved: power_savings_projection.png")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 8 - Power Management Benchmark")
    parser.add_argument("--n-frames", type=int, default=200, help="Frames for E1 (default: 200)")
    parser.add_argument("--n-trials", type=int, default=10, help="Trials for E2/E4 (default: 10)")
    parser.add_argument("--idle-save", type=float, default=3.0, help="Idle-to-save seconds for E3 (default: 3)")
    parser.add_argument("--idle-off", type=float, default=6.0, help="Idle-to-off seconds for E3 (default: 6)")
    args = parser.parse_args()

    log = setup_logging("phase8_power_bench")
    _ensure_dirs()

    log.info("Phase 8 - Power Management Benchmark")
    log.info("psutil available: %s", _HAS_PSUTIL)

    # ── E1: Resource usage ─────────────────────────────────────────────────
    e1_results = experiment_resource_usage(log, n_frames=args.n_frames)
    e1_path = PHASE_DIR / "resource_usage.csv"
    with open(e1_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=e1_results[0].keys())
        w.writeheader()
        w.writerows(e1_results)
    log.info("Saved: %s", e1_path)

    # ── E2: Wake-up latency ───────────────────────────────────────────────
    e2_results = experiment_wake_latency(log, n_trials=args.n_trials)
    e2_path = PHASE_DIR / "wake_latency.csv"
    with open(e2_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=e2_results[0].keys())
        w.writeheader()
        w.writerows(e2_results)
    log.info("Saved: %s", e2_path)

    # ── E3: Transition timeline ───────────────────────────────────────────
    e3_results = experiment_transition_timeline(log, idle_save=args.idle_save, idle_off=args.idle_off)
    e3_path = PHASE_DIR / "transition_timeline.csv"
    with open(e3_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=e3_results[0].keys())
        w.writeheader()
        w.writerows(e3_results)
    log.info("Saved: %s", e3_path)

    # ── E4: Timer accuracy ────────────────────────────────────────────────
    e4_results = experiment_timer_accuracy(log, n_trials=args.n_trials)
    e4_path = PHASE_DIR / "timer_accuracy.csv"
    with open(e4_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=e4_results[0].keys())
        w.writeheader()
        w.writerows(e4_results)
    log.info("Saved: %s", e4_path)

    # ── E5: Thread safety ─────────────────────────────────────────────────
    e5_results = experiment_thread_safety(log)
    e5_path = PHASE_DIR / "thread_safety.csv"
    with open(e5_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=e5_results[0].keys())
        w.writeheader()
        w.writerows(e5_results)
    log.info("Saved: %s", e5_path)

    # ── E6: Power savings projection ──────────────────────────────────────
    e6_results = experiment_power_projection(log, e1_results)
    e6_path = PHASE_DIR / "power_projection.csv"
    with open(e6_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=e6_results[0].keys())
        w.writeheader()
        w.writerows(e6_results)
    log.info("Saved: %s", e6_path)

    # ── Generate plots ────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("Generating plots...")
    log.info("=" * 60)
    generate_plots(log, e1_results, e2_results, e3_results, e4_results, e5_results, e6_results)

    # ── Summary ───────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("PHASE 8 COMPLETE - All results in %s", PHASE_DIR)
    log.info("=" * 60)

    log.info("\nE1 Resource Usage:")
    for r in e1_results:
        log.info("  %-12s  FPS=%6.1f  CPU=%5.1f%%  Frames=%d/%d",
                 r["state"], r["fps"], r["cpu_percent"], r["frames_processed"], r["n_input_frames"])

    save_lats = [r["latency_ms"] for r in e2_results if r["transition"] == "POWER_SAVE_to_ACTIVE"]
    off_lats = [r["latency_ms"] for r in e2_results if r["transition"] == "POWER_OFF_to_ACTIVE"]
    log.info("\nE2 Wake Latency:")
    log.info("  POWER_SAVE -> ACTIVE: mean=%.4fms, max=%.4fms", np.mean(save_lats), np.max(save_lats))
    log.info("  POWER_OFF  -> ACTIVE: mean=%.4fms, max=%.4fms", np.mean(off_lats), np.max(off_lats))

    log.info("\nE3 Timeline: %d transitions logged", len(e3_results))

    all_pass = all(r["save_pass"] and r["off_pass"] for r in e4_results)
    log.info("\nE4 Timer Accuracy: %s", "ALL PASS" if all_pass else "SOME FAILED")

    log.info("\nE5 Thread Safety: %s (%d ops, %d errors)",
             "PASS" if e5_results[0]["pass"] else "FAIL",
             e5_results[0]["total_ops"], e5_results[0]["errors"])

    best_savings = max(r["savings_pct"] for r in e6_results)
    log.info("\nE6 Power Projection: max savings = %.1f%% (at 10%% active usage)", best_savings)


if __name__ == "__main__":
    main()
