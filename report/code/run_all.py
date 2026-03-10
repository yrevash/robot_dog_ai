#!/usr/bin/env python3
"""
experiments/run_all.py
=======================
Master runner — runs all REVO experiments in the correct order and
produces a consolidated Markdown summary report.

Experiment phases
-----------------
  Phase 2a  eval_face_recognition.py  (demo mode)
  Phase 2b  sweep_threshold.py
  Phase 3   sweep_voting.py
  Phase 4   eval_gesture.py           (dataset mode if data exists, else skip)
  Phase 5   bench_rpi.py              (--camera -1  synthetic mode)
  Phase 6   latency_measure.py        (component timing only, no server)

Each phase is run as an isolated subprocess so a failure in one phase does
not abort subsequent phases.  All stdout/stderr is captured to::

    results/logs/run_all_<timestamp>.log

After all phases finish the script reads every output CSV and writes::

    results/SUMMARY.md

Usage
-----
    python experiments/run_all.py              # run everything
    python experiments/run_all.py --only 2     # run only phase 2 (both 2a and 2b)
    python experiments/run_all.py --skip 4 5   # skip phases 4 and 5
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Bootstrap project paths ───────────────────────────────────────────────────
_EXPERIMENTS = Path(__file__).resolve().parent
_PROJECT_ROOT = _EXPERIMENTS.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

from utils import (  # noqa: E402
    DB_FILE,
    GESTURE_DATA,
    KNOWN_FACES,
    MODELS_DIR,
    RESULTS_DIR,
    setup_logging,
)

# ── Experiment registry ───────────────────────────────────────────────────────
# Each entry: (phase_number, label, script_path_relative_to_experiments, extra_cli_args)
_EXPERIMENTS_DEF: list[tuple[int, str, str, list[str]]] = [
    (2,  "Phase 2a: Face Recognition (demo)",    "eval_face_recognition.py",  ["--mode", "demo"]),
    (2,  "Phase 2b: Threshold Sweep (ROC)",      "sweep_threshold.py",        []),
    (3,  "Phase 3:  Voting Window Sweep",        "sweep_voting.py",           []),
    (4,  "Phase 4:  Gesture Accuracy Eval",      "eval_gesture.py",           []),
    (5,  "Phase 5:  RPi Benchmark (synthetic)",  "bench_rpi.py",              ["--camera", "-1"]),
    (6,  "Phase 6:  Latency Breakdown",          "latency_measure.py",        []),
]

# Expected output CSVs produced by each phase (used for the summary report)
_PHASE_CSVS: dict[str, Path] = {
    "gate_comparison":        RESULTS_DIR / "phase2" / "gate_comparison.csv",
    "lighting_ablation":      RESULTS_DIR / "phase2" / "lighting_ablation.csv",
    "voting_sweep":           RESULTS_DIR / "phase3" / "voting_sweep.csv",
    "gesture_per_class":      RESULTS_DIR / "phase4" / "gesture_per_class_metrics.csv",
    "benchmark_results":      RESULTS_DIR / "phase5" / "benchmark_results.csv",
    "latency_summary":        RESULTS_DIR / "phase6" / "latency_summary.csv",
}

_SUBPROCESS_TIMEOUT = 300   # seconds per experiment
_SUMMARY_PATH = RESULTS_DIR / "SUMMARY.md"


# ── Prerequisite checks ───────────────────────────────────────────────────────

def check_prerequisites(log) -> dict[str, bool]:
    """
    Inspect the workspace and report readiness.
    Returns a dict of check_name -> bool.
    """
    checks: dict[str, bool] = {}

    # face_db.npz
    if DB_FILE.exists():
        log.info(f"[OK] face_db.npz found: {DB_FILE}")
        checks["face_db"] = True
    else:
        log.warning(f"[WARN] face_db.npz NOT found at {DB_FILE}")
        log.warning("       Phases 2 and 3 may fail.  Run: python face_embedding.py build")
        checks["face_db"] = False

    # known_faces/
    if KNOWN_FACES.exists():
        images = list(KNOWN_FACES.rglob("*.jpg")) + list(KNOWN_FACES.rglob("*.png"))
        if images:
            log.info(f"[OK] known_faces/ has {len(images)} image(s).")
            checks["known_faces"] = True
        else:
            log.warning("[WARN] known_faces/ exists but contains no images.")
            checks["known_faces"] = False
    else:
        log.warning(f"[WARN] known_faces/ not found at {KNOWN_FACES}")
        checks["known_faces"] = False

    # gesture_dataset/
    if GESTURE_DATA.exists():
        samples = list(GESTURE_DATA.rglob("*.jpg")) + list(GESTURE_DATA.rglob("*.png"))
        log.info(f"[OK] gesture_dataset/ found with {len(samples)} sample(s).")
        checks["gesture_data"] = True
    else:
        log.info("[INFO] gesture_dataset/ not found — Phase 4 will be skipped gracefully.")
        checks["gesture_data"] = False

    # ONNX models
    model_files = list(MODELS_DIR.glob("*.onnx")) + list(MODELS_DIR.glob("*.task"))
    if model_files:
        log.info(f"[OK] {len(model_files)} model file(s) in {MODELS_DIR}.")
        checks["models"] = True
    else:
        log.warning(f"[WARN] No model files in {MODELS_DIR} — models will be downloaded on first run.")
        checks["models"] = False

    return checks


# ── Subprocess runner ─────────────────────────────────────────────────────────

def run_experiment(
    label: str,
    script_path: Path,
    extra_args: list[str],
    log_fh,
    log,
) -> tuple[bool, int]:
    """
    Run a single experiment script as a subprocess.

    Parameters
    ----------
    label:       Human-readable label for logging.
    script_path: Absolute path to the .py script.
    extra_args:  Extra CLI arguments forwarded to the script.
    log_fh:      Open file handle for the master log (stdout+stderr written here).
    log:         Logger for console output.

    Returns
    -------
    (success, output_file_count)
    """
    if not script_path.exists():
        log.warning(f"  Script not found, skipping: {script_path}")
        log_fh.write(f"\n{'='*60}\n[SKIP] {label} — script not found: {script_path}\n{'='*60}\n")
        return False, 0

    cmd = [sys.executable, str(script_path)] + extra_args
    log.info(f"  CMD: {' '.join(cmd)}")
    log_fh.write(f"\n{'='*60}\n[RUN] {label}\nCMD: {' '.join(cmd)}\n{'='*60}\n")
    log_fh.flush()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(_PROJECT_ROOT),
            timeout=_SUBPROCESS_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        log.error(f"  TIMEOUT after {_SUBPROCESS_TIMEOUT}s: {label}")
        log_fh.write(f"\n[TIMEOUT] {label}\n")
        return False, 0
    except Exception as exc:
        log.error(f"  EXCEPTION running {label}: {exc}")
        log_fh.write(f"\n[EXCEPTION] {label}: {exc}\n")
        return False, 0

    log_fh.write(result.stdout)
    if result.stderr:
        log_fh.write("\n--- STDERR ---\n")
        log_fh.write(result.stderr)
    log_fh.write(f"\n[EXIT CODE] {result.returncode}\n")
    log_fh.flush()

    success = result.returncode == 0
    if success:
        log.info(f"  PASS (exit 0)")
    else:
        log.error(f"  FAIL (exit {result.returncode})")
        if result.stderr.strip():
            # Print last 10 lines of stderr to console for quick diagnosis
            tail = result.stderr.strip().splitlines()[-10:]
            for line in tail:
                log.error(f"    {line}")

    # Count output files produced (rough proxy for "something happened")
    output_count = _count_new_outputs()
    return success, output_count


def _count_new_outputs() -> int:
    """Count CSV and PNG files across all results/ phase dirs."""
    total = 0
    for phase_dir in RESULTS_DIR.glob("phase*"):
        total += len(list(phase_dir.glob("*.csv")))
        total += len(list(phase_dir.glob("*.png")))
    return total


# ── CSV → Markdown table converter ───────────────────────────────────────────

def _csv_to_md_table(path: Path, max_rows: int = 50) -> str:
    """
    Read a CSV and return a GitHub-flavoured Markdown table string.
    Returns an empty string if the file is missing or unreadable.
    """
    if not path.exists():
        return ""
    try:
        rows = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            for i, row in enumerate(reader):
                if i >= max_rows:
                    break
                rows.append(row)
        if not fieldnames or not rows:
            return ""

        # Header
        header  = "| " + " | ".join(fieldnames) + " |"
        divider = "| " + " | ".join("---" for _ in fieldnames) + " |"
        lines   = [header, divider]

        # Data rows — truncate long cells
        for row in rows:
            cells = []
            for col in fieldnames:
                val = str(row.get(col, "")).replace("|", "\\|")
                if len(val) > 40:
                    val = val[:37] + "..."
                cells.append(val)
            lines.append("| " + " | ".join(cells) + " |")

        return "\n".join(lines)
    except Exception:
        return ""


# ── Key findings extractor ────────────────────────────────────────────────────

def _extract_key_findings(log) -> list[str]:
    """
    Read output CSVs and auto-generate bullet-point findings.
    Gracefully handles missing files.
    """
    findings: list[str] = []

    # Gate comparison
    p = _PHASE_CSVS["gate_comparison"]
    if p.exists():
        try:
            rows: list[dict] = []
            with open(p, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if rows:
                best = max(rows, key=lambda r: float(r.get("ACC", 0) or 0))
                worst_far = max(rows, key=lambda r: float(r.get("FAR", 1) or 1))
                findings.append(
                    f"Best gate config by accuracy: **{best.get('config', best.get('Config', '?'))}** "
                    f"(ACC={float(best.get('ACC', 0)):.3f}, "
                    f"TAR={float(best.get('TAR', 0)):.3f}, "
                    f"FAR={float(best.get('FAR', 0)):.3f})."
                )
                findings.append(
                    f"Highest FAR (worst impostor rejection): "
                    f"**{worst_far.get('config', worst_far.get('Config', '?'))}** "
                    f"(FAR={float(worst_far.get('FAR', 0)):.3f})."
                )
        except Exception as exc:
            log.debug(f"key findings — gate_comparison: {exc}")

    # Voting sweep
    p = _PHASE_CSVS["voting_sweep"]
    if p.exists():
        try:
            with open(p, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if rows:
                # Try to find column that looks like accuracy
                acc_col = next(
                    (c for c in rows[0] if "acc" in c.lower() or "tar" in c.lower()), None
                )
                if acc_col:
                    best = max(rows, key=lambda r: float(r.get(acc_col, 0) or 0))
                    findings.append(
                        f"Best voting window config: "
                        f"**{best.get('window', best.get('config', '?'))}** "
                        f"({acc_col}={float(best.get(acc_col, 0)):.3f})."
                    )
        except Exception as exc:
            log.debug(f"key findings — voting_sweep: {exc}")

    # Gesture per-class
    p = _PHASE_CSVS["gesture_per_class"]
    if p.exists():
        try:
            with open(p, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if rows:
                acc_col = next(
                    (c for c in rows[0] if "acc" in c.lower() or "f1" in c.lower()), None
                )
                cls_col = next(
                    (c for c in rows[0] if "class" in c.lower() or "gesture" in c.lower()), None
                )
                if acc_col and cls_col:
                    best  = max(rows, key=lambda r: float(r.get(acc_col, 0) or 0))
                    worst = min(rows, key=lambda r: float(r.get(acc_col, 0) or 0))
                    findings.append(
                        f"Highest gesture accuracy: **{best.get(cls_col, '?')}** "
                        f"({acc_col}={float(best.get(acc_col, 0)):.3f})."
                    )
                    findings.append(
                        f"Lowest gesture accuracy: **{worst.get(cls_col, '?')}** "
                        f"({acc_col}={float(worst.get(acc_col, 0)):.3f})."
                    )
        except Exception as exc:
            log.debug(f"key findings — gesture_per_class: {exc}")

    # Latency summary
    p = _PHASE_CSVS["latency_summary"]
    if p.exists():
        try:
            with open(p, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if rows:
                for row in rows:
                    comp = row.get("component", row.get("Component", "?"))
                    mean = row.get("mean_ms", row.get("Mean_ms", row.get("mean", "?")))
                    findings.append(f"Latency — {comp}: {mean} ms (mean).")
        except Exception as exc:
            log.debug(f"key findings — latency_summary: {exc}")

    if not findings:
        findings.append("No output CSV data found — run experiments first to populate findings.")

    return findings


# ── SUMMARY.md writer ─────────────────────────────────────────────────────────

def write_summary(phase_results: list[dict], log) -> None:
    """
    Read all output CSVs and write results/SUMMARY.md.

    phase_results is a list of dicts with keys:
        label, phase, success, output_count, skipped
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines += [
        "# REVO Experiment Summary",
        "",
        f"Generated: {ts}",
        "",
        "---",
        "",
    ]

    # ── Run status table ──────────────────────────────────────────────────────
    lines += [
        "## Experiment Run Status",
        "",
        "| Phase | Experiment | Status | Output Files |",
        "| --- | --- | --- | --- |",
    ]
    for r in phase_results:
        status  = "PASS" if r["success"] else ("SKIP" if r["skipped"] else "FAIL")
        emoji   = {"PASS": "✓", "SKIP": "–", "FAIL": "✗"}[status]
        lines.append(
            f"| {r['phase']} | {r['label']} | {emoji} {status} | {r['output_count']} files |"
        )
    lines += ["", "---", ""]

    # ── Phase 2a: Gate comparison ─────────────────────────────────────────────
    lines += ["## Phase 2a — Face Recognition: Gate Configuration Comparison", ""]
    table = _csv_to_md_table(_PHASE_CSVS["gate_comparison"])
    if table:
        lines += [table, ""]
    else:
        lines += ["_No data — run Phase 2a first._", ""]

    # ── Phase 2a: Lighting ablation ───────────────────────────────────────────
    lines += ["## Phase 2a — Face Recognition: Lighting Normalisation Ablation", ""]
    table = _csv_to_md_table(_PHASE_CSVS["lighting_ablation"])
    if table:
        lines += [table, ""]
    else:
        lines += ["_No data — run Phase 2a first._", ""]

    lines += ["---", ""]

    # ── Phase 3: Voting sweep ─────────────────────────────────────────────────
    lines += ["## Phase 3 — Temporal Voting Window Sweep", ""]
    table = _csv_to_md_table(_PHASE_CSVS["voting_sweep"])
    if table:
        lines += [table, ""]
    else:
        lines += ["_No data — run Phase 3 first._", ""]

    lines += ["---", ""]

    # ── Phase 4: Gesture per-class ────────────────────────────────────────────
    lines += ["## Phase 4 — Gesture Recognition: Per-Class Metrics", ""]
    table = _csv_to_md_table(_PHASE_CSVS["gesture_per_class"])
    if table:
        lines += [table, ""]
    else:
        lines += [
            "_No gesture data — collect a dataset first with:_",
            "",
            "```",
            "python experiments/collect_gesture_dataset.py --name <YourName>",
            "```",
            "",
        ]

    lines += ["---", ""]

    # ── Phase 5: RPi benchmark ────────────────────────────────────────────────
    lines += ["## Phase 5 — Raspberry Pi Benchmark Results", ""]
    table = _csv_to_md_table(_PHASE_CSVS["benchmark_results"])
    if table:
        lines += [table, ""]
    else:
        lines += ["_No data — run Phase 5 first._", ""]

    lines += ["---", ""]

    # ── Phase 6: Latency breakdown ────────────────────────────────────────────
    lines += ["## Phase 6 — Latency Breakdown", ""]
    table = _csv_to_md_table(_PHASE_CSVS["latency_summary"])
    if table:
        lines += [table, ""]
    else:
        lines += ["_No data — run Phase 6 first._", ""]

    lines += ["---", ""]

    # ── Key findings ──────────────────────────────────────────────────────────
    lines += ["## Key Findings", ""]
    findings = _extract_key_findings(log)
    for f in findings:
        lines.append(f"- {f}")
    lines += ["", "---", ""]

    # ── Figures ───────────────────────────────────────────────────────────────
    png_files = sorted(RESULTS_DIR.rglob("*.png"))
    lines += ["## Figures", ""]
    if png_files:
        for p in png_files:
            rel = p.relative_to(RESULTS_DIR)
            lines.append(f"- `results/{rel}`")
    else:
        lines.append("_No figures generated yet._")
    lines += ["", "---", ""]

    lines += [
        "_Report auto-generated by `experiments/run_all.py`._",
        "",
    ]

    _SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Summary written: {_SUMMARY_PATH}")


# ── Status table (console) ────────────────────────────────────────────────────
_BOX_TOP    = "┌" + "─" * 37 + "┬" + "─" * 10 + "┬" + "─" * 10 + "┐"
_BOX_SEP    = "├" + "─" * 37 + "┼" + "─" * 10 + "┼" + "─" * 10 + "┤"
_BOX_BOT    = "└" + "─" * 37 + "┴" + "─" * 10 + "┴" + "─" * 10 + "┘"
_BOX_HDR    = "│ {:<35} │ {:<8} │ {:<8} │"
_BOX_ROW    = "│ {:<35} │ {:<8} │ {:<8} │"


def _print_status_table(phase_results: list[dict]) -> None:
    print()
    print(_BOX_TOP)
    print(_BOX_HDR.format("Experiment", "Status", "Output"))
    print(_BOX_SEP)
    for r in phase_results:
        if r["skipped"]:
            status = "– SKIP"
        elif r["success"]:
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
        output = f"{r['output_count']} files"
        # Truncate label to fit column
        label = r["label"]
        if len(label) > 35:
            label = label[:32] + "..."
        print(_BOX_ROW.format(label, status, output))
    print(_BOX_BOT)
    print()


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Master runner for all REVO experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python experiments/run_all.py              # run everything
              python experiments/run_all.py --only 2     # run only phase 2
              python experiments/run_all.py --skip 4 5   # skip phases 4 and 5
        """),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--only",
        type=int,
        metavar="PHASE",
        help="Run only experiments whose phase number equals PHASE.",
    )
    group.add_argument(
        "--skip",
        type=int,
        nargs="+",
        metavar="PHASE",
        help="Skip experiments with these phase numbers.",
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    log  = setup_logging("run_all")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = RESULTS_DIR / "logs" / f"run_all_{ts}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("REVO Experiment Master Runner")
    log.info(f"Project root: {_PROJECT_ROOT}")
    log.info(f"Master log:   {log_path}")
    log.info("=" * 60)

    # ── Prerequisites ─────────────────────────────────────────────────────────
    log.info("\nChecking prerequisites …")
    prereqs = check_prerequisites(log)

    # ── Determine which phases to run ─────────────────────────────────────────
    skip_phases: set[int] = set()
    if args.skip:
        skip_phases = set(args.skip)
    only_phase: Optional[int] = args.only

    # Phase 4 requires gesture data; skip automatically if absent
    if not prereqs["gesture_data"] and 4 not in skip_phases:
        log.info("gesture_dataset/ absent — Phase 4 will be marked as skipped.")
        skip_phases.add(4)

    log.info("\nStarting experiments …\n")

    phase_results: list[dict] = []
    cumulative_output_before = _count_new_outputs()

    with open(log_path, "w", encoding="utf-8") as log_fh:
        log_fh.write(f"REVO run_all master log — {ts}\n")
        log_fh.write(f"Project root: {_PROJECT_ROOT}\n\n")

        for phase_num, label, script_rel, extra_args in _EXPERIMENTS_DEF:
            script_path = _EXPERIMENTS / script_rel

            # Apply --only / --skip filters
            if only_phase is not None and phase_num != only_phase:
                phase_results.append({
                    "phase":        phase_num,
                    "label":        label,
                    "success":      False,
                    "output_count": 0,
                    "skipped":      True,
                })
                continue

            if phase_num in skip_phases:
                log.info(f"[SKIP] {label}")
                log_fh.write(f"\n[SKIP] {label} (phase {phase_num} excluded by user)\n")
                phase_results.append({
                    "phase":        phase_num,
                    "label":        label,
                    "success":      False,
                    "output_count": 0,
                    "skipped":      True,
                })
                continue

            log.info(f"[RUN ] {label}")
            outputs_before = _count_new_outputs()
            success, _ = run_experiment(label, script_path, extra_args, log_fh, log)
            outputs_after  = _count_new_outputs()
            new_outputs    = max(0, outputs_after - outputs_before)

            phase_results.append({
                "phase":        phase_num,
                "label":        label,
                "success":      success,
                "output_count": new_outputs,
                "skipped":      False,
            })
            log.info("")

    # ── Write SUMMARY.md ──────────────────────────────────────────────────────
    log.info("Generating SUMMARY.md …")
    try:
        write_summary(phase_results, log)
    except Exception as exc:
        log.error(f"Failed to write SUMMARY.md: {exc}")

    # ── Console status table ──────────────────────────────────────────────────
    _print_status_table(phase_results)

    # ── Final counts ──────────────────────────────────────────────────────────
    passed  = sum(1 for r in phase_results if r["success"])
    failed  = sum(1 for r in phase_results if not r["success"] and not r["skipped"])
    skipped = sum(1 for r in phase_results if r["skipped"])

    log.info(f"Results: {passed} passed, {failed} failed, {skipped} skipped.")
    log.info(f"Master log:  {log_path}")
    log.info(f"Summary:     {_SUMMARY_PATH}")
    log.info("Done.")

    # Exit with non-zero if any experiment actually failed (not just skipped)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
