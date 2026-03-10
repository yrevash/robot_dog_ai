"""
experiments/utils.py
====================
Shared utilities for all REVO experiment scripts.
Imported by every other script in this folder.
"""

import csv
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
EXPERIMENTS   = Path(__file__).resolve().parent
RESULTS_DIR   = PROJECT_ROOT / "results"
KNOWN_FACES   = PROJECT_ROOT / "data" / "known_faces"
TEST_FACES    = PROJECT_ROOT / "data" / "test_faces"
GESTURE_DATA  = PROJECT_ROOT / "data" / "gesture_dataset"
MODELS_DIR    = PROJECT_ROOT / "models"
DB_FILE       = PROJECT_ROOT / "data" / "face_db.npz"

# Ensure project root AND src/ are importable from every script
for _p in [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT), str(EXPERIMENTS)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ── Logging ────────────────────────────────────────────────────────────────────
def setup_logging(phase_name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Creates a logger that writes to both stdout and a timestamped log file
    under results/logs/<phase_name>_YYYYMMDD_HHMMSS.log
    """
    log_dir = RESULTS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{phase_name}_{ts}.log"

    fmt     = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(fmt)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    log = logging.getLogger(phase_name)
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    log.addHandler(console)
    log.addHandler(fh)
    log.propagate = False

    log.info("=" * 60)
    log.info(f"Phase: {phase_name}")
    log.info(f"Log file: {log_file}")
    log.info(f"Project root: {PROJECT_ROOT}")
    log.info("=" * 60)
    return log


def get_results_dir(phase: str) -> Path:
    d = RESULTS_DIR / phase
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── CSV helpers ────────────────────────────────────────────────────────────────
def save_csv(path: Path, rows: list, fieldnames: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def load_csv(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(y_true: list, y_pred: list, enrolled_names: set) -> dict:
    """
    y_true / y_pred: list of string names or "Unknown"
    enrolled_names:  set of names that ARE in the DB

    Returns dict with TAR, FAR, FRR, accuracy, TP, FP, FN, TN counts.
    """
    TP = FP = FN = TN = 0
    for true, pred in zip(y_true, y_pred):
        is_enrolled = true in enrolled_names
        was_accepted = pred != "Unknown"
        correct_id   = (pred == true)

        if is_enrolled and was_accepted and correct_id:
            TP += 1          # enrolled, accepted with right name
        elif is_enrolled and (not was_accepted or not correct_id):
            FN += 1          # enrolled, rejected or wrong name
        elif not is_enrolled and was_accepted:
            FP += 1          # impostor accepted
        else:
            TN += 1          # impostor correctly rejected

    total_enrolled  = TP + FN
    total_impostor  = FP + TN
    TAR = TP / total_enrolled  if total_enrolled  > 0 else 0.0
    FAR = FP / total_impostor  if total_impostor  > 0 else 0.0
    FRR = FN / total_enrolled  if total_enrolled  > 0 else 0.0
    ACC = (TP + TN) / (total_enrolled + total_impostor) if (total_enrolled + total_impostor) > 0 else 0.0

    return dict(TAR=TAR, FAR=FAR, FRR=FRR, ACC=ACC,
                TP=TP, FP=FP, FN=FN, TN=TN,
                total_enrolled=total_enrolled, total_impostor=total_impostor)


# ── Matplotlib style ───────────────────────────────────────────────────────────
def apply_paper_style() -> None:
    """Apply clean, paper-ready matplotlib style."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi":        150,
        "font.size":         11,
        "axes.titlesize":    12,
        "axes.labelsize":    11,
        "legend.fontsize":   10,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "lines.linewidth":   2,
        "figure.autolayout": True,
    })


# ── Data check helpers ─────────────────────────────────────────────────────────
def check_db_exists(log: logging.Logger) -> bool:
    if not DB_FILE.exists():
        log.error(f"face_db.npz not found at {DB_FILE}")
        log.error("Run: python face_embedding.py build")
        return False
    return True


def check_test_faces_exist(log: logging.Logger) -> bool:
    if not TEST_FACES.exists():
        log.warning(f"test_faces/ not found at {TEST_FACES}")
        log.warning("Running in DEMO mode using known_faces/ instead.")
        return False
    gt = TEST_FACES / "ground_truth.csv"
    if not gt.exists():
        log.warning(f"ground_truth.csv not found at {gt}")
        log.warning("Running in DEMO mode using known_faces/ instead.")
        return False
    return True


def check_gesture_data_exist(log: logging.Logger) -> bool:
    if not GESTURE_DATA.exists():
        log.error(f"gesture_dataset/ not found at {GESTURE_DATA}")
        log.error("Run: python experiments/collect_gesture_dataset.py first")
        return False
    return True


def print_metrics_table(metrics_dict: dict, log: logging.Logger) -> None:
    """Print a formatted metrics table to log."""
    log.info("-" * 60)
    log.info(f"{'Config':<30} {'TAR':>6} {'FAR':>6} {'FRR':>6} {'ACC':>6}")
    log.info("-" * 60)
    for name, m in metrics_dict.items():
        log.info(f"{name:<30} {m['TAR']:>6.3f} {m['FAR']:>6.3f} {m['FRR']:>6.3f} {m['ACC']:>6.3f}")
    log.info("-" * 60)
