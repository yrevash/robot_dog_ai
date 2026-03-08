#!/usr/bin/env python3
"""
Face embedding pipeline for Raspberry Pi / robot projects.

Features:
1. Capture approved faces only (per person).
2. Build face embeddings database from captured images.
3. Recognize only enrolled faces in real time.
4. Handle lighting variation using normalization + augmentation.
5. Use strict identity gating to reduce false acceptance.

Folder layout:
    known_faces/
      Alice/
        001.jpg
        002.jpg
      Bob/
        001.jpg

Usage:
    python face_embedding.py capture --name Alice --samples 25
    python face_embedding.py build
    python face_embedding.py recognize

One-step enrollment:
    python face_embedding.py enroll --name Alice --samples 25
"""

from __future__ import annotations

import argparse
import shutil
import time
import urllib.request
from collections import Counter, deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np


KNOWN_FACES_DIR = Path("known_faces")
MODELS_DIR = Path("models")
DB_FILE = Path("face_db.npz")

DETECTOR_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
RECOGNIZER_MODEL = MODELS_DIR / "face_recognition_sface_2021dec.onnx"

MODEL_URLS = {
    DETECTOR_MODEL: "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    RECOGNIZER_MODEL: "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}


def check_opencv_requirements() -> None:
    has_detector = hasattr(cv2, "FaceDetectorYN_create") or hasattr(cv2, "FaceDetectorYN")
    has_recognizer = hasattr(cv2, "FaceRecognizerSF_create") or hasattr(cv2, "FaceRecognizerSF")
    if not (has_detector and has_recognizer):
        raise SystemExit(
            "OpenCV face modules are missing.\n"
            "Install: python -m pip install --upgrade opencv-contrib-python numpy"
        )


def download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Downloading model -> {dst.name}")
    urllib.request.urlretrieve(url, str(dst))


def _validate_model_file(path: Path) -> bool:
    try:
        model = str(path)
        if path == DETECTOR_MODEL:
            if hasattr(cv2, "FaceDetectorYN_create"):
                _ = cv2.FaceDetectorYN_create(model, "", (320, 320), 0.9, 0.3, 5000)
            else:
                _ = cv2.FaceDetectorYN.create(model, "", (320, 320), 0.9, 0.3, 5000)
        elif path == RECOGNIZER_MODEL:
            if hasattr(cv2, "FaceRecognizerSF_create"):
                _ = cv2.FaceRecognizerSF_create(model, "")
            else:
                _ = cv2.FaceRecognizerSF.create(model, "")
        else:
            _ = cv2.dnn.readNet(model)
        return True
    except Exception:
        return False


def ensure_models(auto_download: bool = True) -> None:
    missing = [p for p in MODEL_URLS if not p.exists()]
    if missing and not auto_download:
        names = ", ".join(str(p) for p in missing)
        raise SystemExit(f"Missing model files: {names}")

    for path in MODEL_URLS:
        if not path.exists():
            download_file(MODEL_URLS[path], path)

        if _validate_model_file(path):
            continue

        if not auto_download:
            raise SystemExit(f"Model file is invalid: {path}")

        print(f"[WARN] Invalid model file detected, re-downloading: {path.name}")
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        download_file(MODEL_URLS[path], path)

        if not _validate_model_file(path):
            raise SystemExit(f"Model parse failed after re-download: {path}")


def create_detector(
    input_size: Tuple[int, int],
    score_threshold: float = 0.9,
    nms_threshold: float = 0.3,
    top_k: int = 5000,
):
    model = str(DETECTOR_MODEL)
    if hasattr(cv2, "FaceDetectorYN_create"):
        return cv2.FaceDetectorYN_create(model, "", input_size, score_threshold, nms_threshold, top_k)
    return cv2.FaceDetectorYN.create(model, "", input_size, score_threshold, nms_threshold, top_k)


def create_recognizer():
    model = str(RECOGNIZER_MODEL)
    if hasattr(cv2, "FaceRecognizerSF_create"):
        return cv2.FaceRecognizerSF_create(model, "")
    return cv2.FaceRecognizerSF.create(model, "")


def normalize_lighting(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)

    mean_l = max(1.0, float(np.mean(l_eq)))
    gamma = np.log(128.0 / 255.0) / np.log(mean_l / 255.0)
    gamma = float(np.clip(gamma, 0.75, 1.45))

    l_float = np.power(l_eq.astype(np.float32) / 255.0, gamma) * 255.0
    l_fix = np.clip(l_float, 0, 255).astype(np.uint8)

    merged = cv2.merge((l_fix, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def lighting_variants(frame: np.ndarray) -> List[np.ndarray]:
    variants = [frame]
    variants.append(normalize_lighting(frame))
    variants.append(cv2.convertScaleAbs(frame, alpha=1.0, beta=20))
    variants.append(cv2.convertScaleAbs(frame, alpha=1.0, beta=-20))
    return variants


def detect_faces(detector, frame: np.ndarray) -> List[np.ndarray]:
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    if faces is None or len(faces) == 0:
        return []
    return sorted(faces, key=lambda f: float(f[-1]), reverse=True)


def bbox(face: np.ndarray) -> Tuple[int, int, int, int]:
    x, y, w, h = face[:4]
    x1 = int(max(0, x))
    y1 = int(max(0, y))
    x2 = int(max(0, x + w))
    y2 = int(max(0, y + h))
    return x1, y1, x2, y2


def face_area_ratio(face: np.ndarray, frame_shape: Tuple[int, int, int]) -> float:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox(face)
    fw = max(0, x2 - x1)
    fh = max(0, y2 - y1)
    if w <= 0 or h <= 0:
        return 0.0
    return (fw * fh) / float(w * h)


def face_quality_ok(
    face: np.ndarray,
    frame_shape: Tuple[int, int, int],
    min_area_ratio: float = 0.08,
    edge_margin: int = 8,
) -> Tuple[bool, str]:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox(face)
    fw = max(0, x2 - x1)
    fh = max(0, y2 - y1)
    if fw == 0 or fh == 0:
        return False, "invalid face box"

    ratio = (fw * fh) / float(w * h)
    if ratio < min_area_ratio:
        return False, "move closer"

    if x1 < edge_margin or y1 < edge_margin or x2 > (w - edge_margin) or y2 > (h - edge_margin):
        return False, "center your face"

    if float(face[-1]) < 0.9:
        return False, "hold still"

    return True, "ready"


def l2_normalize(v: np.ndarray) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return None
    return v / n


def embedding_from_face(frame: np.ndarray, face: np.ndarray, recognizer) -> Optional[np.ndarray]:
    aligned = recognizer.alignCrop(frame, face)
    feat = recognizer.feature(aligned).reshape(-1).astype(np.float32)
    return l2_normalize(feat)


def sanitize_name(name: str) -> str:
    allowed = " _-"
    clean = "".join(c for c in name.strip() if c.isalnum() or c in allowed)
    return clean.strip()


def compute_centroids(embeddings: np.ndarray, names: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    unique_names = sorted(set(names.tolist()))
    centroid_list: List[np.ndarray] = []
    centroid_names: List[str] = []
    for person in unique_names:
        person_embs = embeddings[names == person]
        if len(person_embs) == 0:
            continue
        centroid = l2_normalize(np.mean(person_embs, axis=0))
        if centroid is None:
            continue
        centroid_list.append(centroid)
        centroid_names.append(person)
    if not centroid_list:
        raise SystemExit("Could not compute person centroids.")
    return np.vstack(centroid_list).astype(np.float32), np.array(centroid_names, dtype=str)


def save_db(
    db_path: Path,
    embeddings: np.ndarray,
    names: np.ndarray,
    centroids: np.ndarray,
    centroid_names: np.ndarray,
) -> None:
    np.savez_compressed(
        db_path,
        embeddings=embeddings.astype(np.float32),
        names=names.astype(str),
        centroids=centroids.astype(np.float32),
        centroid_names=centroid_names.astype(str),
    )
    print(f"[INFO] Database saved -> {db_path} ({len(names)} embeddings, {len(centroid_names)} identities)")


def load_db(db_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}. Run 'build' first.")
    data = np.load(db_path, allow_pickle=False)

    embeddings = data["embeddings"].astype(np.float32)
    names = data["names"].astype(str)
    if embeddings.ndim != 2 or len(embeddings) == 0 or len(names) == 0:
        raise SystemExit("Invalid or empty face database. Rebuild with 'build'.")

    if "centroids" in data and "centroid_names" in data:
        centroids = data["centroids"].astype(np.float32)
        centroid_names = data["centroid_names"].astype(str)
    else:
        centroids, centroid_names = compute_centroids(embeddings, names)

    return embeddings, names, centroids, centroid_names


def run_capture(args: argparse.Namespace) -> None:
    person_name = sanitize_name(args.name)
    if not person_name:
        raise SystemExit("Invalid name. Use letters/numbers/spaces/_/- only.")

    person_dir = KNOWN_FACES_DIR / person_name
    if args.replace and person_dir.exists():
        shutil.rmtree(person_dir)
    person_dir.mkdir(parents=True, exist_ok=True)

    check_opencv_requirements()
    ensure_models(auto_download=not args.no_download)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise SystemExit("Cannot open camera.")

    detector = create_detector((args.width, args.height), score_threshold=args.det_score)
    recognizer = create_recognizer()

    count = 0
    print(f"[INFO] Capturing {args.samples} samples for '{person_name}'")
    print("[INFO] Controls: SPACE=capture, q=quit")

    try:
        while count < args.samples:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue

            proc = normalize_lighting(frame) if args.light_normalize else frame
            faces = detect_faces(detector, proc)

            msg = "show one face only"
            ready = False
            primary_face = None

            if len(faces) == 1:
                primary_face = faces[0]
                ready, msg = face_quality_ok(primary_face, proc.shape, min_area_ratio=args.min_area_ratio)
            elif len(faces) > 1:
                msg = "only one face in frame"

            view = frame.copy()
            for f in faces[:3]:
                x1, y1, x2, y2 = bbox(f)
                color = (0, 200, 0) if f is primary_face and ready else (0, 0, 255)
                cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                view,
                f"{person_name}: {count}/{args.samples}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                view,
                f"status: {msg}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if ready else (0, 0, 255),
                2,
            )
            cv2.imshow("Face Capture", view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == 32 and ready and primary_face is not None:
                emb = embedding_from_face(proc, primary_face, recognizer)
                if emb is None:
                    print("[WARN] Bad face sample skipped.")
                    continue
                aligned = recognizer.alignCrop(proc, primary_face)
                out_path = person_dir / f"{count + 1:03d}.jpg"
                cv2.imwrite(str(out_path), aligned)
                count += 1
                print(f"[INFO] Saved {out_path}")
                time.sleep(0.15)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"[INFO] Done. Captured {count} samples.")


def run_build(args: argparse.Namespace) -> None:
    check_opencv_requirements()
    ensure_models(auto_download=not args.no_download)

    detector = create_detector((640, 480), score_threshold=args.det_score)
    recognizer = create_recognizer()

    if not KNOWN_FACES_DIR.exists():
        raise SystemExit(f"Missing folder: {KNOWN_FACES_DIR}")

    person_dirs = sorted(p for p in KNOWN_FACES_DIR.iterdir() if p.is_dir())
    if not person_dirs:
        raise SystemExit("No person folders found in known_faces/.")

    all_embeddings: List[np.ndarray] = []
    all_names: List[str] = []

    for person_dir in person_dirs:
        person = person_dir.name
        images = sorted(
            p
            for p in person_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        if not images:
            print(f"[WARN] No images for {person}, skipping.")
            continue

        added_embeddings = 0
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[WARN] Cannot read {img_path}")
                continue

            variants = lighting_variants(img) if args.augment_lighting else [img]
            per_image_added = 0

            for var in variants:
                faces = detect_faces(detector, var)
                if len(faces) != 1:
                    continue

                emb = embedding_from_face(var, faces[0], recognizer)
                if emb is None:
                    continue

                all_embeddings.append(emb)
                all_names.append(person)
                added_embeddings += 1
                per_image_added += 1

            if per_image_added == 0:
                print(f"[WARN] {img_path.name}: could not build embedding (skipped)")

        print(f"[INFO] {person}: {added_embeddings} embeddings added from {len(images)} images.")

    if not all_embeddings:
        raise SystemExit("No embeddings created. Capture clearer images and rebuild.")

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    names = np.array(all_names, dtype=str)
    centroids, centroid_names = compute_centroids(embeddings, names)
    save_db(args.db, embeddings, names, centroids, centroid_names)


def match_identity(
    emb: np.ndarray,
    db_embeddings: np.ndarray,
    db_names: np.ndarray,
    centroids: np.ndarray,
    centroid_names: np.ndarray,
    threshold: float,
    margin: float,
    centroid_threshold: float,
) -> Tuple[str, float, float]:
    sims = (db_embeddings @ emb).astype(np.float32)
    if sims.size == 0:
        return "Unknown", 0.0, 0.0

    unique_names, inverse = np.unique(db_names, return_inverse=True)
    best_per_identity = np.full(len(unique_names), -1.0, dtype=np.float32)
    np.maximum.at(best_per_identity, inverse, sims)

    best_identity_idx = int(np.argmax(best_per_identity))
    candidate_name = str(unique_names[best_identity_idx])
    best_score = float(best_per_identity[best_identity_idx])

    if len(best_per_identity) > 1:
        second = float(np.partition(best_per_identity, -2)[-2])
    else:
        # Only one enrolled identity: rely on threshold + centroid gate.
        second = -1.0

    if best_score < threshold or (best_score - second) < margin:
        return "Unknown", best_score, 0.0

    centroid_sims = centroids @ emb
    centroid_idx = int(np.argmax(centroid_sims))
    centroid_score = float(centroid_sims[centroid_idx])
    centroid_name = str(centroid_names[centroid_idx])

    if centroid_name != candidate_name or centroid_score < centroid_threshold:
        return "Unknown", best_score, centroid_score

    return candidate_name, best_score, centroid_score


def robot_action(name: str) -> None:
    # Map each recognized person to a dog behavior profile.
    # Replace this mapping and print with your real robot command channel.
    instruction_profiles = {
        "Harshhini": "FOLLOW_AND_LISTEN",
    }
    instruction = instruction_profiles.get(name, "IGNORE")
    print(f"[ROBOT] authorized={name} instruction={instruction}")


def run_recognize(args: argparse.Namespace) -> None:
    check_opencv_requirements()
    ensure_models(auto_download=not args.no_download)
    db_embeddings, db_names, centroids, centroid_names = load_db(args.db)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit("Cannot open camera.")

    detector = create_detector((args.width, args.height), score_threshold=args.det_score)
    recognizer = create_recognizer()

    history = deque(maxlen=args.history)
    last_trigger_times: Dict[str, float] = {}
    frame_id = 0
    active = True
    authorized_until_by_name: Dict[str, float] = {}

    print("[INFO] Recognition started.")
    print("[INFO] Controls: s=toggle recognize, q=quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            now = time.time()
            frame_id += 1
            view = frame.copy()
            status = "RECOGNIZING" if active else "PAUSED"
            recognized_names_in_frame: Set[str] = set()

            if active and frame_id % args.frame_skip == 0:
                proc = normalize_lighting(frame) if args.light_normalize else frame
                faces = detect_faces(detector, proc)
                faces = sorted(faces, key=lambda f: float(f[2] * f[3]), reverse=True)

                if faces and (not args.allow_multi_face) and len(faces) != 1:
                    status = "ONE FACE REQUIRED"
                    for face in faces[:3]:
                        x1, y1, x2, y2 = bbox(face)
                        cv2.rectangle(view, (x1, y1), (x2, y2), (0, 0, 255), 2)
                else:
                    for face in faces:
                        if face_area_ratio(face, proc.shape) < args.min_face_area_ratio:
                            continue

                        emb = embedding_from_face(proc, face, recognizer)
                        if emb is None:
                            continue

                        name, sample_score, centroid_score = match_identity(
                            emb,
                            db_embeddings,
                            db_names,
                            centroids,
                            centroid_names,
                            threshold=args.threshold,
                            margin=args.margin,
                            centroid_threshold=args.centroid_threshold,
                        )
                        if name != "Unknown":
                            recognized_names_in_frame.add(name)

                        x1, y1, x2, y2 = bbox(face)
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

                    history.append(recognized_names_in_frame)

                if len(history) == args.history:
                    vote_counts: Counter[str] = Counter()
                    for names_set in history:
                        vote_counts.update(names_set)

                    for person, count in vote_counts.items():
                        if count >= args.stable_count:
                            authorized_until_by_name[person] = now + args.track_timeout
                            last_time = last_trigger_times.get(person, 0.0)
                            if (now - last_time) > args.cooldown:
                                last_trigger_times[person] = now
                                robot_action(person)
            else:
                if not active:
                    history.clear()
                    authorized_until_by_name.clear()

            expired = [name for name, until in authorized_until_by_name.items() if now > until]
            for name in expired:
                authorized_until_by_name.pop(name, None)

            if active and authorized_until_by_name:
                active_names = sorted(authorized_until_by_name.keys())
                status = "AUTHORIZED: " + ", ".join(active_names[:3])

            color = (0, 255, 0) if active else (0, 0, 255)
            cv2.putText(view, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.imshow("Robot Face Recognition", view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                active = not active
                history.clear()
                print(f"[INFO] recognition active: {active}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_enroll(args: argparse.Namespace) -> None:
    run_capture(args)
    build_args = argparse.Namespace(
        db=args.db,
        det_score=args.det_score,
        no_download=args.no_download,
        augment_lighting=args.augment_lighting,
    )
    run_build(build_args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Face embedding system for Raspberry Pi robot dog")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--camera", type=int, default=0, help="Camera index")
    common.add_argument("--width", type=int, default=640, help="Capture width")
    common.add_argument("--height", type=int, default=480, help="Capture height")
    common.add_argument("--det-score", type=float, default=0.9, help="Face detector score threshold")
    common.add_argument("--no-download", action="store_true", help="Disable model auto-download")
    common.add_argument("--db", type=Path, default=DB_FILE, help="Embedding database file path")
    common.add_argument(
        "--no-light-normalize",
        action="store_true",
        help="Disable CLAHE/gamma light normalization",
    )

    p_capture = sub.add_parser("capture", parents=[common], help="Capture approved faces for one person")
    p_capture.add_argument("--name", required=True, help="Person name")
    p_capture.add_argument("--samples", type=int, default=25, help="Number of face samples")
    p_capture.add_argument("--replace", action="store_true", help="Delete existing samples for this person")
    p_capture.add_argument("--min-area-ratio", type=float, default=0.08, help="Minimum face area ratio")

    p_build = sub.add_parser("build", parents=[common], help="Build embedding database from known_faces/")
    p_build.add_argument(
        "--no-augment-lighting",
        action="store_true",
        help="Disable lighting augmentation while building embeddings",
    )

    p_rec = sub.add_parser("recognize", parents=[common], help="Run real-time recognition")
    p_rec.add_argument("--threshold", type=float, default=0.42, help="Sample cosine similarity threshold")
    p_rec.add_argument("--margin", type=float, default=0.06, help="Best-vs-second sample score margin")
    p_rec.add_argument("--centroid-threshold", type=float, default=0.40, help="Centroid cosine threshold")
    p_rec.add_argument("--frame-skip", type=int, default=1, help="Process every Nth frame")
    p_rec.add_argument("--history", type=int, default=6, help="Temporal smoothing history")
    p_rec.add_argument("--stable-count", type=int, default=4, help="Votes needed from history")
    p_rec.add_argument("--cooldown", type=float, default=3.0, help="Seconds between repeated triggers")
    p_rec.add_argument(
        "--track-timeout",
        type=float,
        default=2.0,
        help="Keep authorized identity active for this many seconds after last good match",
    )
    face_mode = p_rec.add_mutually_exclusive_group()
    face_mode.add_argument(
        "--allow-multi-face",
        dest="allow_multi_face",
        action="store_true",
        help="Recognize multiple faces in the same frame (default)",
    )
    face_mode.add_argument(
        "--single-face-only",
        dest="allow_multi_face",
        action="store_false",
        help="Require exactly one face in frame",
    )
    p_rec.add_argument("--min-face-area-ratio", type=float, default=0.05, help="Ignore very small faces")
    p_rec.set_defaults(allow_multi_face=True)

    p_enroll = sub.add_parser("enroll", parents=[common], help="Capture then rebuild database")
    p_enroll.add_argument("--name", required=True, help="Person name")
    p_enroll.add_argument("--samples", type=int, default=25, help="Number of face samples")
    p_enroll.add_argument("--replace", action="store_true", help="Delete existing samples for this person")
    p_enroll.add_argument("--min-area-ratio", type=float, default=0.08, help="Minimum face area ratio")
    p_enroll.add_argument(
        "--no-augment-lighting",
        action="store_true",
        help="Disable lighting augmentation while building embeddings",
    )

    args = parser.parse_args()
    args.light_normalize = not args.no_light_normalize
    if hasattr(args, "no_augment_lighting"):
        args.augment_lighting = not args.no_augment_lighting
    return args


def main() -> None:
    args = parse_args()
    if args.command == "capture":
        run_capture(args)
    elif args.command == "build":
        run_build(args)
    elif args.command == "recognize":
        run_recognize(args)
    elif args.command == "enroll":
        run_enroll(args)
    else:
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
