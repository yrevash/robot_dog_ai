"""
Face detection + recognition wrapper.

Reuses `src/face_embedding.py` from the parent repo so we never duplicate the
two-gate identity logic. Only adds a temporal voting helper on top.
"""
from __future__ import annotations

import sys
from collections import Counter, deque
from pathlib import Path
from typing import Deque, Optional, Tuple

import cv2
import numpy as np

# Make the parent repo's `src/` importable (pi_deploy lives at repo_root/pi_deploy).
# src/ has no __init__.py, so add it directly to sys.path and import by module name.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import face_embedding as fe  # noqa: E402

from ..config import settings


class FaceRecognizer:
    def __init__(self):
        fe.check_opencv_requirements()
        fe.ensure_models(auto_download=True)

        self.detector = fe.create_detector(
            (settings.frame_width, settings.frame_height), score_threshold=0.9
        )
        self.recognizer = fe.create_recognizer()

        self.db_embeddings, self.db_names, self.centroids, self.centroid_names = fe.load_db(
            settings.face_db_file
        )

        self._history: Deque[str] = deque(maxlen=settings.vote_history)

    def process(self, frame: np.ndarray) -> Tuple[Optional[str], Optional[Tuple[int, int, int, int]]]:
        """Return (authorized_name_or_None, bbox_or_None) for the largest face."""
        faces = fe.detect_faces(self.detector, frame)
        if not faces:
            self._history.append("__none__")
            return None, None

        face = faces[0]
        ok, _ = fe.face_quality_ok(face, frame.shape)
        box = fe.bbox(face)
        if not ok:
            self._history.append("__low_q__")
            return None, box

        emb = fe.embedding_from_face(frame, face, self.recognizer)
        if emb is None:
            self._history.append("__no_emb__")
            return None, box

        name, _best, _cent = fe.match_identity(
            emb,
            self.db_embeddings,
            self.db_names,
            self.centroids,
            self.centroid_names,
            threshold=settings.recog_threshold,
            margin=settings.recog_margin,
            centroid_threshold=settings.recog_centroid_threshold,
        )
        self._history.append(name)

        # Vote: require `vote_required` matching identities in the last `vote_history` frames.
        counts = Counter(self._history)
        top_name, top_count = counts.most_common(1)[0]
        if top_name not in ("Unknown", "__none__", "__low_q__", "__no_emb__") \
                and top_count >= settings.vote_required:
            return top_name, box

        return None, box

    def reset(self):
        self._history.clear()
