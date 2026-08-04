"""Fetching the pose model.

This is the only code in the project that opens a network connection, it runs
once during `posture-guard setup`, and it downloads a model file. Nothing from
the camera is ever uploaded -- there is no code path that could.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
MIN_BYTES = 1_000_000


def ensure_model(path: Path, *, force: bool = False, url: str = MODEL_URL) -> Path:
    """Download the pose landmarker bundle unless it is already there."""
    path = Path(path)
    if path.exists() and path.stat().st_size >= MIN_BYTES and not force:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix=".part") as tmp:
        tmp_path = Path(tmp.name)
    try:
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - fixed https URL
            if getattr(response, "status", 200) != 200:
                raise RuntimeError(f"download failed with status {response.status}")
            with tmp_path.open("wb") as out:
                shutil.copyfileobj(response, out)
        if tmp_path.stat().st_size < MIN_BYTES:
            raise RuntimeError(f"downloaded file is only {tmp_path.stat().st_size} bytes")
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return path


def model_digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
