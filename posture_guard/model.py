"""Fetching the pose model.

This is the only code in the project that opens a network connection, it runs
once during `posture-guard setup`, and it downloads a model file. Nothing from
the camera is ever uploaded -- there is no code path that could.

Certificates are the wrinkle. A Python installed from python.org, or via pyenv
or conda, does not use the macOS keychain; it looks for a CA bundle that a fresh
install does not have, and every HTTPS request fails with
CERTIFICATE_VERIFY_FAILED until someone runs Install Certificates.command. So
the bundle from ``certifi`` is used explicitly rather than left to the default,
which works regardless of how Python got onto the machine.
"""

from __future__ import annotations

import hashlib
import shutil
import ssl
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
MIN_BYTES = 1_000_000


class ModelDownloadError(RuntimeError):
    """Download failed, with a message that says what to do about it."""


def ssl_context() -> ssl.SSLContext:
    """A context that can verify certificates on any Python install."""
    try:
        import certifi  # noqa: PLC0415 - optional, and only needed here
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def manual_download_hint(path: Path, url: str = MODEL_URL) -> str:
    """Fallback instructions. curl uses the system trust store, so it tends to work."""
    return f'curl -L --create-dirs -o "{path}" "{url}"'


def ensure_model(path: Path, *, force: bool = False, url: str = MODEL_URL) -> Path:
    """Download the pose landmarker bundle unless it is already there."""
    path = Path(path)
    if path.exists() and path.stat().st_size >= MIN_BYTES and not force:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix=".part") as tmp:
        tmp_path = Path(tmp.name)
    try:
        try:
            with urllib.request.urlopen(  # noqa: S310 - fixed https URL
                url, timeout=60, context=ssl_context()
            ) as response:
                if getattr(response, "status", 200) != 200:
                    raise ModelDownloadError(f"server answered with status {response.status}")
                with tmp_path.open("wb") as out:
                    shutil.copyfileobj(response, out)
        except urllib.error.URLError as exc:
            raise ModelDownloadError(_explain(exc, path, url)) from exc

        size = tmp_path.stat().st_size
        if size < MIN_BYTES:
            raise ModelDownloadError(
                f"the download stopped after {size} bytes, which is far short of the "
                f"~5.8 MB model. Try again, or fetch it by hand:\n"
                f"  {manual_download_hint(path, url)}"
            )
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return path


def _reason_text(reason: object) -> str:
    """Readable text for an exception whose repr is a bare args tuple."""
    args = getattr(reason, "args", ())
    if args and isinstance(args[-1], str):
        return args[-1]
    return str(reason)


def _explain(exc: urllib.error.URLError, path: Path, url: str) -> str:
    reason = getattr(exc, "reason", exc)
    text = _reason_text(reason)
    if isinstance(reason, ssl.SSLError) or "CERTIFICATE_VERIFY_FAILED" in text:
        return (
            f"the connection could not be verified ({text}).\n"
            "This Python has no CA bundle it can use. Either install one:\n"
            "  pip install --upgrade certifi\n"
            'and on a python.org build also run "Install Certificates.command" from\n'
            "  /Applications/Python 3.x/\n"
            "or sidestep it entirely and fetch the model with curl, which uses the\n"
            "system trust store:\n"
            f"  {manual_download_hint(path, url)}"
        )
    return (
        f"could not reach the model host ({text}).\n"
        f"If you are behind a proxy or offline, fetch it by hand instead:\n"
        f"  {manual_download_hint(path, url)}"
    )


def model_digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
