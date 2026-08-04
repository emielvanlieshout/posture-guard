"""macOS camera authorisation.

Asking OpenCV to open a camera you have no permission for gets you
``camera failed to properly initialize`` and five lines of backend noise, which
says nothing about the actual problem. AVFoundation will state the authorisation
status outright, and will raise the prompt and wait for an answer, so this module
asks it directly.

The wrinkle worth knowing: permission belongs to the application that *launched*
Python -- Terminal, iTerm, VS Code -- not to posture-guard, which macOS does not
consider a thing that exists. And a permission granted in System Settings does
not reach an already-running process, so the terminal has to be quit and
reopened. That trips up nearly everyone, so the guidance below always says it.

Everything degrades gracefully: on any platform without AVFoundation the status
is UNAVAILABLE and callers fall back to simply trying the camera.
"""

from __future__ import annotations

import os
import sys
import threading

NOT_DETERMINED = "not-determined"
RESTRICTED = "restricted"
DENIED = "denied"
AUTHORIZED = "authorized"
UNAVAILABLE = "unavailable"

# AVAuthorizationStatus, from AVCaptureDevice.h
_STATUS_BY_CODE = {0: NOT_DETERMINED, 1: RESTRICTED, 2: DENIED, 3: AUTHORIZED}

_TERMINAL_NAMES = {
    "Apple_Terminal": "Terminal",
    "iTerm.app": "iTerm",
    "vscode": "Visual Studio Code",
    "WarpTerminal": "Warp",
    "Hyper": "Hyper",
    "ghostty": "Ghostty",
    "kitty": "kitty",
    "alacritty": "Alacritty",
}


def responsible_app() -> str:
    """Best guess at which app macOS will attribute the camera request to."""
    term = os.environ.get("TERM_PROGRAM", "")
    return _TERMINAL_NAMES.get(term, term or "your terminal")


def _avcapture():
    """AVCaptureDevice and the video media type, or None where unavailable."""
    if sys.platform != "darwin":
        return None
    try:
        from AVFoundation import AVCaptureDevice, AVMediaTypeVideo  # noqa: PLC0415
    except ImportError:
        return None
    return AVCaptureDevice, AVMediaTypeVideo


def camera_status() -> str:
    """Current authorisation, without prompting."""
    handle = _avcapture()
    if handle is None:
        return UNAVAILABLE
    AVCaptureDevice, AVMediaTypeVideo = handle
    try:
        code = AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeVideo)
    except Exception:  # noqa: BLE001 - a diagnostic must never be the thing that crashes
        return UNAVAILABLE
    return _STATUS_BY_CODE.get(int(code), UNAVAILABLE)


def request_camera_access(timeout: float = 15.0) -> str:
    """Raise the system prompt if it has not been answered, and wait for it.

    Returns the resulting status. Only NOT_DETERMINED can be moved by this;
    once refused, macOS will not ask again and only System Settings will do.

    Pass ``timeout=0`` to raise the prompt and carry on without waiting. That
    matters inside an app bundle: the dialog is driven by the main run loop, so
    blocking the main thread on the answer can be the very thing that stops it
    appearing.
    """
    status = camera_status()
    if status != NOT_DETERMINED:
        return status

    handle = _avcapture()
    if handle is None:
        return UNAVAILABLE
    AVCaptureDevice, AVMediaTypeVideo = handle

    answered = threading.Event()

    def completion(granted) -> None:
        answered.set()

    try:
        AVCaptureDevice.requestAccessForMediaType_completionHandler_(
            AVMediaTypeVideo, completion
        )
    except Exception:  # noqa: BLE001
        return camera_status()

    answered.wait(timeout)
    # The completion flag is ignored on purpose: re-reading the status is the
    # authority, and it stays correct even if the callback never fires.
    return camera_status()


def describe(status: str) -> str:
    """What the user should do about this status. Empty when nothing is wrong."""
    app = responsible_app()
    if status == AUTHORIZED:
        return ""
    if status == NOT_DETERMINED:
        return (
            "macOS has not asked yet. Run `posture-guard doctor` and approve the "
            "prompt when it appears."
        )
    if status == DENIED:
        return (
            f"Camera access was refused. macOS attributes the request to {app}, not to "
            "posture-guard.\n"
            f"  1. System Settings > Privacy & Security > Camera, switch on {app}\n"
            f"  2. Quit {app} completely (Cmd-Q, not just the window) and reopen it\n"
            "Step 2 is not optional: a permission change does not reach a process that "
            "is already running."
        )
    if status == RESTRICTED:
        return (
            "Camera access is blocked by a policy on this Mac, usually Screen Time or "
            "a device-management profile. That has to be lifted before any app can use "
            "the camera."
        )
    return (
        "Could not read the camera permission. Install the AVFoundation bindings for a "
        "clearer answer:  pip install pyobjc-framework-AVFoundation"
    )


def list_video_devices() -> list[tuple[int, str]]:
    """Cameras AVFoundation can see, in the order OpenCV indexes them.

    The ordering matches in practice but is not contractual, so callers should
    present these as likely indices rather than certainties.
    """
    handle = _avcapture()
    if handle is None:
        return []
    AVCaptureDevice, AVMediaTypeVideo = handle
    try:
        devices = AVCaptureDevice.devicesWithMediaType_(AVMediaTypeVideo)
    except Exception:  # noqa: BLE001
        return []
    out = []
    for index, device in enumerate(devices or []):
        try:
            out.append((index, str(device.localizedName())))
        except Exception:  # noqa: BLE001
            out.append((index, "unknown camera"))
    return out
