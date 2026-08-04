"""Camera and pose detection: the only module that ever holds an image.

A frame is read, handed to mediapipe, converted into landmark coordinates and
then goes out of scope. Nothing is written to disk, kept in a buffer or passed
on; every consumer downstream sees numbers only.

Imports of cv2 and mediapipe are deferred to construction time so that the rest
of the package -- and its whole test suite -- stays importable on a machine with
neither installed.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Iterator

# OpenCV's AVFoundation backend asks macOS for camera access itself, and to show
# that dialog it has to spin the main run loop. Capture runs on a worker thread,
# where it cannot, so it gives up with "can not spin main run loop from other
# thread" and the camera never opens. Authorisation is ours to handle -- see
# permissions.request_camera_access, which does it on the main thread and waits.
os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")

import numpy as np

from .landmarks import N_LANDMARKS, PoseFrame


class CameraError(RuntimeError):
    pass


class PoseSource:
    """Opens a camera and yields :class:`PoseFrame` objects at a target rate."""

    def __init__(
        self,
        model_path,
        *,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: float = 6.0,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        keep_frame: bool = False,
    ):
        self.model_path = str(model_path)
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.period = 1.0 / max(fps, 0.1)
        self._min_detection = min_detection_confidence
        self._min_tracking = min_tracking_confidence
        # Opt-in, and only `posture-guard preview` turns it on: that command
        # exists to help you aim the camera, which needs a picture. Monitoring
        # never sets it, and the image is never written anywhere.
        self.keep_frame = keep_frame
        self.last_frame = None
        self._cap = None
        self._landmarker = None
        self._last_ms = -1

    def open(self) -> "PoseSource":
        import cv2  # noqa: PLC0415 - deferred on purpose
        from mediapipe.tasks.python import BaseOptions  # noqa: PLC0415
        from mediapipe.tasks.python import vision  # noqa: PLC0415

        backend = getattr(cv2, "CAP_AVFOUNDATION", 0)
        cap = cv2.VideoCapture(self.camera_index, backend) if backend else cv2.VideoCapture(
            self.camera_index
        )
        if not cap.isOpened():
            cap.release()
            raise CameraError(
                f"could not open camera {self.camera_index}. "
                "On macOS the app asking for the camera is the terminal or IDE you "
                "launched from: check System Settings > Privacy & Security > Camera. "
                "Run `posture-guard doctor` to list the cameras that do open."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:  # noqa: BLE001 - unsupported on some backends, harmless
            pass
        self._cap = cap

        self._landmarker = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.model_path),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=self._min_detection,
                min_tracking_confidence=self._min_tracking,
                output_segmentation_masks=False,
            )
        )
        return self

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> "PoseSource":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    def read(self) -> PoseFrame | None:
        """One detection, or None when nobody is in view."""
        import cv2  # noqa: PLC0415
        import mediapipe as mp  # noqa: PLC0415

        if self._cap is None or self._landmarker is None:
            raise CameraError("PoseSource used before open()")

        # Drop whatever the driver queued while we were asleep, so the pose we
        # score is the pose right now rather than a second ago.
        self._cap.grab()
        ok, bgr = self._cap.retrieve()
        if not ok or bgr is None:
            ok, bgr = self._cap.read()
        if not ok or bgr is None:
            raise CameraError("camera stopped delivering frames")

        ts = time.time()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # mediapipe's VIDEO mode demands strictly increasing timestamps.
        ms = max(int(ts * 1000), self._last_ms + 1)
        self._last_ms = ms
        result = self._landmarker.detect_for_video(image, ms)

        height, width = rgb.shape[:2]
        self.last_frame = bgr if self.keep_frame else None
        del bgr, rgb, image  # nothing here outlives the call
        return _to_pose_frame(result, ts, width, height)

    def frames(self) -> Iterator[PoseFrame | None]:
        """Paced stream of detections. Yields None while nobody is in view."""
        next_at = time.monotonic()
        while True:
            now = time.monotonic()
            if now < next_at:
                time.sleep(next_at - now)
            next_at = max(time.monotonic(), next_at + self.period)
            yield self.read()


def _to_pose_frame(result, ts: float, width: int, height: int) -> PoseFrame | None:
    landmarks = getattr(result, "pose_landmarks", None)
    if not landmarks:
        return None
    points = landmarks[0]
    if len(points) < N_LANDMARKS:
        return None

    aspect = width / height if height else 1.0
    xy = np.empty((N_LANDMARKS, 2), float)
    visibility = np.empty(N_LANDMARKS, float)
    for i in range(N_LANDMARKS):
        lm = points[i]
        # x is scaled by the aspect ratio so both axes end up in units of image
        # height; without this every ratio would depend on the resolution.
        xy[i] = (lm.x * aspect, lm.y)
        visibility[i] = max(getattr(lm, "visibility", 1.0) or 0.0,
                            getattr(lm, "presence", 0.0) or 0.0)

    world = None
    world_landmarks = getattr(result, "pose_world_landmarks", None)
    if world_landmarks:
        wl = world_landmarks[0]
        if len(wl) >= N_LANDMARKS:
            world = np.array([[wl[i].x, wl[i].y, wl[i].z] for i in range(N_LANDMARKS)], float)

    return PoseFrame(ts=ts, xy=xy, visibility=visibility, world=world)


def list_cameras(limit: int = 5, indices: list[int] | None = None) -> list[int]:
    """Indices that actually open. Used by `posture-guard doctor`.

    Pass ``indices`` when the platform can already enumerate its cameras.
    Probing a blind range past the last real device just prints "out device of
    bound" once per miss, which reads like a fault and is not one.

    Skipped entirely when macOS has refused access: probing would emit a backend
    error per index and still tell you nothing you did not already know.
    """
    from .permissions import AUTHORIZED, UNAVAILABLE, camera_status  # noqa: PLC0415

    status = camera_status()
    if status not in (AUTHORIZED, UNAVAILABLE):
        return []

    import cv2  # noqa: PLC0415

    found = []
    # An index that does not answer is the expected outcome here, not a fault,
    # but OpenCV logs each one as an error and a backtrace. Muted for the probe
    # only; capture itself keeps the normal log level so real faults still speak.
    with _quiet_opencv():
        for index in indices if indices is not None else range(limit):
            cap = cv2.VideoCapture(index)
            try:
                if cap.isOpened() and cap.read()[0]:
                    found.append(index)
            finally:
                cap.release()
    return found


@contextmanager
def _quiet_opencv():
    try:
        from cv2.utils import logging as cv_logging  # noqa: PLC0415
    except ImportError:
        yield
        return

    previous = cv_logging.getLogLevel()
    cv_logging.setLogLevel(cv_logging.LOG_LEVEL_SILENT)
    try:
        yield
    finally:
        cv_logging.setLogLevel(previous)
