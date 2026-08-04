"""Landmark indices and the frame container the rest of the pipeline speaks.

Deliberately free of any mediapipe import: everything downstream of this module
can be exercised without a camera, a model file, or a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

N_LANDMARKS = 33


class LM(IntEnum):
    """Subset of the 33 BlazePose landmarks this project actually uses."""

    NOSE = 0
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24


@dataclass(frozen=True)
class PoseFrame:
    """One detected pose.

    ``xy`` is *aspect corrected*: x is scaled by the image aspect ratio so that
    both axes are expressed in units of image height. Without that correction a
    horizontal distance and a vertical distance are not comparable and every
    ratio in features.py would silently depend on the camera resolution.

    ``world`` holds mediapipe's metric landmarks (metres, origin at the hip
    midpoint) or None when unavailable.

    No pixel data is carried here. Frames are converted and dropped inside the
    capture loop; nothing downstream can accidentally retain an image.
    """

    ts: float
    xy: np.ndarray  # (33, 2) float
    visibility: np.ndarray  # (33,) float in [0, 1]
    world: np.ndarray | None = None  # (33, 3) float, metres

    def __post_init__(self) -> None:
        if self.xy.shape != (N_LANDMARKS, 2):
            raise ValueError(f"xy must be ({N_LANDMARKS}, 2), got {self.xy.shape}")
        if self.visibility.shape != (N_LANDMARKS,):
            raise ValueError(
                f"visibility must be ({N_LANDMARKS},), got {self.visibility.shape}"
            )
        if self.world is not None and self.world.shape != (N_LANDMARKS, 3):
            raise ValueError(f"world must be ({N_LANDMARKS}, 3), got {self.world.shape}")

    def p(self, index: LM | int) -> np.ndarray:
        return self.xy[int(index)]

    def vis(self, *indices: LM | int) -> float:
        """Lowest visibility across the given landmarks."""
        return float(min(self.visibility[int(i)] for i in indices))
