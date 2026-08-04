"""posture-guard: a local webcam posture monitor aimed at shoulder protraction.

Nothing here reaches the network at runtime. The pose model is downloaded once
during `posture-guard setup`; after that the camera feed is processed in this
process, reduced to a handful of numbers, and the image is discarded.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
