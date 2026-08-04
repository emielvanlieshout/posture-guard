"""A parametric torso that produces landmarks, so the pipeline can be tested without a camera.

Shoulders are modelled as two points on a circle around the spine. Protraction
rotates both of them forward by an angle; everything the frontal camera sees --
the shoulders narrowing, the head keeping its size -- then follows from ordinary
perspective projection rather than from a hand-tuned fudge factor.

The couplings (shoulders also rise a little, the head drifts forward, the chin
drops) are what a slouch actually looks like, and they are what gives the
companion features something to latch onto.

This is a geometric stand-in, not a biomechanical model. It is good enough to
prove that the metric responds to protraction, is invariant to camera distance,
and rejects a turned torso -- which is exactly what the tests assert.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .geometry import rot_x, rot_y, rot_z
from .landmarks import LM, N_LANDMARKS, PoseFrame


@dataclass(frozen=True)
class Body:
    """Segment lengths in metres, roughly a 50th-percentile adult."""

    shoulder_half: float = 0.190  # half the biacromial width
    ear_half: float = 0.078
    eye_half: float = 0.045
    mouth_half: float = 0.026
    head_height: float = 0.135  # C7 up to the ear line
    face_depth: float = 0.075  # ear line forward to the eye plane
    nose_depth: float = 0.105
    mouth_depth: float = 0.085
    eye_rise: float = 0.018
    nose_drop: float = 0.012
    mouth_drop: float = 0.058
    torso_length: float = 0.500  # C7 down to the hip centre
    hip_half: float = 0.130


@dataclass(frozen=True)
class Posture:
    protraction_deg: float = 0.0
    rise_coupling: float = 0.25  # shoulder elevation per unit forward travel
    forward_coupling: float = 0.55  # head translation per unit forward travel
    pitch_coupling: float = 0.35  # degrees of chin drop per degree of protraction
    torso_lean_deg: float = 0.0
    yaw_deg: float = 0.0
    roll_deg: float = 0.0


@dataclass(frozen=True)
class Camera:
    distance: float = 0.62  # metres from the camera to the neutral C7 plane
    focal: float = 1.0
    height: float = 0.10  # camera above C7, i.e. perched on the screen bezel


def _body_points(body: Body, posture: Posture) -> dict[int, np.ndarray]:
    """Landmark positions in a right-handed frame: x left, y up, z toward the camera."""
    theta = math.radians(posture.protraction_deg)
    a = body.shoulder_half
    forward = a * math.sin(theta)

    l_sh = np.array([a * math.cos(theta), posture.rise_coupling * forward, forward])
    r_sh = np.array([-a * math.cos(theta), posture.rise_coupling * forward, forward])

    head_c = np.array([0.0, body.head_height, posture.forward_coupling * forward])

    head_local = {
        int(LM.LEFT_EAR): np.array([body.ear_half, 0.0, 0.0]),
        int(LM.RIGHT_EAR): np.array([-body.ear_half, 0.0, 0.0]),
        int(LM.LEFT_EYE_OUTER): np.array([body.eye_half, body.eye_rise, body.face_depth]),
        int(LM.RIGHT_EYE_OUTER): np.array([-body.eye_half, body.eye_rise, body.face_depth]),
        int(LM.NOSE): np.array([0.0, -body.nose_drop, body.nose_depth]),
        int(LM.MOUTH_LEFT): np.array([body.mouth_half, -body.mouth_drop, body.mouth_depth]),
        int(LM.MOUTH_RIGHT): np.array([-body.mouth_half, -body.mouth_drop, body.mouth_depth]),
    }
    # Chin drops as the shoulders roll forward; rotate the head about its own
    # ear-line axis so the ears stay put and the face swings down.
    pitch = rot_x(math.radians(-posture.pitch_coupling * posture.protraction_deg))

    points = {int(LM.LEFT_SHOULDER): l_sh, int(LM.RIGHT_SHOULDER): r_sh}
    for idx, local in head_local.items():
        points[idx] = head_c + pitch @ local

    hip_y = -body.torso_length
    points[int(LM.LEFT_HIP)] = np.array([body.hip_half, hip_y, 0.0])
    points[int(LM.RIGHT_HIP)] = np.array([-body.hip_half, hip_y, 0.0])
    return points


def _apply_global(
    points: dict[int, np.ndarray], body: Body, posture: Posture
) -> dict[int, np.ndarray]:
    hip_centre = np.array([0.0, -body.torso_length, 0.0])
    lean = rot_x(math.radians(posture.torso_lean_deg))
    hips = {int(LM.LEFT_HIP), int(LM.RIGHT_HIP)}
    leaned = {
        i: (hip_centre + lean @ (p - hip_centre)) if i not in hips else p
        for i, p in points.items()
    }
    turn = rot_z(math.radians(posture.roll_deg)) @ rot_y(math.radians(posture.yaw_deg))
    return {i: hip_centre + turn @ (p - hip_centre) for i, p in leaned.items()}


def synth_frame(
    ts: float = 0.0,
    *,
    body: Body | None = None,
    posture: Posture | None = None,
    camera: Camera | None = None,
    aspect: float = 4 / 3,
    hip_visibility: float = 0.25,
    noise: float = 0.0,
    world_noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> PoseFrame:
    """Render one synthetic :class:`PoseFrame`.

    ``hip_visibility`` defaults low because a laptop webcam frames head and
    shoulders only; the hip-dependent features are then gated off, exactly as
    they will be in real use.
    """
    body = body or Body()
    posture = posture or Posture()
    camera = camera or Camera()
    rng = rng or np.random.default_rng(0)

    points = _apply_global(_body_points(body, posture), body, posture)

    xy = np.zeros((N_LANDMARKS, 2), float)
    visibility = np.zeros(N_LANDMARKS, float)
    world = np.zeros((N_LANDMARKS, 3), float)

    hip_mid = (points[int(LM.LEFT_HIP)] + points[int(LM.RIGHT_HIP)]) / 2.0

    for idx, p in points.items():
        depth = camera.distance - p[2]
        if depth <= 0.05:
            raise ValueError("camera is inside the subject; increase Camera.distance")
        u = camera.focal * p[0] / depth
        v = camera.focal * (camera.height - p[1]) / depth
        xy[idx] = (0.5 * aspect + u, 0.5 + v)
        rel = p - hip_mid
        # mediapipe world axes: x right, y down, z decreasing toward the camera.
        world[idx] = (-rel[0], -rel[1], -rel[2])
        visibility[idx] = 1.0

    for hip in (LM.LEFT_HIP, LM.RIGHT_HIP):
        visibility[int(hip)] = hip_visibility

    if noise:
        xy += rng.normal(0.0, noise, xy.shape)
    if world_noise:
        world += rng.normal(0.0, world_noise, world.shape)

    return PoseFrame(ts=ts, xy=xy, visibility=visibility, world=world)


def synth_series(
    protraction_deg: float,
    *,
    n: int = 60,
    start_ts: float = 0.0,
    period: float = 1 / 6,
    jitter_deg: float = 1.2,
    noise: float = 0.0015,
    world_noise: float = 0.006,
    seed: int = 0,
    **frame_kwargs,
) -> list[PoseFrame]:
    """A run of frames held at one posture, with the wobble a real person has."""
    rng = np.random.default_rng(seed)
    base = frame_kwargs.pop("posture", Posture())
    frames = []
    for i in range(n):
        posture = replace(
            base,
            protraction_deg=protraction_deg + float(rng.normal(0.0, jitter_deg)),
            yaw_deg=base.yaw_deg + float(rng.normal(0.0, 2.0)),
        )
        frames.append(
            synth_frame(
                ts=start_ts + i * period,
                posture=posture,
                noise=noise,
                world_noise=world_noise,
                rng=rng,
                **frame_kwargs,
            )
        )
    return frames
