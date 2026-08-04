"""Turn a pose into scale-invariant posture features, for a frontal or a side camera.

What a frontal webcam can and cannot see
----------------------------------------
The obvious idea is that protraction rotates the shoulders forward around the
spine, so the projected distance between the acromions shrinks while the head
keeps its apparent size. That reasoning omits perspective. Rotating forward also
brings the shoulders *closer to the lens*, which magnifies them. Projected width
is roughly ``2a·cos(theta) / (D - a·sin(theta))``: the cosine term is second
order in theta while the magnification is first order, so at small angles the
projection actually gets *wider*. It only starts shrinking past
``tan(theta) > a/D`` -- about 17 degrees at a 62 cm laptop distance.

The consequence, reproduced by tests/test_features.py against the synthetic
model: seen head-on from a laptop, a neutral posture and 35 degrees of
protraction produce almost the same shoulder-to-eye ratio. Frontally, isolated
protraction is close to invisible.

What a frontal camera does see is the rest of the slouch complex -- the chin
dropping, the head drifting forward, the shoulders riding up. For most people
those travel together with protraction, which is why the frontal feature set is
still worth having. It is measuring the company protraction keeps, not
protraction itself, and calibration reports exactly how well that stand-in
separates your two postures.

A side camera measures the real thing: the horizontal offset between the ear and
the acromion is monotonic and near-linear in protraction angle. That is also the
plane clinicians use for the sagittal shoulder angle.

Why the hips have to be in shot
-------------------------------
The ear is a treacherous reference. ``shoulder_ahead`` and ``neck_incline``
measure the acromion against the ear, and forward head posture moves the ear the
same way protraction moves the shoulder -- so the two partly cancel. Against the
synthetic model, 30 degrees of protraction reads 0.99 on ``shoulder_ahead`` when
the head stays put, 0.11 when the head drifts forward alongside it, and *minus*
0.80 when the head runs further forward than the shoulders. Sign inverted: the
feature now says "shoulders back" about someone whose shoulders came forward.

Anything referenced to the pelvis is immune, because the pelvis does not move
when you crane your neck. ``trunk_incline`` responds to protraction and ignores
head travel entirely; ``head_over_hip`` does the reverse. With both in play the
two motions are separable, and calibration can weight the one that matters.

So a side camera wants your hip in frame, and calibration says so when it is
missing. Without it, retracting your head alone will satisfy the profile while
your shoulders stay exactly where they were.

Every feature in both sets is a ratio or an angle, never a raw distance, so
leaning toward the screen cannot masquerade as a change in posture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .geometry import EPS, angle_from_horizontal_deg, angle_from_vertical_deg, dist, midpoint
from .landmarks import LM, PoseFrame

FRONTAL_FEATURES: tuple[str, ...] = (
    "shoulder_over_eye",  # biacromial width / interocular width
    "shoulder_over_face",  # biacromial width / ear-to-mouth height
    "neck_length",  # ear-to-shoulder drop / face height
    "head_pitch",  # nose below the ear line / face height
    "torso_incline",  # shoulder-to-hip tilt off vertical (needs hips)
    "shoulder_depth",  # metric shoulder-vs-hip depth (world landmarks)
)

SIDE_FEATURES: tuple[str, ...] = (
    "shoulder_ahead",  # acromion in front of the ear / face height
    "neck_incline",  # ear-to-shoulder segment off vertical, degrees
    "head_pitch",  # ear-to-nose line off horizontal, degrees
    "face_over_neck",  # face height / ear-to-shoulder distance
    "ear_shoulder_hip",  # angle at the shoulder, degrees (needs hips)
    "trunk_incline",  # shoulder-to-hip tilt off vertical (needs hips)
    "head_over_hip",  # ear in front of the hip / face height (needs hips)
)


@dataclass(frozen=True)
class QualityLimits:
    """Gates deciding whether a frame may be measured at all.

    A turned or tilted torso shortens a projected span exactly like posture
    does. Rather than correcting for that, offending frames are dropped; at 6 Hz
    there are plenty left.
    """

    min_visibility: float = 0.55
    min_hip_visibility: float = 0.60
    min_eye_width: float = 0.02  # image-height units; guards bogus detections
    min_face_height: float = 0.02

    # frontal only
    max_yaw: float = 0.30  # nose offset as a fraction of half the ear width
    max_roll_deg: float = 14.0

    # side only
    max_ear_over_face: float = 0.70  # forces a genuine side view (~73 deg or more)
    min_facing: float = 0.25  # nose must sit clearly ahead of the ears


@dataclass(frozen=True)
class FeatureSample:
    """Feature vector for one frame. Unavailable features are NaN, never zero."""

    ts: float
    values: np.ndarray
    ok: bool
    reason: str = ""
    pose_hint: float = math.nan  # yaw (frontal) or facing sign (side), for diagnostics

    @property
    def available(self) -> np.ndarray:
        return np.isfinite(self.values)


def _empty(ts: float, n: int, reason: str, hint: float = math.nan) -> FeatureSample:
    return FeatureSample(ts=ts, values=np.full(n, np.nan), ok=False, reason=reason, pose_hint=hint)


_HEAD_CORE = (
    LM.LEFT_EAR,
    LM.RIGHT_EAR,
    LM.MOUTH_LEFT,
    LM.MOUTH_RIGHT,
    LM.NOSE,
)


def _common(frame: PoseFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    ear_mid = midpoint(frame.p(LM.LEFT_EAR), frame.p(LM.RIGHT_EAR))
    mouth_mid = midpoint(frame.p(LM.MOUTH_LEFT), frame.p(LM.MOUTH_RIGHT))
    shoulder_mid = midpoint(frame.p(LM.LEFT_SHOULDER), frame.p(LM.RIGHT_SHOULDER))
    face_h = dist(ear_mid, mouth_mid)
    ear_w = dist(frame.p(LM.LEFT_EAR), frame.p(LM.RIGHT_EAR))
    return ear_mid, mouth_mid, shoulder_mid, face_h, ear_w


def extract_frontal(frame: PoseFrame, limits: QualityLimits | None = None) -> FeatureSample:
    """Features for a camera facing the subject, e.g. the built-in laptop webcam."""
    limits = limits or QualityLimits()
    n = len(FRONTAL_FEATURES)

    core = (*_HEAD_CORE, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER, LM.LEFT_EYE_OUTER, LM.RIGHT_EYE_OUTER)
    if frame.vis(*core) < limits.min_visibility:
        return _empty(frame.ts, n, "landmarks not visible enough")

    ear_mid, _, shoulder_mid, face_h, ear_w = _common(frame)
    l_sh, r_sh = frame.p(LM.LEFT_SHOULDER), frame.p(LM.RIGHT_SHOULDER)
    nose = frame.p(LM.NOSE)
    shoulder_w = dist(l_sh, r_sh)
    eye_w = dist(frame.p(LM.LEFT_EYE_OUTER), frame.p(LM.RIGHT_EYE_OUTER))

    if eye_w < limits.min_eye_width or face_h < limits.min_face_height or ear_w < EPS:
        return _empty(frame.ts, n, "face too small or degenerate")

    # Head turn as the nose's sideways offset relative to half the ear width:
    # ~0 looking straight at the lens, ~±1 in full profile.
    yaw = float((nose[0] - ear_mid[0]) / (ear_w / 2.0 + EPS))
    roll = angle_from_horizontal_deg(r_sh, l_sh)
    roll = (roll + 180.0) % 180.0
    roll = roll - 180.0 if roll > 90.0 else roll

    if abs(yaw) > limits.max_yaw:
        return _empty(frame.ts, n, "head turned away", yaw)
    if abs(roll) > limits.max_roll_deg:
        return _empty(frame.ts, n, "torso tilted sideways", yaw)

    values = np.full(n, np.nan)
    values[0] = shoulder_w / eye_w
    values[1] = shoulder_w / face_h
    # Positive while the shoulders sit below the ears, which is where they
    # belong. Shrugging or the head sinking into the shoulders both shrink it.
    values[2] = (shoulder_mid[1] - ear_mid[1]) / face_h
    values[3] = (nose[1] - ear_mid[1]) / face_h

    if frame.vis(LM.LEFT_HIP, LM.RIGHT_HIP) >= limits.min_hip_visibility:
        hip_mid = midpoint(frame.p(LM.LEFT_HIP), frame.p(LM.RIGHT_HIP))
        values[4] = angle_from_vertical_deg(hip_mid, shoulder_mid)
    values[5] = _shoulder_depth(frame, limits)

    return FeatureSample(ts=frame.ts, values=values, ok=True, pose_hint=yaw)


def extract_side(frame: PoseFrame, limits: QualityLimits | None = None) -> FeatureSample:
    """Features for a camera roughly perpendicular to the subject.

    Left and right landmarks nearly coincide in profile, so midpoints are used
    throughout: that sidesteps having to decide which side faces the lens and
    averages away mediapipe's guess about the occluded half.
    """
    limits = limits or QualityLimits()
    n = len(SIDE_FEATURES)

    core = (*_HEAD_CORE, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER)
    if frame.vis(*core) < limits.min_visibility:
        return _empty(frame.ts, n, "landmarks not visible enough")

    ear_mid, _, shoulder_mid, face_h, ear_w = _common(frame)
    nose = frame.p(LM.NOSE)

    if face_h < limits.min_face_height:
        return _empty(frame.ts, n, "face too small")
    if ear_w / face_h > limits.max_ear_over_face:
        return _empty(frame.ts, n, "not a side view; both ears still visible")

    # Which way the subject faces, from the nose sitting ahead of the ear line.
    # Every signed feature is multiplied by this, so the numbers mean the same
    # thing whether the camera stands to the left or to the right.
    facing_raw = (nose[0] - ear_mid[0]) / face_h
    if abs(facing_raw) < limits.min_facing:
        return _empty(frame.ts, n, "facing direction ambiguous", facing_raw)
    facing = math.copysign(1.0, facing_raw)

    neck = shoulder_mid - ear_mid
    neck_len = float(np.linalg.norm(neck))
    if neck_len < EPS:
        return _empty(frame.ts, n, "degenerate neck segment", facing)

    values = np.full(n, np.nan)
    # The measurement that matters: how far the acromion sits in front of the
    # ear, in units of face height. Rises monotonically with protraction.
    values[0] = facing * neck[0] / face_h
    values[1] = math.degrees(math.atan2(facing * neck[0], neck[1] + EPS))
    values[2] = facing * angle_from_horizontal_deg(ear_mid, nose)
    values[3] = face_h / neck_len

    if frame.vis(LM.LEFT_HIP, LM.RIGHT_HIP) >= limits.min_hip_visibility:
        hip_mid = midpoint(frame.p(LM.LEFT_HIP), frame.p(LM.RIGHT_HIP))
        v1 = ear_mid - shoulder_mid
        v2 = hip_mid - shoulder_mid
        denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
        if denom > EPS:
            cos = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
            values[4] = math.degrees(math.acos(cos))
        # Referenced to the pelvis, which stays put while you crane your neck.
        # These two are what separate protraction from forward head posture.
        trunk = shoulder_mid - hip_mid
        values[5] = math.degrees(math.atan2(facing * trunk[0], -trunk[1] + EPS))
        values[6] = facing * (ear_mid[0] - hip_mid[0]) / face_h

    return FeatureSample(ts=frame.ts, values=values, ok=True, pose_hint=facing)


def _shoulder_depth(frame: PoseFrame, limits: QualityLimits) -> float:
    """How far the shoulders sit in front of the hips, from metric landmarks.

    In a laptop framing the hips are off-screen and mediapipe extrapolates them,
    so this is frequently noise. It is emitted regardless and left to
    calibration: a feature that fails to separate the two postures ends up
    weighted zero.
    """
    if frame.world is None:
        return math.nan
    if frame.vis(LM.LEFT_HIP, LM.RIGHT_HIP) < limits.min_hip_visibility:
        return math.nan

    w = frame.world
    sh = (w[int(LM.LEFT_SHOULDER)] + w[int(LM.RIGHT_SHOULDER)]) / 2.0
    hip = (w[int(LM.LEFT_HIP)] + w[int(LM.RIGHT_HIP)]) / 2.0
    scale = float(np.linalg.norm(w[int(LM.LEFT_SHOULDER)] - w[int(LM.RIGHT_SHOULDER)]))
    if scale < EPS:
        return math.nan
    # mediapipe world z decreases toward the camera; negate so the feature grows
    # as the shoulders travel *back*, matching the other features' direction.
    return float(-(sh[2] - hip[2]) / scale)


@dataclass(frozen=True)
class FeatureSet:
    view: str
    names: tuple[str, ...]
    extract: Callable[..., FeatureSample]
    #: Features that measure protraction itself rather than something adjacent.
    primary: tuple[int, ...]
    #: Features that go NaN when the hips are out of frame.
    hip_dependent: tuple[int, ...]
    #: Multiplier applied before calibration normalises the weights. Separation
    #: measures how well a feature tells your two poses apart; it cannot tell
    #: whether the feature means what its name says. This is where the geometry
    #: gets a vote -- see the module docstring on ear-referenced features.
    prior: tuple[float, ...]
    blurb: str

    @property
    def n(self) -> int:
        return len(self.names)


FEATURE_SETS: dict[str, FeatureSet] = {
    "side": FeatureSet(
        view="side",
        names=SIDE_FEATURES,
        extract=extract_side,
        # trunk_incline alone: the ear-referenced pair conflates protraction
        # with forward head posture and can invert outright. See the module
        # docstring and tests/test_features.py::TestProtractionVersusHead.
        primary=(5,),
        hip_dependent=(4, 5, 6),
        # Halved for the three ear-referenced features, because forward head
        # posture moves their reference point. They stay in -- with the hips out
        # of frame they are all there is, and normalisation hands them the full
        # weight again once the pelvis-referenced ones drop out.
        prior=(0.5, 0.5, 1.0, 0.5, 1.0, 1.0, 1.0),
        blurb="camera to your side; measures protraction directly, hips in frame",
    ),
    "frontal": FeatureSet(
        view="frontal",
        names=FRONTAL_FEATURES,
        extract=extract_frontal,
        # Only shoulder_over_eye depends on shoulder width alone. Everything else
        # in this set is confounded by the head, which is why a frontal profile
        # always comes with a caveat -- see calibration.verdict.
        primary=(0,),
        hip_dependent=(4, 5),
        # Nothing to prefer: head-on there is no unambiguous alternative to fall
        # back on, so separation decides on its own.
        prior=(1.0,) * len(FRONTAL_FEATURES),
        blurb="built-in webcam; measures the slouch complex, not protraction itself",
    ),
}


def get_feature_set(view: str) -> FeatureSet:
    try:
        return FEATURE_SETS[view]
    except KeyError:
        raise ValueError(f"unknown view {view!r}; expected one of {sorted(FEATURE_SETS)}") from None
