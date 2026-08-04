"""Calibration profile and the score derived from it.

The score is anchored on two postures you demonstrate yourself: the one you want
(shoulders back and down) and the one you fall into. 0 means you are sitting the
way you showed as good, 1 means you are back in your habitual slouch. That beats
a fixed threshold because it needs no assumption about your build, your desk, or
where the camera sits.

Feature weights are not hand-tuned. Each feature is scored on how far apart it
puts the two calibration postures relative to its own noise -- essentially d'.
Features that fail to separate them get weight zero and drop out. That is what
lets the same code serve a frontal and a side camera without either one needing
its own set of magic numbers.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .geometry import median_abs_deviation

# Bumped whenever a feature set changes shape. An old profile's weights would
# line up against the wrong features, so it is refused rather than reinterpreted.
PROFILE_VERSION = 2

# A feature must separate the two postures by at least this many noise widths
# before it is allowed to contribute at all.
MIN_SEPARATION = 1.0
# Above this, extra discriminative power stops buying extra weight, so one loud
# feature cannot drown out the rest.
MAX_SEPARATION = 5.0
# A feature needs a finite reading in at least this share of calibration frames.
MIN_COVERAGE = 0.5
# A live frame needs this share of the profile's total weight to be scoreable.
MIN_LIVE_WEIGHT = 0.4
# Features never agree perfectly, so disagreement is only treated as evidence of
# an off-axis posture once it exceeds what was seen during calibration -- with a
# floor, because two held poses understate the disagreement of ordinary sitting.
MIN_DISAGREEMENT_REF = 0.35
# On-axis postures disagree by 0.05 to 0.16 against the synthetic model, so the
# floor above already carries better than twice the headroom it needs. The gain
# only ever multiplies what is left after subtracting the reference, so raising
# it sharpens genuinely off-axis postures without moving on-axis ones at all.
DISAGREEMENT_GAIN = 1.5


class CalibrationError(RuntimeError):
    """Raised when the two demonstrated postures cannot be told apart."""


@dataclass
class Profile:
    view: str
    names: tuple[str, ...]
    good_med: np.ndarray
    good_mad: np.ndarray
    bad_med: np.ndarray
    bad_mad: np.ndarray
    separation: np.ndarray
    weights: np.ndarray
    n_good: int
    n_bad: int
    #: How much the features disagreed with each other during calibration.
    #: Anything beyond this means the posture is not on the line between your two
    #: demonstrated poses -- see :func:`score`.
    disagreement_ref: float = MIN_DISAGREEMENT_REF
    created: float = field(default_factory=time.time)
    version: int = PROFILE_VERSION

    @property
    def delta(self) -> np.ndarray:
        return self.good_med - self.bad_med

    @property
    def usable(self) -> np.ndarray:
        return self.weights > 0

    def describe(self) -> str:
        lines = [
            f"view: {self.view}   frames: {self.n_good} good / {self.n_bad} slouched",
            f"{'feature':<20}{'good':>10}{'slouch':>10}{'separation':>12}{'weight':>9}",
        ]
        order = np.argsort(-self.weights)
        for i in order:
            sep = self.separation[i]
            lines.append(
                f"{self.names[i]:<20}"
                f"{self.good_med[i]:>10.3f}"
                f"{self.bad_med[i]:>10.3f}"
                f"{'  n/a' if not np.isfinite(sep) else f'{sep:>12.2f}'}"
                f"{self.weights[i]:>9.2f}"
            )
        return "\n".join(lines)

    def to_json(self) -> str:
        payload = asdict(self)
        payload["names"] = list(self.names)
        for key in ("good_med", "good_mad", "bad_med", "bad_mad", "separation", "weights"):
            payload[key] = [None if not np.isfinite(v) else float(v) for v in getattr(self, key)]
        return json.dumps(payload, indent=2)

    @classmethod
    def from_json(cls, raw: str) -> "Profile":
        payload = json.loads(raw)
        if payload.get("version") != PROFILE_VERSION:
            raise ValueError(
                f"profile version {payload.get('version')} is not supported "
                f"(expected {PROFILE_VERSION}); run `posture-guard calibrate` again"
            )
        arrays = {
            key: np.array([np.nan if v is None else v for v in payload[key]], float)
            for key in ("good_med", "good_mad", "bad_med", "bad_mad", "separation", "weights")
        }
        return cls(
            view=payload["view"],
            names=tuple(payload["names"]),
            n_good=payload["n_good"],
            n_bad=payload["n_bad"],
            disagreement_ref=payload.get("disagreement_ref", MIN_DISAGREEMENT_REF),
            created=payload["created"],
            version=payload["version"],
            **arrays,
        )


def _column_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_rows = values.shape[0]
    med = np.full(values.shape[1], np.nan)
    mad = np.full(values.shape[1], np.nan)
    coverage = np.zeros(values.shape[1])
    for i in range(values.shape[1]):
        col = values[:, i]
        finite = col[np.isfinite(col)]
        coverage[i] = finite.size / max(n_rows, 1)
        if finite.size:
            med[i] = float(np.median(finite))
            mad[i] = median_abs_deviation(finite)
    return med, mad, coverage


def fit_profile(
    view: str,
    names: tuple[str, ...],
    good: np.ndarray,
    bad: np.ndarray,
    *,
    min_separation: float = MIN_SEPARATION,
    prior: np.ndarray | tuple[float, ...] | None = None,
) -> Profile:
    """Build a profile from two stacks of feature vectors.

    ``prior`` lets the feature set express what separation cannot: whether a
    feature measures what its name claims. See :class:`features.FeatureSet`.
    """
    good = np.atleast_2d(np.asarray(good, float))
    bad = np.atleast_2d(np.asarray(bad, float))
    if good.shape[1] != len(names) or bad.shape[1] != len(names):
        raise ValueError("calibration matrices do not match the feature set")
    if good.shape[0] < 5 or bad.shape[0] < 5:
        raise CalibrationError(
            f"not enough usable frames ({good.shape[0]} good, {bad.shape[0]} slouched); "
            "hold each pose still, facing the camera"
        )

    good_med, good_mad, good_cov = _column_stats(good)
    bad_med, bad_mad, bad_cov = _column_stats(bad)

    delta = good_med - bad_med
    pooled = (np.nan_to_num(good_mad, nan=0.0) + np.nan_to_num(bad_mad, nan=0.0)) / 2.0
    # A feature that never moves within a pose would divide by zero and look
    # infinitely discriminative; floor the noise at something small but real.
    floor = np.maximum(np.abs(good_med), np.abs(bad_med)) * 1e-3 + 1e-9
    separation = np.abs(delta) / np.maximum(pooled, floor)

    covered = (good_cov >= MIN_COVERAGE) & (bad_cov >= MIN_COVERAGE)
    separation = np.where(covered & np.isfinite(separation), separation, np.nan)

    raw = np.clip(np.nan_to_num(separation, nan=0.0), 0.0, MAX_SEPARATION) - min_separation
    raw = np.clip(raw, 0.0, None)
    if prior is not None:
        prior = np.asarray(prior, float)
        if prior.shape != raw.shape:
            raise ValueError("prior does not match the feature set")
        raw = raw * prior
    total = raw.sum()
    if total <= 0:
        best = np.nanmax(separation) if np.any(np.isfinite(separation)) else 0.0
        raise CalibrationError(
            "the two postures look the same to this camera "
            f"(best separation {best:.2f}, need more than {min_separation:.2f}). "
            "Exaggerate the difference, or switch to a side camera with "
            "`posture-guard calibrate --view side`."
        )

    profile = Profile(
        view=view,
        names=tuple(names),
        good_med=good_med,
        good_mad=good_mad,
        bad_med=bad_med,
        bad_mad=bad_mad,
        separation=separation,
        weights=raw / total,
        n_good=int(good.shape[0]),
        n_bad=int(bad.shape[0]),
    )
    profile.disagreement_ref = _calibration_disagreement(profile, good, bad)
    return profile


def _calibration_disagreement(profile: Profile, good: np.ndarray, bad: np.ndarray) -> float:
    """How far apart the features sat while you were holding a pose on purpose.

    This is the yardstick for judging disagreement later. Taken high in the
    distribution rather than at the median, because the occasional noisy frame
    should not be enough to trip the off-axis penalty, and floored because two
    carefully held poses understate how much ordinary sitting varies.
    """
    observed = []
    for row in np.vstack([good, bad]):
        parts = score_parts(profile, row)
        if parts is not None:
            observed.append(parts.disagreement)
    if not observed:
        return MIN_DISAGREEMENT_REF
    return float(max(MIN_DISAGREEMENT_REF, np.percentile(observed, 95)))


@dataclass(frozen=True)
class ScoreParts:
    """A score broken into the two things it is made of."""

    value: float  # what the rest of the app uses
    axis: float  # position along good -> slouch
    disagreement: float  # how much the features contradict each other
    penalty: float  # what that disagreement added
    per_feature: np.ndarray  # each feature's own verdict, for diagnostics


def score_parts(profile: Profile, values: np.ndarray) -> ScoreParts | None:
    """Score one feature vector, and show the working.

    Averaging the features gives a position on the line between your two
    calibrated postures. That is the right answer for postures that lie on that
    line, and a misleading one for postures that do not.

    Craning at the screen with your shoulders back is the case that matters here.
    Measured against the ear, the shoulder now sits *behind* where it does in
    your good posture, so ``shoulder_ahead`` and ``neck_incline`` report better
    than perfect while ``head_over_hip`` reports fully slouched. Averaged, they
    cancel and the app calls it good posture.

    What gives it away is not the average but the argument: features that agree
    within a few percent for every on-axis posture are suddenly a full scale
    apart. So the spread across features is measured too, and anything beyond
    what calibration saw is added to the score. A posture nobody demonstrated is
    not assumed to be a good one.

    Returns None when too little of the profile's weight is available in this
    frame -- better than a confident number built from one surviving feature.
    """
    values = np.asarray(values, float)
    if values.shape[-1] != len(profile.names):
        raise ValueError(
            f"profile has {len(profile.names)} features but the frame produced "
            f"{values.shape[-1]}; the feature set changed, so recalibrate"
        )
    delta = profile.delta
    usable = profile.usable & np.isfinite(values) & np.isfinite(delta) & (np.abs(delta) > 0)
    if not usable.any():
        return None

    w = profile.weights[usable]
    total = w.sum()
    if total < MIN_LIVE_WEIGHT:
        return None

    per_feature = (profile.good_med[usable] - values[usable]) / delta[usable]
    # Allow some headroom past both anchors: sitting up straighter than your own
    # demonstration is real, and so is slouching worse than you showed.
    per_feature = np.clip(per_feature, -0.5, 1.5)
    axis = float(np.dot(w, per_feature) / total)

    disagreement = float(np.sqrt(np.dot(w, (per_feature - axis) ** 2) / total))
    penalty = max(0.0, disagreement - profile.disagreement_ref) * DISAGREEMENT_GAIN
    value = float(np.clip(axis + penalty, -0.5, 1.5))

    full = np.full(len(profile.names), np.nan)
    full[usable] = per_feature
    return ScoreParts(
        value=value,
        axis=axis,
        disagreement=disagreement,
        penalty=penalty,
        per_feature=full,
    )


def score(profile: Profile, values: np.ndarray) -> float | None:
    """Position between your good posture (0) and your slouch (1)."""
    parts = score_parts(profile, values)
    return None if parts is None else parts.value


class Scorer:
    """Applies the profile and smooths the result over time.

    An exponential moving average with a time constant of a second or two turns
    per-frame jitter into something stable enough to threshold, while still
    reacting fast enough that sitting up feels immediate.
    """

    def __init__(self, profile: Profile, tau: float = 1.5):
        self.profile = profile
        self.tau = max(tau, 1e-3)
        self._value: float | None = None
        self._last_ts: float | None = None

    @property
    def value(self) -> float | None:
        return self._value

    def reset(self) -> None:
        self._value = None
        self._last_ts = None

    def update(self, ts: float, values: np.ndarray) -> float | None:
        raw = score(self.profile, values)
        if raw is None:
            return None
        if self._value is None or self._last_ts is None or ts <= self._last_ts:
            self._value = raw
        else:
            dt = ts - self._last_ts
            alpha = 1.0 - float(np.exp(-dt / self.tau))
            self._value += alpha * (raw - self._value)
        self._last_ts = ts
        return self._value
