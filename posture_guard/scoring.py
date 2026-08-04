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

PROFILE_VERSION = 1

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
) -> Profile:
    """Build a profile from two stacks of feature vectors."""
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
    total = raw.sum()
    if total <= 0:
        best = np.nanmax(separation) if np.any(np.isfinite(separation)) else 0.0
        raise CalibrationError(
            "the two postures look the same to this camera "
            f"(best separation {best:.2f}, need more than {min_separation:.2f}). "
            "Exaggerate the difference, or switch to a side camera with "
            "`posture-guard calibrate --view side`."
        )

    return Profile(
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


def score(profile: Profile, values: np.ndarray) -> float | None:
    """Position of one feature vector between good (0) and slouched (1).

    Returns None when too little of the profile's weight is available in this
    frame, which is the honest answer -- better than a confident number built
    from one surviving feature.
    """
    values = np.asarray(values, float)
    delta = profile.delta
    usable = profile.usable & np.isfinite(values) & np.isfinite(delta) & (np.abs(delta) > 0)
    if not usable.any():
        return None

    w = profile.weights[usable]
    if w.sum() < MIN_LIVE_WEIGHT:
        return None

    per_feature = (profile.good_med[usable] - values[usable]) / delta[usable]
    # Allow some headroom past both anchors: sitting up straighter than your own
    # demonstration is real, and so is slouching worse than you showed.
    per_feature = np.clip(per_feature, -0.5, 1.5)
    return float(np.dot(w, per_feature) / w.sum())


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
