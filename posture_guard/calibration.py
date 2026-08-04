"""Teaching the app what your two postures look like.

You demonstrate the posture you want and the one you fall into, and the
difference between them becomes the scale. Nothing is assumed about your build,
your desk or where the camera sits, which is what lets the same code work from
the front and from the side.

The collection step is deliberately separated from any camera handling: it
consumes an iterable of frames, so the tests drive it with the synthetic model.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .features import FeatureSet, QualityLimits
from .landmarks import PoseFrame
from .scoring import CalibrationError, Profile, fit_profile


@dataclass
class Collected:
    values: np.ndarray  # (n_accepted, n_features)
    rejected: Counter
    seen: int

    @property
    def accepted(self) -> int:
        return int(self.values.shape[0])

    @property
    def acceptance(self) -> float:
        return self.accepted / self.seen if self.seen else 0.0

    def explain(self) -> str:
        if not self.rejected:
            return "every frame usable"
        worst = ", ".join(f"{reason} ({n})" for reason, n in self.rejected.most_common(3))
        return f"{self.accepted}/{self.seen} frames usable; discarded: {worst}"


def collect_samples(
    frames: Iterable[PoseFrame | None],
    feature_set: FeatureSet,
    limits: QualityLimits | None = None,
) -> Collected:
    """Run frames through the extractor and keep the ones that pass the gates."""
    limits = limits or QualityLimits()
    rows: list[np.ndarray] = []
    rejected: Counter = Counter()
    seen = 0

    for frame in frames:
        seen += 1
        if frame is None:
            rejected["nobody detected"] += 1
            continue
        sample = feature_set.extract(frame, limits)
        if sample.ok:
            rows.append(sample.values)
        else:
            rejected[sample.reason or "rejected"] += 1

    values = np.array(rows, float) if rows else np.empty((0, feature_set.n))
    return Collected(values=values, rejected=rejected, seen=seen)


def build_profile(
    feature_set: FeatureSet,
    good: Collected,
    bad: Collected,
    *,
    min_separation: float = 1.0,
) -> Profile:
    """Fit a profile, translating thin data into an actionable message."""
    for label, collected in (("good", good), ("slouched", bad)):
        if collected.accepted < 5:
            raise CalibrationError(
                f"only {collected.accepted} usable frames for the {label} pose "
                f"({collected.explain()}). Hold still, keep your whole head and both "
                "shoulders in frame, and try again."
            )
    return fit_profile(
        feature_set.view,
        feature_set.names,
        good.values,
        bad.values,
        min_separation=min_separation,
    )


def verdict(profile: Profile, feature_set: FeatureSet) -> tuple[bool, str]:
    """Judge whether this profile is worth trusting, and say why.

    Two separate questions. Can the camera tell the two postures apart at all --
    that is the separation number. And is it telling them apart *by the
    shoulders* -- which, on a frontal camera, it cannot, so that caveat is stated
    every time rather than left for you to discover after three weeks of
    training the wrong thing.
    """
    best = float(np.nanmax(profile.separation)) if np.any(np.isfinite(profile.separation)) else 0.0

    if best < 1.5:
        return False, (
            f"Weak separation (best feature {best:.1f} noise widths). The camera can "
            "barely tell your two postures apart; expect false alarms and misses. "
            "Exaggerate the difference between the poses, improve the lighting, or "
            "move the camera so your whole upper body is in frame."
        )

    strength = "Strong" if best >= 3.0 else "Workable"
    note = f"{strength} separation (best feature {best:.1f} noise widths)."

    if feature_set.view == "frontal":
        shoulder_weight = float(sum(profile.weights[i] for i in feature_set.primary))
        note += (
            f" Head-on, {1 - shoulder_weight:.0%} of the weight lands on head and neck"
            " features rather than on shoulder width, because perspective hides"
            " protraction from a frontal camera at desk distance. You will be training"
            " the whole slouch, which normally drags the shoulders along with it. If the"
            " shoulders themselves are the point, a side camera measures them directly:"
            " `posture-guard calibrate --view side`."
        )
    return True, note
