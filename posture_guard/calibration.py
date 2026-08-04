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
from dataclasses import dataclass, field
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
    #: Median of each gate quantity across every frame, accepted or not.
    metrics: dict = field(default_factory=dict)

    @property
    def accepted(self) -> int:
        return int(self.values.shape[0])

    @property
    def acceptance(self) -> float:
        return self.accepted / self.seen if self.seen else 0.0

    @property
    def worst_reason(self) -> str:
        return self.rejected.most_common(1)[0][0] if self.rejected else ""

    def explain(self) -> str:
        if not self.rejected:
            return "every frame usable"
        worst = ", ".join(f"{reason} ({n})" for reason, n in self.rejected.most_common(3))
        return f"{self.accepted}/{self.seen} frames usable; discarded: {worst}"

    def measured(self) -> str:
        if not self.metrics:
            return ""
        return ", ".join(f"{name} {value:.2f}" for name, value in sorted(self.metrics.items()))


def diagnose(feature_set: FeatureSet, collected: Collected, limits: QualityLimits) -> str:
    """Say what the numbers mean and what to do, not merely that it failed.

    "not a side view" is the same message whether the camera is pointed at your
    face or the threshold is a shade too tight, and those need opposite actions.
    The measured value tells them apart, so it goes in the advice.
    """
    reason = collected.worst_reason
    metrics = collected.metrics

    if feature_set.view == "side" and reason.startswith("not a side view"):
        ratio = metrics.get("ear_over_face", float("nan"))
        facing = abs(metrics.get("facing", float("nan")))
        looks_frontal = ratio > limits.max_ear_over_face * 2 or facing < limits.min_facing / 2
        if looks_frontal:
            return (
                f"Measured ear span {ratio:.2f} of face height (a profile is well under "
                f"{limits.max_ear_over_face:.2f}) and facing {facing:.2f} (a profile is "
                f"over {limits.min_facing:.2f}). This camera is looking at your front.\n"
                "  Either point a camera at your side and select it:\n"
                "    posture-guard doctor                        # lists cameras by name\n"
                "    posture-guard config --set camera_index=N\n"
                "  Or calibrate the built-in webcam instead, which measures the slouch\n"
                "  complex rather than protraction itself:\n"
                "    posture-guard calibrate --view frontal"
            )
        return (
            f"Measured ear span {ratio:.2f} of face height, just over the "
            f"{limits.max_ear_over_face:.2f} a profile should be under, with facing "
            f"{facing:.2f}. The camera is close to side-on but not quite there. Rotate "
            "your chair or the camera a little further, or loosen the gate:\n"
            "    posture-guard config --set max_ear_over_face=1.0"
        )

    if reason.startswith("facing"):
        return (
            f"Measured facing {abs(metrics.get('facing', float('nan'))):.2f}, under the "
            f"{limits.min_facing:.2f} needed to tell which way you are pointing. Move the "
            "camera further round to your side."
        )
    if reason.startswith("head turned"):
        return (
            f"Measured yaw {abs(metrics.get('yaw', float('nan'))):.2f}, over the "
            f"{limits.max_yaw:.2f} limit. Face the camera squarely while calibrating."
        )
    if reason.startswith("landmarks not visible"):
        return (
            f"Landmark visibility {metrics.get('visibility', float('nan')):.2f}, under "
            f"{limits.min_visibility:.2f}. More light, or move so your whole upper body "
            "is in frame."
        )
    if reason.startswith("nobody detected"):
        return "No pose was found at all. Check `posture-guard preview` to see what the camera sees."
    return ""


def collect_samples(
    frames: Iterable[PoseFrame | None],
    feature_set: FeatureSet,
    limits: QualityLimits | None = None,
) -> Collected:
    """Run frames through the extractor and keep the ones that pass the gates."""
    limits = limits or QualityLimits()
    rows: list[np.ndarray] = []
    rejected: Counter = Counter()
    observed: dict[str, list[float]] = {}
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
        # Gathered from rejected frames too: when nothing passes, these are the
        # only evidence of why.
        for name, value in sample.metrics.items():
            if np.isfinite(value):
                observed.setdefault(name, []).append(float(value))

    values = np.array(rows, float) if rows else np.empty((0, feature_set.n))
    metrics = {name: float(np.median(vals)) for name, vals in observed.items() if vals}
    return Collected(values=values, rejected=rejected, seen=seen, metrics=metrics)


def build_profile(
    feature_set: FeatureSet,
    good: Collected,
    bad: Collected,
    *,
    min_separation: float = 1.0,
    limits: QualityLimits | None = None,
) -> Profile:
    """Fit a profile, translating thin data into an actionable message."""
    limits = limits or QualityLimits()
    for label, collected in (("good", good), ("slouched", bad)):
        if collected.accepted < 5:
            advice = diagnose(feature_set, collected, limits)
            raise CalibrationError(
                f"only {collected.accepted} usable frames for the {label} pose.\n"
                f"  {collected.explain()}\n"
                + (f"\n{advice}" if advice else "  Hold still and keep your whole upper "
                   "body in frame, then try again.")
            )
    return fit_profile(
        feature_set.view,
        feature_set.names,
        good.values,
        bad.values,
        min_separation=min_separation,
        prior=feature_set.prior,
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

    if feature_set.view == "side":
        hip_weight = float(sum(profile.weights[i] for i in feature_set.hip_dependent))
        if hip_weight < 0.15:
            return True, note + (
                " Your hips were not in frame, so every feature here measures one part"
                " of you against another part that moves too. Forward head posture"
                " shifts the ear the same way protraction shifts the shoulder, and the"
                " two cancel -- badly enough to invert the sign when the head leads."
                " In practice that means pulling your chin back would satisfy this"
                " profile with your shoulders untouched. Move the camera back or down"
                " until your hip is in shot, then calibrate again."
            )
        protraction_weight = float(sum(profile.weights[i] for i in feature_set.primary))
        if protraction_weight < 0.15:
            return True, note + (
                f" Note that only {protraction_weight:.0%} of the weight sits on the"
                " pelvis-referenced shoulder measurement; the rest is head and neck."
                " Try exaggerating the shoulder difference between the two poses while"
                " keeping your head in the same place."
            )

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
