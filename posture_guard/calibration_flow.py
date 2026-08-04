"""The calibration flow, as a state machine with no user interface attached.

Separated from the window on purpose. The interesting part -- when the
countdown ends, which pose is being held, what to say when frames are being
rejected, whether the result is worth saving -- is ordinary logic, and keeping
it here means it can be driven by a fake clock and synthetic frames in the
tests rather than only by a person sitting in front of a camera.

The window in ``ui/calibrate_window.py`` renders a :class:`View` and forwards
button presses. It makes no decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .calibration import Collected, build_profile, diagnose, verdict
from .features import FeatureSample, FeatureSet, QualityLimits
from .landmarks import PoseFrame
from .scoring import CalibrationError, Profile

COUNTDOWN_S = 5.0
CAPTURE_S = 12.0


class Phase(str, Enum):
    READY = "ready"  # explaining the pose, waiting for the button
    COUNTDOWN = "countdown"
    CAPTURING = "capturing"
    REVIEW = "review"  # a profile was built, waiting for save or retry
    FAILED = "failed"  # it could not be built, with a reason


@dataclass(frozen=True)
class Pose:
    key: str
    heading: str
    detail: str


POSES: tuple[Pose, ...] = (
    Pose(
        "good",
        "Sit the way you want to sit",
        "Shoulders rolled back and down, chest open, ears above your shoulders. "
        "Hold it while the bar fills.",
    ),
    Pose(
        "slouch",
        "Now let yourself slouch",
        "Let the shoulders roll forward and up, exactly the way you catch yourself "
        "sitting. Exaggerate it a little.",
    ),
)


@dataclass(frozen=True)
class View:
    """Everything the window needs to draw. No behaviour, no camera, no AppKit."""

    phase: Phase
    heading: str
    detail: str
    hint: str = ""
    hint_ok: bool = True
    countdown: int | None = None
    progress: float = 0.0
    accepted: int = 0
    step: str = ""
    button: str | None = None
    secondary: str | None = None
    finished: bool = False


def hint_for(sample: FeatureSample | None, view: str) -> tuple[str, bool]:
    """Plain-language feedback on the frame in front of the camera right now.

    This is the whole reason for having a window. In the terminal you find out
    that every frame was unusable once the twelve seconds are over; here you find
    out while there is still time to move.
    """
    if sample is None:
        return ("No-one in view — make sure your head and both shoulders are in frame.", False)
    if sample.ok:
        return ("Got you.", True)

    reason = sample.reason
    if reason.startswith("not a side view"):
        return (
            "This camera is looking at your front. Close this and run "
            "`posture-guard config --set view=frontal`, or move the camera to your side.",
            False,
        )
    if reason.startswith("facing"):
        return ("Turn a little further away from the camera.", False)
    if reason.startswith("head turned"):
        return ("Face the camera squarely.", False)
    if reason.startswith("torso tilted"):
        return ("Sit level — one shoulder is much higher than the other.", False)
    if reason.startswith("landmarks not visible"):
        return ("Hard to see you. More light, or move so your upper body fills more of the frame.", False)
    if reason.startswith("face too small"):
        return ("Too far away. Move closer, or move the camera in.", False)
    return (reason, False)


class CalibrationSession:
    """Drives the two poses and builds the profile. Feed it a clock and frames."""

    def __init__(
        self,
        feature_set: FeatureSet,
        limits: QualityLimits | None = None,
        *,
        countdown_s: float = COUNTDOWN_S,
        capture_s: float = CAPTURE_S,
        min_separation: float = 1.0,
    ):
        self.feature_set = feature_set
        self.limits = limits or QualityLimits()
        self.countdown_s = countdown_s
        self.capture_s = capture_s
        self.min_separation = min_separation

        self.phase = Phase.READY
        self.index = 0
        self.profile: Profile | None = None
        self.verdict_text = ""
        self.verdict_ok = False
        self.error = ""

        self._phase_started: float | None = None
        self._last_sample: FeatureSample | None = None
        self._rows: dict[str, list[np.ndarray]] = {p.key: [] for p in POSES}
        self._seen: dict[str, int] = {p.key: 0 for p in POSES}
        self._rejects: dict[str, dict[str, int]] = {p.key: {} for p in POSES}
        self._metrics: dict[str, dict[str, list[float]]] = {p.key: {} for p in POSES}

    # -- input ---------------------------------------------------------------

    @property
    def pose(self) -> Pose:
        return POSES[min(self.index, len(POSES) - 1)]

    def press(self, now: float) -> None:
        """The primary button. Starts a pose, or restarts after a result."""
        if self.phase is Phase.READY:
            self.phase = Phase.COUNTDOWN
            self._phase_started = now
        elif self.phase in (Phase.REVIEW, Phase.FAILED):
            self.restart(now)

    def restart(self, now: float | None = None) -> None:
        """Back to the first pose, throwing away everything collected."""
        self.phase = Phase.READY
        self.index = 0
        self.profile = None
        self.verdict_text = ""
        self.verdict_ok = False
        self.error = ""
        self._phase_started = None
        self._last_sample = None
        self._rows = {p.key: [] for p in POSES}
        self._seen = {p.key: 0 for p in POSES}
        self._rejects = {p.key: {} for p in POSES}
        self._metrics = {p.key: {} for p in POSES}

    def offer(self, frame: PoseFrame | None) -> FeatureSample | None:
        """Hand over the latest frame. Kept only while a pose is being captured."""
        sample = self.feature_set.extract(frame, self.limits) if frame is not None else None
        self._last_sample = sample

        if self.phase is not Phase.CAPTURING:
            return sample

        key = self.pose.key
        self._seen[key] += 1
        if sample is None:
            self._rejects[key]["nobody detected"] = self._rejects[key].get("nobody detected", 0) + 1
            return sample
        if sample.ok:
            self._rows[key].append(sample.values)
        else:
            reason = sample.reason or "rejected"
            self._rejects[key][reason] = self._rejects[key].get(reason, 0) + 1
        for name, value in sample.metrics.items():
            if np.isfinite(value):
                self._metrics[key].setdefault(name, []).append(float(value))
        return sample

    # -- output --------------------------------------------------------------

    def tick(self, now: float) -> View:
        """Advance time and describe what should be on screen."""
        self._advance(now)
        return self._render(now)

    def _advance(self, now: float) -> None:
        if self._phase_started is None:
            return
        elapsed = now - self._phase_started

        if self.phase is Phase.COUNTDOWN and elapsed >= self.countdown_s:
            self.phase = Phase.CAPTURING
            self._phase_started = now
        elif self.phase is Phase.CAPTURING and elapsed >= self.capture_s:
            if self.index + 1 < len(POSES):
                self.index += 1
                self.phase = Phase.COUNTDOWN
                self._phase_started = now
            else:
                self._finish()

    def _collected(self, key: str) -> Collected:
        rows = self._rows[key]
        from collections import Counter

        return Collected(
            values=np.array(rows, float) if rows else np.empty((0, self.feature_set.n)),
            rejected=Counter(self._rejects[key]),
            seen=self._seen[key],
            metrics={
                name: float(np.median(values))
                for name, values in self._metrics[key].items()
                if values
            },
        )

    def _finish(self) -> None:
        self._phase_started = None
        good, bad = self._collected("good"), self._collected("slouch")
        try:
            self.profile = build_profile(
                self.feature_set, good, bad, min_separation=self.min_separation, limits=self.limits
            )
        except CalibrationError as exc:
            self.phase = Phase.FAILED
            worst = good if good.accepted <= bad.accepted else bad
            advice = diagnose(self.feature_set, worst, self.limits)
            self.error = f"{exc}\n\n{advice}" if advice and advice not in str(exc) else str(exc)
            return
        self.verdict_ok, self.verdict_text = verdict(self.profile, self.feature_set)
        self.phase = Phase.REVIEW

    def _render(self, now: float) -> View:
        pose = self.pose
        step = f"Step {self.index + 1} of {len(POSES)}"
        elapsed = 0.0 if self._phase_started is None else now - self._phase_started
        hint, hint_ok = hint_for(self._last_sample, self.feature_set.view)

        if self.phase is Phase.READY:
            return View(
                phase=self.phase,
                heading=pose.heading,
                detail=pose.detail,
                hint=hint,
                hint_ok=hint_ok,
                step=step,
                button="Start",
            )
        if self.phase is Phase.COUNTDOWN:
            remaining = max(0.0, self.countdown_s - elapsed)
            return View(
                phase=self.phase,
                heading=pose.heading,
                detail=pose.detail,
                hint=hint,
                hint_ok=hint_ok,
                countdown=int(remaining) + 1,
                progress=min(1.0, elapsed / self.countdown_s),
                step=step,
            )
        if self.phase is Phase.CAPTURING:
            return View(
                phase=self.phase,
                heading=f"Hold it — {pose.heading.lower()}",
                detail=pose.detail,
                hint=hint,
                hint_ok=hint_ok,
                progress=min(1.0, elapsed / self.capture_s),
                accepted=len(self._rows[pose.key]),
                step=step,
            )
        if self.phase is Phase.REVIEW:
            return View(
                phase=self.phase,
                heading="Done" if self.verdict_ok else "That will not work well",
                detail=self.verdict_text,
                hint=self.profile.describe() if self.profile else "",
                hint_ok=self.verdict_ok,
                progress=1.0,
                button="Save and close" if self.verdict_ok else None,
                secondary="Try again",
                finished=True,
            )
        return View(
            phase=Phase.FAILED,
            heading="Calibration failed",
            detail=self.error,
            hint_ok=False,
            secondary="Try again",
            finished=True,
        )
