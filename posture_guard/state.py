"""The alert state machine: when to complain, how hard, and when to stop.

Two deliberate asymmetries shape the behaviour.

The dwell delay means a brief slouch -- reaching for a mug, turning to someone --
never triggers anything. The escalation after that is slow, so the screen fades
rather than snapping, which is easy to sit through if you are mid-thought.

Release is the opposite: the moment the score drops under the exit threshold the
dim is gone in a fraction of a second. Correcting your posture pays out
immediately, while the cost of ignoring it accumulates. That asymmetry is the
whole behavioural mechanism; the thresholds are just bookkeeping.

Entry and exit thresholds differ so the overlay cannot flicker while you hover on
the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class State(str, Enum):
    CALM = "calm"
    WARN = "warn"  # over threshold, still inside the dwell window
    ALERT = "alert"
    ABSENT = "absent"
    PAUSED = "paused"


class EventKind(str, Enum):
    ALERT_STARTED = "alert_started"
    ALERT_ENDED = "alert_ended"
    ABSENT = "absent"
    RETURNED = "returned"
    PAUSED = "paused"
    RESUMED = "resumed"


@dataclass(frozen=True)
class Event:
    ts: float
    kind: EventKind
    detail: dict = field(default_factory=dict)


@dataclass(frozen=True)
class AlertPolicy:
    enter: float = 0.55  # fraction of the way to your calibrated slouch
    exit: float = 0.35
    dwell_s: float = 8.0
    ramp_s: float = 25.0  # time to reach full intensity once alerting
    release_s: float = 0.5  # time to clear once you sit up
    absent_after_s: float = 5.0
    min_intensity: float = 0.15  # first hint must still be perceptible
    # Largest time step the ramp will honour in one update. Without this, a
    # closed lid or a suspended machine produces a dt of hours, and reopening it
    # while slouched would slam the screen to full dim in a single frame.
    max_step_s: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < self.exit < self.enter < 2.0:
            raise ValueError("need 0 < exit < enter")
        for name in ("dwell_s", "ramp_s", "release_s", "absent_after_s"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class Tick:
    ts: float
    state: State
    intensity: float  # 0..1, scaled by the alerter into its own units
    score: float | None
    events: tuple[Event, ...] = ()


class Monitor:
    """Feeds on smoothed scores and emits the current alert state."""

    def __init__(self, policy: AlertPolicy | None = None):
        self.policy = policy or AlertPolicy()
        self.state = State.CALM
        self.intensity = 0.0
        self._last_ts: float | None = None
        self._last_valid_ts: float | None = None
        self._bad_since: float | None = None
        self._alert_since: float | None = None
        self._paused_until: float | None = None

    # -- pausing -------------------------------------------------------------

    def pause(self, ts: float, seconds: float) -> Event:
        self._paused_until = ts + seconds
        self.state = State.PAUSED
        self._bad_since = None
        self._alert_since = None
        return Event(ts, EventKind.PAUSED, {"seconds": seconds})

    def resume(self, ts: float) -> Event:
        self._paused_until = None
        self.state = State.CALM
        return Event(ts, EventKind.RESUMED, {})

    @property
    def paused(self) -> bool:
        return self._paused_until is not None

    def pause_remaining(self, ts: float) -> float:
        return max(0.0, self._paused_until - ts) if self._paused_until else 0.0

    # -- main loop -----------------------------------------------------------

    def update(self, ts: float, score: float | None) -> Tick:
        p = self.policy
        events: list[Event] = []
        dt = 0.0 if self._last_ts is None else max(0.0, ts - self._last_ts)
        self._last_ts = ts

        if self._paused_until is not None:
            if ts >= self._paused_until:
                events.append(self.resume(ts))
            else:
                self.state = State.PAUSED

        if self.state is State.PAUSED:
            target = 0.0
        elif score is None:
            # Short dropouts are normal -- a hand crossing the lens, one bad
            # detection. Only a sustained gap counts as having left the desk.
            gone = self._last_valid_ts is None or (ts - self._last_valid_ts) > p.absent_after_s
            if gone:
                if self.state is not State.ABSENT:
                    events.append(Event(ts, EventKind.ABSENT, {}))
                    if self.state is State.ALERT and self._alert_since is not None:
                        events.append(
                            Event(
                                ts,
                                EventKind.ALERT_ENDED,
                                {"duration": ts - self._alert_since, "reason": "absent"},
                            )
                        )
                self.state = State.ABSENT
                self._bad_since = None
                self._alert_since = None
            target = 0.0 if self.state is not State.ALERT else self._target_for(score)
        else:
            self._last_valid_ts = ts
            if self.state is State.ABSENT:
                events.append(Event(ts, EventKind.RETURNED, {}))
                self.state = State.CALM

            if self.state in (State.CALM, State.WARN):
                if score >= p.enter:
                    if self._bad_since is None:
                        self._bad_since = ts
                        self.state = State.WARN
                    elif (ts - self._bad_since) >= p.dwell_s:
                        self.state = State.ALERT
                        self._alert_since = ts
                        events.append(Event(ts, EventKind.ALERT_STARTED, {"score": score}))
                else:
                    self._bad_since = None
                    self.state = State.CALM
            elif self.state is State.ALERT and score <= p.exit:
                duration = ts - self._alert_since if self._alert_since else 0.0
                events.append(
                    Event(ts, EventKind.ALERT_ENDED, {"duration": duration, "reason": "corrected"})
                )
                self.state = State.CALM
                self._bad_since = None
                self._alert_since = None

            target = self._target_for(score)

        self._move_intensity(target, dt)
        return Tick(
            ts=ts,
            state=self.state,
            intensity=self.intensity,
            score=score,
            events=tuple(events),
        )

    def _target_for(self, score: float | None) -> float:
        if self.state is not State.ALERT or score is None:
            return 0.0
        p = self.policy
        span = max(1.0 - p.exit, 1e-6)
        severity = min(max((score - p.exit) / span, 0.0), 1.0)
        return max(p.min_intensity, severity)

    def _move_intensity(self, target: float, dt: float) -> None:
        p = self.policy
        if dt <= 0:
            return
        rising = target > self.intensity
        step = min(dt, p.max_step_s) / (p.ramp_s if rising else p.release_s)
        delta = target - self.intensity
        self.intensity += max(-step, min(step, delta))
        self.intensity = min(1.0, max(0.0, self.intensity))
