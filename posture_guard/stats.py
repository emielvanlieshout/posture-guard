"""Aggregations over the bucket history, shared by the reports and the ratchet.

Two different numbers matter and they answer different questions.

``good_fraction`` is measured against the *current* threshold, so it says how you
are doing against today's target. It is what the ratchet acts on, and it is
deliberately not comparable across a threshold change.

``mean_score`` is anchored on your calibration instead, so it stays comparable
for as long as the profile does. It is the one to watch over months.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

from .storage import Bucket, Store


@dataclass(frozen=True)
class WindowStats:
    start_ts: float
    end_ts: float
    measured_hours: float
    good_hours: float
    bad_hours: float
    alert_hours: float
    mean_score: float | None
    alerts: int
    longest_good_minutes: float

    @property
    def good_fraction(self) -> float | None:
        total = self.good_hours + self.bad_hours
        return self.good_hours / total if total > 0 else None

    @property
    def alerts_per_hour(self) -> float | None:
        return self.alerts / self.measured_hours if self.measured_hours > 0 else None


@dataclass(frozen=True)
class DayStats:
    day: date
    stats: WindowStats


def _summarise(
    buckets: list[Bucket], alerts: int, start_ts: float, end_ts: float, bucket_seconds: int
) -> WindowStats:
    good = sum(b.secs_good for b in buckets)
    bad = sum(b.secs_bad for b in buckets)
    alert = sum(b.secs_alert for b in buckets)
    n_valid = sum(b.n_valid for b in buckets)
    score_sum = sum(b.score_sum for b in buckets)

    return WindowStats(
        start_ts=start_ts,
        end_ts=end_ts,
        measured_hours=(good + bad) / 3600.0,
        good_hours=good / 3600.0,
        bad_hours=bad / 3600.0,
        alert_hours=alert / 3600.0,
        mean_score=(score_sum / n_valid) if n_valid else None,
        alerts=alerts,
        longest_good_minutes=_longest_good_run(buckets, bucket_seconds) / 60.0,
    )


def _longest_good_run(buckets: list[Bucket], bucket_seconds: int) -> float:
    """Longest unbroken stretch without a single second over threshold.

    Resolution is one bucket, and a gap in the timeline (lunch, a closed lid)
    ends the run rather than silently bridging it.
    """
    best = current = 0.0
    previous_ts: int | None = None
    for b in buckets:
        contiguous = previous_ts is not None and b.bucket_ts - previous_ts <= bucket_seconds
        if b.secs_bad == 0 and b.secs_good > 0:
            current = (current if contiguous else 0.0) + b.secs_good
            best = max(best, current)
        else:
            current = 0.0
        previous_ts = b.bucket_ts
    return best


def window_stats(
    store: Store, start_ts: float, end_ts: float, bucket_seconds: int = 30
) -> WindowStats:
    buckets = store.buckets_between(start_ts, end_ts)
    alerts = store.count_events(start_ts, end_ts, "alert_started")
    return _summarise(buckets, alerts, start_ts, end_ts, bucket_seconds)


def daily_stats(store: Store, days: int = 14, bucket_seconds: int = 30) -> list[DayStats]:
    """One entry per calendar day (local time), oldest first, gaps included."""
    today = datetime.now().date()
    first = today - timedelta(days=days - 1)
    out: list[DayStats] = []
    for offset in range(days):
        day = first + timedelta(days=offset)
        start = datetime.combine(day, datetime.min.time()).timestamp()
        end = start + 86400
        out.append(DayStats(day=day, stats=window_stats(store, start, end, bucket_seconds)))
    return out


def trend(series: list[float | None]) -> float | None:
    """Least-squares slope per step over the non-empty points, or None."""
    points = [(i, v) for i, v in enumerate(series) if v is not None]
    if len(points) < 3:
        return None
    n = len(points)
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    denom = sum((x - mean_x) ** 2 for x, _ in points)
    if denom == 0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in points) / denom
