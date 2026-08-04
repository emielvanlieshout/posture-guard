"""Progressive overload for posture.

A fixed threshold stops working the moment you clear it: you sit at 86% good
weeks on end and nothing more happens. So the target moves. Once a week, if the
last seven days cleared the promotion bar, the entry threshold tightens by 8%,
which means a posture that used to pass now registers as a slouch. If the week
went badly the threshold loosens by the same step, because an alarm you cannot
satisfy is one you learn to ignore.

Bounds on both ends keep it sane: it will not tighten past ``ratchet_min_enter``,
where the score is mostly measurement noise, and it will not loosen past
``ratchet_max_enter``, where it would stop asking anything of you.

Every change is written to the event log, so `posture-guard report` can show the
staircase rather than leaving you to wonder why the alerts got stricter.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .config import Config
from .stats import window_stats
from .storage import Store

STATE_ENTER = "ratchet_enter"
STATE_CHANGED_TS = "ratchet_changed_ts"
EVENT_KIND = "threshold_changed"


@dataclass(frozen=True)
class RatchetDecision:
    changed: bool
    old_enter: float
    new_enter: float
    reason: str
    good_fraction: float | None = None
    measured_hours: float = 0.0


def current_enter(store: Store, cfg: Config) -> float:
    """The entry threshold in force: the ratchet's value if any, else the config."""
    raw = store.get_state(STATE_ENTER)
    if raw is None:
        return cfg.enter
    try:
        return max(cfg.ratchet_min_enter, min(cfg.ratchet_max_enter, float(raw)))
    except ValueError:
        return cfg.enter


def decide(
    cfg: Config,
    *,
    enter: float,
    good_fraction: float | None,
    measured_hours: float,
    days_since_change: float,
) -> RatchetDecision:
    """Pure decision step, so the policy can be tested without a database."""
    if not cfg.ratchet_enabled:
        return RatchetDecision(False, enter, enter, "ratchet disabled", good_fraction, measured_hours)
    if days_since_change < cfg.ratchet_min_days:
        return RatchetDecision(
            False,
            enter,
            enter,
            f"only {days_since_change:.1f} of {cfg.ratchet_min_days} days since the last change",
            good_fraction,
            measured_hours,
        )
    if measured_hours < cfg.ratchet_min_hours or good_fraction is None:
        return RatchetDecision(
            False,
            enter,
            enter,
            f"only {measured_hours:.1f}h measured, need {cfg.ratchet_min_hours:.0f}h",
            good_fraction,
            measured_hours,
        )

    if good_fraction >= cfg.ratchet_promote_at:
        proposed = round(enter * cfg.ratchet_step, 4)
        if proposed < cfg.ratchet_min_enter:
            return RatchetDecision(
                False, enter, enter, "already at the tightest threshold", good_fraction, measured_hours
            )
        return RatchetDecision(
            True,
            enter,
            proposed,
            f"{good_fraction:.0%} good over {measured_hours:.0f}h, tightening",
            good_fraction,
            measured_hours,
        )

    if good_fraction <= cfg.ratchet_demote_at:
        proposed = round(min(enter / cfg.ratchet_step, cfg.ratchet_max_enter), 4)
        if proposed <= enter:
            return RatchetDecision(
                False, enter, enter, "already at the loosest threshold", good_fraction, measured_hours
            )
        return RatchetDecision(
            True,
            enter,
            proposed,
            f"{good_fraction:.0%} good over {measured_hours:.0f}h, easing off",
            good_fraction,
            measured_hours,
        )

    return RatchetDecision(
        False,
        enter,
        enter,
        f"{good_fraction:.0%} good sits between the bars, holding",
        good_fraction,
        measured_hours,
    )


def evaluate(store: Store, cfg: Config, now: float | None = None) -> RatchetDecision:
    """Look at the past week and apply the decision, persisting any change."""
    now = time.time() if now is None else now
    enter = current_enter(store, cfg)

    raw_ts = store.get_state(STATE_CHANGED_TS)
    if raw_ts is None:
        first = store.first_bucket_ts()
        since_ts = float(first) if first is not None else now
    else:
        try:
            since_ts = float(raw_ts)
        except ValueError:
            since_ts = now
    days_since = max(0.0, (now - since_ts) / 86400.0)

    week = window_stats(store, now - cfg.ratchet_min_days * 86400, now, cfg.bucket_seconds)
    decision = decide(
        cfg,
        enter=enter,
        good_fraction=week.good_fraction,
        measured_hours=week.measured_hours,
        days_since_change=days_since,
    )

    if decision.changed:
        store.set_state(STATE_ENTER, str(decision.new_enter))
        store.set_state(STATE_CHANGED_TS, str(now))
        store.log_event(
            now,
            EVENT_KIND,
            {
                "from": decision.old_enter,
                "to": decision.new_enter,
                "reason": decision.reason,
                "good_fraction": decision.good_fraction,
                "measured_hours": round(decision.measured_hours, 2),
            },
        )
    return decision
