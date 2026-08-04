"""End-to-end check on synthetic data, so the install can be verified without a camera.

Everything except the camera driver and the AppKit overlay is exercised here:
feature extraction, calibration, scoring, the alert state machine, the database
and the report. If this passes, a problem on the real thing is a camera, a
permission or a calibration problem -- not a broken pipeline.
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .calibration import build_profile, collect_samples, verdict
from .config import Config
from .features import get_feature_set
from .report import render_html, render_text
from .scoring import Scorer, score
from .state import AlertPolicy, EventKind, Monitor
from .storage import BucketAggregator, Store
from .synth import Camera, Posture, synth_series

GOOD_ANGLE = 2.0
SLOUCH_ANGLE = 26.0


@dataclass
class Check:
    name: str
    passed: bool
    detail: str

    def __str__(self) -> str:
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.name}: {self.detail}"


def _posture(view: str, angle: float) -> Posture:
    return Posture(protraction_deg=angle, yaw_deg=90.0 if view == "side" else 0.0)


def _series(view: str, angle: float, n: int = 90, seed: int = 0, **kw):
    return synth_series(angle, n=n, seed=seed, posture=_posture(view, 0.0), **kw)


def run_selftest(view: str = "side") -> list[Check]:
    feature_set = get_feature_set(view)
    checks: list[Check] = []

    # 1. Calibration -------------------------------------------------------
    good = collect_samples(_series(view, GOOD_ANGLE, seed=1), feature_set)
    bad = collect_samples(_series(view, SLOUCH_ANGLE, seed=2), feature_set)
    try:
        profile = build_profile(feature_set, good, bad)
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("calibration", False, str(exc)))
        return checks

    best = float(np.nanmax(profile.separation))
    checks.append(
        Check(
            "calibration",
            best >= 1.5,
            f"{good.accepted}+{bad.accepted} frames, best separation {best:.1f} noise widths",
        )
    )
    ok, note = verdict(profile, feature_set)
    checks.append(Check("calibration verdict", ok, note))

    # 2. Anchors -----------------------------------------------------------
    good_score = float(np.median([score(profile, v) or np.nan for v in good.values]))
    bad_score = float(np.median([score(profile, v) or np.nan for v in bad.values]))
    checks.append(
        Check(
            "score anchors",
            abs(good_score) < 0.25 and abs(bad_score - 1.0) < 0.25,
            f"good posture scores {good_score:+.2f} (want ~0), slouch {bad_score:+.2f} (want ~1)",
        )
    )

    # 3. Monotonicity ------------------------------------------------------
    angles = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    curve = []
    for angle in angles:
        samples = collect_samples(_series(view, angle, n=40, seed=3), feature_set)
        curve.append(float(np.median([score(profile, v) or np.nan for v in samples.values])))
    increases = sum(1 for a, b in zip(curve, curve[1:]) if b > a)
    checks.append(
        Check(
            "monotonic in protraction",
            increases >= len(angles) - 2 and curve[-1] > curve[0] + 0.5,
            "score by angle: " + ", ".join(f"{a:.0f}deg={s:+.2f}" for a, s in zip(angles, curve)),
        )
    )

    # 4. Camera distance ---------------------------------------------------
    base = Camera().distance
    spread = []
    for factor in (0.8, 1.0, 1.25):
        samples = collect_samples(
            _series(view, 18.0, n=40, seed=4, camera=Camera(distance=base * factor)),
            feature_set,
        )
        spread.append(float(np.median([score(profile, v) or np.nan for v in samples.values])))
    drift = max(spread) - min(spread)
    checks.append(
        Check(
            "distance invariance",
            drift < 0.25,
            f"score moves {drift:.2f} across a +-25% change in camera distance",
        )
    )

    # 5. Alert state machine ----------------------------------------------
    policy = AlertPolicy(enter=0.55, exit=0.35, dwell_s=8.0, ramp_s=25.0, release_s=0.5)
    monitor = Monitor(policy)
    scorer = Scorer(profile, tau=1.0)
    started = ended = None
    peak = 0.0
    t0 = 1_700_000_000.0
    timeline = [(SLOUCH_ANGLE, 60.0), (GOOD_ANGLE, 20.0)]

    t = t0
    corrected_at = None
    for index, (angle, duration) in enumerate(timeline):
        if index == 1:
            corrected_at = t  # the moment the synthetic sitter straightens up
        frames = _series(view, angle, n=int(duration * 6), seed=5)
        for frame in frames:
            sample = feature_set.extract(frame)
            value = scorer.update(t, sample.values) if sample.ok else None
            tick = monitor.update(t, value)
            peak = max(peak, tick.intensity)
            for event in tick.events:
                if event.kind is EventKind.ALERT_STARTED and started is None:
                    started = t - t0
                elif event.kind is EventKind.ALERT_ENDED and ended is None:
                    ended = t - t0
            t += 1 / 6
    release_s = None if ended is None or corrected_at is None else ended - (corrected_at - t0)

    dwell_ok = started is not None and 7.0 <= started <= 20.0
    checks.append(
        Check(
            "alert fires after the dwell window",
            dwell_ok,
            f"first alert at {started:.1f}s (dwell is {policy.dwell_s:.0f}s)"
            if started is not None
            else "no alert fired while slouching",
        )
    )
    checks.append(
        Check(
            "alert clears on correction",
            release_s is not None and release_s < 5.0 and peak > 0.2,
            f"cleared {release_s:.1f}s after sitting up, peak dim reached {peak:.0%}"
            if release_s is not None
            else "alert never cleared",
        )
    )

    # 6. Storage and report ------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "selftest.db"
        cfg = Config(view=view)
        with Store(db) as store:
            aggregator = BucketAggregator(store, cfg.bucket_seconds, good_below=policy.enter)
            now = time.time()
            for i in range(240):
                aggregator.add(now - (240 - i) * 30, 0.3 if i % 3 else 0.8, 30.0)
            aggregator.flush()
            store.log_event(now, "alert_started", {"score": 0.8})
            text = render_text(store, cfg, days=3)
            html = render_html(store, cfg, days=3)
        checks.append(
            Check(
                "history and report",
                "posture-guard history" in text and html.startswith("<!doctype html>"),
                f"{len(text.splitlines())} lines of text report, {len(html)} bytes of HTML",
            )
        )

    return checks


def selftest_passed(checks: list[Check]) -> bool:
    return all(c.passed for c in checks)
