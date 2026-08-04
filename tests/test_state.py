from __future__ import annotations

import pytest

from posture_guard.state import AlertPolicy, EventKind, Monitor, State

POLICY = AlertPolicy(enter=0.6, exit=0.4, dwell_s=8.0, ramp_s=20.0, release_s=0.5, absent_after_s=5.0)


def feed(monitor, score, seconds, start=1000.0, step=0.25):
    """Run a constant score for a while; returns the events seen and the last tick."""
    events, tick, t = [], None, start
    for _ in range(int(seconds / step)):
        tick = monitor.update(t, score)
        events.extend(tick.events)
        t += step
    return events, tick, t


class TestDwell:
    def test_a_brief_slouch_never_alerts(self):
        monitor = Monitor(POLICY)
        events, tick, _ = feed(monitor, 0.9, seconds=6.0)
        assert not events
        assert tick.state is State.WARN
        assert tick.intensity == 0.0

    def test_a_sustained_slouch_alerts_after_the_dwell(self):
        monitor = Monitor(POLICY)
        events, tick, _ = feed(monitor, 0.9, seconds=12.0)
        assert [e.kind for e in events] == [EventKind.ALERT_STARTED]
        assert tick.state is State.ALERT

    def test_the_dwell_timer_resets_on_sitting_up(self):
        monitor = Monitor(POLICY)
        _, _, t = feed(monitor, 0.9, seconds=6.0)
        _, _, t = feed(monitor, 0.1, seconds=1.0, start=t)
        events, tick, _ = feed(monitor, 0.9, seconds=6.0, start=t)
        assert not events, "the clock should have restarted"
        assert tick.state is State.WARN


class TestHysteresis:
    def test_hovering_between_the_thresholds_does_not_flicker(self):
        monitor = Monitor(POLICY)
        feed(monitor, 0.9, seconds=12.0)
        events, tick, _ = feed(monitor, 0.5, seconds=10.0, start=2000.0)
        assert not events, "0.5 is under enter but over exit: stay alerting"
        assert tick.state is State.ALERT

    def test_dropping_under_exit_clears_it(self):
        monitor = Monitor(POLICY)
        feed(monitor, 0.9, seconds=12.0)
        events, tick, _ = feed(monitor, 0.2, seconds=2.0, start=2000.0)
        assert [e.kind for e in events] == [EventKind.ALERT_ENDED]
        assert events[0].detail["reason"] == "corrected"
        assert tick.state is State.CALM

    def test_exit_must_sit_below_enter(self):
        with pytest.raises(ValueError):
            AlertPolicy(enter=0.3, exit=0.5)


class TestIntensity:
    def test_it_ramps_slowly_and_releases_fast(self):
        monitor = Monitor(POLICY)
        _, _, t = feed(monitor, 1.0, seconds=12.0)  # alert starts around t+8
        _, mid, t = feed(monitor, 1.0, seconds=5.0, start=t)
        assert 0.1 < mid.intensity < 0.6, "part way up a 20 second ramp"

        _, after, t = feed(monitor, 1.0, seconds=25.0, start=t)
        assert after.intensity == pytest.approx(1.0, abs=0.02)

        _, cleared, _ = feed(monitor, 0.0, seconds=1.0, start=t)
        assert cleared.intensity == 0.0, "gone within a second of sitting up"

    def test_waking_from_sleep_does_not_slam_the_screen(self):
        """A closed lid produces an enormous dt; the ramp must not honour it."""
        monitor = Monitor(POLICY)
        _, before, t = feed(monitor, 1.0, seconds=12.0)
        assert before.intensity < 0.3

        after_sleep = monitor.update(t + 4 * 3600, 1.0)
        assert after_sleep.intensity < before.intensity + 0.1

    def test_a_deeper_slouch_dims_further(self):
        shallow, deep = Monitor(POLICY), Monitor(POLICY)
        _, a, _ = feed(shallow, 0.65, seconds=40.0)
        _, b, _ = feed(deep, 1.0, seconds=40.0)
        assert b.intensity > a.intensity

    def test_intensity_never_leaves_the_unit_interval(self):
        monitor = Monitor(POLICY)
        _, tick, _ = feed(monitor, 5.0, seconds=120.0)
        assert 0.0 <= tick.intensity <= 1.0


class TestAbsence:
    def test_a_short_dropout_is_ignored(self):
        monitor = Monitor(POLICY)
        _, _, t = feed(monitor, 0.9, seconds=12.0)
        events, tick, _ = feed(monitor, None, seconds=2.0, start=t)
        assert not events
        assert tick.state is State.ALERT, "a hand across the lens is not leaving the desk"

    def test_a_long_gap_counts_as_gone_and_clears_the_dim(self):
        monitor = Monitor(POLICY)
        _, _, t = feed(monitor, 0.9, seconds=12.0)
        events, tick, _ = feed(monitor, None, seconds=10.0, start=t)
        kinds = [e.kind for e in events]
        assert EventKind.ABSENT in kinds
        assert EventKind.ALERT_ENDED in kinds
        assert tick.state is State.ABSENT
        assert tick.intensity == 0.0

    def test_coming_back_is_reported(self):
        monitor = Monitor(POLICY)
        _, _, t = feed(monitor, None, seconds=10.0)
        events, tick, _ = feed(monitor, 0.1, seconds=1.0, start=t)
        assert [e.kind for e in events] == [EventKind.RETURNED]
        assert tick.state is State.CALM


class TestPausing:
    def test_pausing_suppresses_the_dim(self):
        monitor = Monitor(POLICY)
        feed(monitor, 0.9, seconds=12.0)
        monitor.pause(2000.0, 600.0)
        _, tick, _ = feed(monitor, 0.9, seconds=30.0, start=2000.0)
        assert tick.state is State.PAUSED
        assert tick.intensity == 0.0

    def test_it_expires_on_its_own(self):
        monitor = Monitor(POLICY)
        monitor.pause(1000.0, 5.0)
        events, tick, _ = feed(monitor, 0.1, seconds=10.0, start=1000.0)
        assert EventKind.RESUMED in [e.kind for e in events]
        assert tick.state is not State.PAUSED

    def test_resume_is_immediate(self):
        monitor = Monitor(POLICY)
        monitor.pause(1000.0, 600.0)
        monitor.resume(1005.0)
        assert not monitor.paused
        _, tick, _ = feed(monitor, 0.9, seconds=12.0, start=1005.0)
        assert tick.state is State.ALERT

    def test_remaining_time_counts_down(self):
        monitor = Monitor(POLICY)
        monitor.pause(1000.0, 600.0)
        assert monitor.pause_remaining(1300.0) == pytest.approx(300.0)
        assert monitor.pause_remaining(9999.0) == 0.0
