from __future__ import annotations

import time

import numpy as np
import pytest

from posture_guard.alerts import SafeAlerter
from posture_guard.app import Runner, Shared, _clear_pause_flag, _read_pause_flag, _write_pause_flag
from posture_guard.config import Config
from posture_guard.scoring import fit_profile
from posture_guard.state import State, Tick
from posture_guard.storage import Store
from posture_guard.tray import title_for


@pytest.fixture(autouse=True)
def isolated_home(monkeypatch, tmp_path):
    monkeypatch.setenv("POSTURE_GUARD_HOME", str(tmp_path / "home"))


@pytest.fixture
def profile():
    rng = np.random.default_rng(0)
    names = ("shoulder_ahead", "neck_incline", "head_pitch", "face_over_neck",
             "ear_shoulder_hip", "trunk_incline")
    good = rng.normal([0.05, 3.0, 5.0, 1.2, 160.0, 2.0], 0.02, (40, 6))
    bad = rng.normal([0.45, 20.0, -8.0, 0.9, 140.0, 8.0], 0.02, (40, 6))
    return fit_profile("side", names, good, bad)


class Recorder:
    """Minimal alerter that just remembers what it was told."""

    name = "recorder"

    def __init__(self, fail_on_apply=False):
        self.ticks: list[Tick] = []
        self.started = self.stopped = False
        self.fail_on_apply = fail_on_apply

    def start(self):
        self.started = True

    def apply(self, tick):
        if self.fail_on_apply:
            raise RuntimeError("display went away")
        self.ticks.append(tick)

    def stop(self):
        self.stopped = True


class TestShared:
    def test_an_unset_slot_reads_as_infinitely_old(self):
        tick, age = Shared().read()
        assert tick is None
        assert age == float("inf")

    def test_publishing_resets_the_age(self):
        shared = Shared()
        shared.publish(Tick(ts=1.0, state=State.CALM, intensity=0.3, score=0.2))
        tick, age = shared.read()
        assert tick.intensity == 0.3
        assert age < 0.5


class TestWatchdog:
    def test_a_stale_worker_clears_the_alerters(self, profile, tmp_path):
        """The core safety property: no fresh score means no dim, whatever went wrong."""
        cfg = Config(alerters=[])
        with Store(tmp_path / "t.db") as store:
            runner = Runner(cfg, profile, store, tmp_path / "model.task", log=lambda _: None)
            recorder = Recorder()
            runner.alerters = [SafeAlerter(recorder)]

            # Worker published a full-intensity alert, then stopped publishing.
            runner.shared.publish(Tick(ts=1.0, state=State.ALERT, intensity=1.0, score=0.9))
            runner.shared._updated_at -= 30.0

            runner.pump()
            assert recorder.ticks[-1].intensity == 0.0
            assert recorder.ticks[-1].state is State.ABSENT

    def test_a_fresh_tick_is_passed_through(self, profile, tmp_path):
        cfg = Config(alerters=[])
        with Store(tmp_path / "t.db") as store:
            runner = Runner(cfg, profile, store, tmp_path / "model.task", log=lambda _: None)
            recorder = Recorder()
            runner.alerters = [SafeAlerter(recorder)]
            runner.shared.publish(Tick(ts=time.time(), state=State.ALERT, intensity=0.7, score=0.9))
            runner.pump()
            assert recorder.ticks[-1].intensity == pytest.approx(0.7)


class TestSafeAlerter:
    def test_a_failing_alerter_is_disabled_not_propagated(self):
        messages = []
        safe = SafeAlerter(Recorder(fail_on_apply=True), on_error=messages.append)
        safe.start()
        safe.apply(Tick(ts=0.0, state=State.ALERT, intensity=1.0, score=0.9))
        assert safe.failed
        assert "disabled" in messages[0]
        safe.apply(Tick(ts=1.0, state=State.ALERT, intensity=1.0, score=0.9))  # stays quiet

    def test_a_healthy_alerter_is_left_alone(self):
        recorder = Recorder()
        safe = SafeAlerter(recorder)
        safe.start()
        safe.apply(Tick(ts=0.0, state=State.CALM, intensity=0.0, score=0.1))
        safe.stop()
        assert recorder.started and recorder.stopped
        assert not safe.failed


class TestPolicyFromRatchet:
    def test_a_tightened_threshold_keeps_the_hysteresis_gap(self, profile, tmp_path):
        cfg = Config(enter=0.55, exit=0.35)
        with Store(tmp_path / "t.db") as store:
            store.set_state("ratchet_enter", "0.30")  # now below the configured exit
            runner = Runner(cfg, profile, store, tmp_path / "model.task", log=lambda _: None)
            assert runner.worker.policy.enter == 0.30
            assert runner.worker.policy.exit < 0.30


class TestPauseFlag:
    def test_write_read_clear(self):
        assert _read_pause_flag() is None
        until = time.time() + 600
        _write_pause_flag(until)
        assert _read_pause_flag() == pytest.approx(until)
        _clear_pause_flag()
        assert _read_pause_flag() is None

    def test_a_corrupt_flag_is_ignored(self):
        from posture_guard.config import pause_flag_path

        path = pause_flag_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not a number")
        assert _read_pause_flag() is None


class TestTrayTitle:
    def test_the_glyph_grows_with_the_score(self):
        assert title_for(State.CALM, 0.0) == "▁"
        assert title_for(State.ALERT, 1.0) == "█"
        assert title_for(State.CALM, 0.5) not in ("▁", "█")

    def test_absent_and_paused_have_their_own_marks(self):
        assert title_for(State.ABSENT, None) == "·"
        assert title_for(State.CALM, 0.2, paused_for=300) == "‖6m"

    def test_out_of_range_scores_are_clamped(self):
        assert title_for(State.ALERT, 5.0) == "█"
        assert title_for(State.CALM, -3.0) == "▁"
