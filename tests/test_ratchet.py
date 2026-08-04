from __future__ import annotations

import time

import pytest

from posture_guard.config import Config
from posture_guard.ratchet import EVENT_KIND, current_enter, decide, evaluate
from posture_guard.storage import Store


@pytest.fixture
def cfg():
    return Config()


def call(cfg, *, enter=0.55, good=0.9, hours=20.0, days=8.0):
    return decide(
        cfg, enter=enter, good_fraction=good, measured_hours=hours, days_since_change=days
    )


class TestDecision:
    def test_a_good_week_tightens_the_threshold(self, cfg):
        decision = call(cfg, good=0.9)
        assert decision.changed
        assert decision.new_enter < decision.old_enter
        assert "tightening" in decision.reason

    def test_a_bad_week_eases_off(self, cfg):
        decision = call(cfg, good=0.4)
        assert decision.changed
        assert decision.new_enter > decision.old_enter
        assert "easing off" in decision.reason

    def test_the_middle_holds(self, cfg):
        decision = call(cfg, good=0.7)
        assert not decision.changed
        assert "holding" in decision.reason

    def test_it_waits_a_full_week(self, cfg):
        decision = call(cfg, good=0.95, days=3.0)
        assert not decision.changed
        assert "days since the last change" in decision.reason

    def test_it_needs_enough_measured_time(self, cfg):
        decision = call(cfg, good=0.95, hours=2.0)
        assert not decision.changed
        assert "measured" in decision.reason

    def test_no_data_means_no_change(self, cfg):
        assert not call(cfg, good=None).changed

    def test_it_stops_at_the_strict_bound(self, cfg):
        decision = call(cfg, enter=cfg.ratchet_min_enter, good=0.99)
        assert not decision.changed
        assert "tightest" in decision.reason

    def test_it_stops_at_the_loose_bound(self, cfg):
        decision = call(cfg, enter=cfg.ratchet_max_enter, good=0.1)
        assert not decision.changed
        assert "loosest" in decision.reason

    def test_disabling_it_works(self, cfg):
        cfg.ratchet_enabled = False
        assert not call(cfg, good=0.99).changed

    def test_repeated_good_weeks_keep_tightening(self, cfg):
        enter = cfg.enter
        for _ in range(6):
            decision = call(cfg, enter=enter, good=0.95)
            if not decision.changed:
                break
            enter = decision.new_enter
        assert enter < cfg.enter * 0.7
        assert enter >= cfg.ratchet_min_enter


class TestPersistence:
    def test_it_starts_from_the_configured_threshold(self, tmp_path, cfg):
        with Store(tmp_path / "t.db") as store:
            assert current_enter(store, cfg) == cfg.enter

    def test_a_change_is_stored_and_logged(self, tmp_path, cfg):
        now = time.time()
        with Store(tmp_path / "t.db") as store:
            # Ten days of solid good posture, well past every gate.
            for i in range(6 * 60 * 8 * 7):
                ts = now - 7 * 86400 + i * 30
                store.add_bucket(
                    int(ts // 30) * 30,
                    n_frames=180, n_valid=180, score_sum=180 * 0.2, score_max=0.3,
                    secs_good=30.0, secs_bad=0.0, secs_absent=0.0, secs_alert=0.0,
                )
            store.set_state("ratchet_changed_ts", str(now - 10 * 86400))

            decision = evaluate(store, cfg, now=now)
            assert decision.changed
            assert current_enter(store, cfg) == decision.new_enter

            events = store.events_between(now - 60, now + 60)
            assert [e["kind"] for e in events] == [EVENT_KIND]

    def test_a_stored_value_is_clamped_to_the_bounds(self, tmp_path, cfg):
        with Store(tmp_path / "t.db") as store:
            store.set_state("ratchet_enter", "0.01")
            assert current_enter(store, cfg) == cfg.ratchet_min_enter
            store.set_state("ratchet_enter", "nonsense")
            assert current_enter(store, cfg) == cfg.enter

    def test_an_empty_history_changes_nothing(self, tmp_path, cfg):
        with Store(tmp_path / "t.db") as store:
            assert not evaluate(store, cfg).changed
