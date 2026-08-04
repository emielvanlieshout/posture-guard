from __future__ import annotations

import time

import pytest

from posture_guard.config import Config
from posture_guard.report import render_html, render_text
from posture_guard.stats import daily_stats, trend, window_stats
from posture_guard.storage import BucketAggregator, Store


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "posture.db") as s:
        yield s


class TestBuckets:
    def test_writes_that_land_in_one_slot_are_merged(self, store):
        for _ in range(3):
            store.add_bucket(
                1000, n_frames=10, n_valid=8, score_sum=4.0, score_max=0.6,
                secs_good=20.0, secs_bad=10.0, secs_absent=0.0, secs_alert=5.0,
            )
        buckets = store.buckets_between(0, 2000)
        assert len(buckets) == 1
        assert buckets[0].n_frames == 30
        assert buckets[0].secs_good == pytest.approx(60.0)
        assert buckets[0].score_max == pytest.approx(0.6), "max, not a sum"
        assert buckets[0].score_mean == pytest.approx(0.5)

    def test_range_queries_are_half_open(self, store):
        for ts in (100, 200, 300):
            store.add_bucket(
                ts, n_frames=1, n_valid=1, score_sum=0.5, score_max=0.5,
                secs_good=30.0, secs_bad=0.0, secs_absent=0.0, secs_alert=0.0,
            )
        assert [b.bucket_ts for b in store.buckets_between(100, 300)] == [100, 200]

    def test_an_empty_bucket_reports_no_mean(self, store):
        store.add_bucket(
            0, n_frames=5, n_valid=0, score_sum=0.0, score_max=0.0,
            secs_good=0.0, secs_bad=0.0, secs_absent=150.0, secs_alert=0.0,
        )
        assert store.buckets_between(-1, 1)[0].score_mean is None


class TestAggregator:
    def test_it_splits_on_bucket_boundaries(self, store):
        agg = BucketAggregator(store, bucket_seconds=30, good_below=0.5)
        for i in range(6):  # 1200..1250, straddling the boundary at 1230
            agg.add(1200 + i * 10, 0.2, dt=10.0)
        agg.flush()
        assert [b.bucket_ts for b in store.buckets_between(1100, 1400)] == [1200, 1230]

    def test_good_and_bad_are_split_at_the_threshold(self, store):
        agg = BucketAggregator(store, bucket_seconds=30, good_below=0.5)
        agg.add(0, 0.2, dt=10.0)
        agg.add(1, 0.8, dt=10.0)
        agg.add(2, None, dt=10.0)
        agg.flush()
        bucket = store.buckets_between(-1, 60)[0]
        assert bucket.secs_good == pytest.approx(10.0)
        assert bucket.secs_bad == pytest.approx(10.0)
        assert bucket.secs_absent == pytest.approx(10.0)
        assert bucket.n_valid == 2

    def test_a_sleep_gap_does_not_inflate_the_day(self, store):
        agg = BucketAggregator(store, bucket_seconds=30, good_below=0.5)
        agg.add(0, 0.2, dt=8 * 3600)  # laptop lid was shut for eight hours
        agg.flush()
        bucket = store.buckets_between(-1, 60)[0]
        assert bucket.secs_good <= 60.0, "a gap must be clamped, not counted as desk time"

    def test_flushing_nothing_is_harmless(self, store):
        BucketAggregator(store, 30).flush()
        assert store.buckets_between(0, time.time() + 1) == []


class TestStats:
    def _fill(self, store, *, days=3, good_ratio=0.6, start=None):
        start = start or (time.time() - days * 86400)
        for day in range(days):
            for i in range(120):  # one hour of 30s buckets
                ts = start + day * 86400 + i * 30
                good = 30.0 if i < 120 * good_ratio else 0.0
                store.add_bucket(
                    int(ts // 30) * 30,
                    n_frames=180, n_valid=180, score_sum=180 * 0.4, score_max=0.9,
                    secs_good=good, secs_bad=30.0 - good, secs_absent=0.0, secs_alert=0.0,
                )

    def test_window_stats_add_up(self, store):
        self._fill(store, days=2, good_ratio=0.75)
        stats = window_stats(store, time.time() - 3 * 86400, time.time() + 1)
        assert stats.measured_hours == pytest.approx(2.0, abs=0.05)
        assert stats.good_fraction == pytest.approx(0.75, abs=0.02)
        assert stats.mean_score == pytest.approx(0.4, abs=0.01)

    def test_no_data_gives_none_not_zero(self, store):
        stats = window_stats(store, 0, 100)
        assert stats.good_fraction is None
        assert stats.mean_score is None
        assert stats.alerts_per_hour is None

    def test_alerts_are_counted_from_events(self, store):
        now = time.time()
        for i in range(4):
            store.log_event(now - i * 60, "alert_started", {})
        store.log_event(now, "returned", {})
        assert window_stats(store, now - 3600, now + 1).alerts == 4

    def test_longest_run_stops_at_a_gap(self, store):
        for i in list(range(4)) + list(range(20, 22)):
            store.add_bucket(
                i * 30, n_frames=1, n_valid=1, score_sum=0.1, score_max=0.1,
                secs_good=30.0, secs_bad=0.0, secs_absent=0.0, secs_alert=0.0,
            )
        stats = window_stats(store, 0, 10_000)
        assert stats.longest_good_minutes == pytest.approx(2.0), "4 buckets, not 6"

    def test_daily_stats_cover_every_day_including_empty_ones(self, store):
        self._fill(store, days=2)
        history = daily_stats(store, days=5)
        assert len(history) == 5
        assert history[0].day < history[-1].day
        assert any(d.stats.measured_hours == 0 for d in history)


class TestTrend:
    def test_a_falling_series_has_a_negative_slope(self):
        assert trend([0.8, 0.7, 0.6, 0.5]) < 0

    def test_gaps_are_skipped(self):
        assert trend([0.8, None, 0.6, None, 0.4]) < 0

    def test_too_few_points_gives_none(self):
        assert trend([0.5, None, None]) is None


class TestReports:
    def test_text_report_survives_an_empty_database(self, store):
        text = render_text(store, Config(), days=7)
        assert "posture-guard history" in text
        assert "n/a" in text

    def test_html_is_self_contained(self, store):
        store.add_bucket(
            int(time.time() // 30) * 30,
            n_frames=10, n_valid=10, score_sum=3.0, score_max=0.5,
            secs_good=25.0, secs_bad=5.0, secs_absent=0.0, secs_alert=0.0,
        )
        html = render_html(store, Config(), days=7)
        assert html.startswith("<!doctype html>")
        for forbidden in ("http://", "https://", "<script"):
            assert forbidden not in html, f"report must not reference {forbidden}"

    def test_threshold_changes_show_up(self, store):
        now = time.time()
        store.log_event(
            now, "threshold_changed", {"from": 0.55, "to": 0.506, "reason": "90% good, tightening"}
        )
        text = render_text(store, Config(), days=7)
        assert "0.550 -> 0.506" in text
        assert "tightening" in text
