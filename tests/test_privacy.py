"""The promise this project makes: nothing leaves the machine while it runs.

These tests are the enforcement. If someone later adds an analytics call or a
crash reporter to a runtime module, the suite fails.
"""

from __future__ import annotations

import ast
import socket
import time
from pathlib import Path

import pytest

from posture_guard.calibration import build_profile, collect_samples
from posture_guard.config import Config
from posture_guard.features import get_feature_set
from posture_guard.report import render_html, render_text
from posture_guard.scoring import Scorer
from posture_guard.state import Monitor
from posture_guard.storage import BucketAggregator, Store
from posture_guard.synth import Posture, synth_series

PACKAGE = Path(__file__).resolve().parent.parent / "posture_guard"
NETWORK_MODULES = {"urllib", "http", "requests", "socket", "ftplib", "smtplib", "telnetlib"}
# The one module allowed to reach the network, and only during `setup`.
ALLOWED = {"model.py"}


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return found


@pytest.mark.parametrize(
    "path",
    sorted(p for p in PACKAGE.rglob("*.py") if p.name not in ALLOWED),
    ids=lambda p: str(p.relative_to(PACKAGE)),
)
def test_runtime_modules_do_not_import_networking(path):
    offending = _imported_modules(path) & NETWORK_MODULES
    assert not offending, f"{path.name} imports {offending}; only {ALLOWED} may talk to a network"


def test_the_downloader_is_the_only_exception():
    assert _imported_modules(PACKAGE / "model.py") & NETWORK_MODULES


def test_the_whole_pipeline_runs_with_networking_broken(monkeypatch, tmp_path):
    """Calibrate, score, alert, store and report with every socket refused."""

    def refuse(*args, **kwargs):
        raise AssertionError("posture-guard attempted a network connection")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setenv("POSTURE_GUARD_HOME", str(tmp_path / "home"))

    view = "side"
    feature_set = get_feature_set(view)
    base = Posture(yaw_deg=90.0)
    good = collect_samples(synth_series(2.0, n=60, seed=1, posture=base), feature_set)
    bad = collect_samples(synth_series(26.0, n=60, seed=2, posture=base), feature_set)
    profile = build_profile(feature_set, good, bad)

    scorer = Scorer(profile)
    monitor = Monitor()
    cfg = Config(view=view)
    now = time.time()

    with Store(tmp_path / "posture.db") as store:
        aggregator = BucketAggregator(store, cfg.bucket_seconds, good_below=cfg.enter)
        t = now - 600
        for frame in synth_series(26.0, n=120, seed=3, posture=base):
            sample = feature_set.extract(frame)
            value = scorer.update(t, sample.values) if sample.ok else None
            tick = monitor.update(t, value)
            for event in tick.events:
                store.log_event(event.ts, event.kind.value, event.detail)
            aggregator.add(t, value, dt=1 / 6)
            t += 1 / 6
        aggregator.flush()

        assert "posture-guard history" in render_text(store, cfg, days=2)
        assert render_html(store, cfg, days=2).startswith("<!doctype html>")


def test_no_frame_data_reaches_the_database(tmp_path):
    """A stored row must never be big enough to hold an image."""
    feature_set = get_feature_set("side")
    with Store(tmp_path / "posture.db") as store:
        aggregator = BucketAggregator(store, 30, good_below=0.55)
        for i, frame in enumerate(synth_series(20.0, n=60, posture=Posture(yaw_deg=90.0))):
            sample = feature_set.extract(frame)
            aggregator.add(1000 + i, 0.5 if sample.ok else None, dt=1.0)
        aggregator.flush()

        columns = {row[1] for row in store.conn.execute("PRAGMA table_info(buckets)")}
        assert columns == {
            "bucket_ts", "n_frames", "n_valid", "score_sum", "score_max",
            "secs_good", "secs_bad", "secs_absent", "secs_alert",
        }
        assert store.path.stat().st_size < 200_000


def test_the_html_report_pulls_nothing_from_the_internet(tmp_path):
    with Store(tmp_path / "posture.db") as store:
        store.add_bucket(
            int(time.time() // 30) * 30,
            n_frames=10, n_valid=10, score_sum=3.0, score_max=0.5,
            secs_good=25.0, secs_bad=5.0, secs_absent=0.0, secs_alert=0.0,
        )
        html = render_html(store, Config(), days=7)
    for marker in ("http://", "https://", "<script", "<iframe", "@import", "url("):
        assert marker not in html
