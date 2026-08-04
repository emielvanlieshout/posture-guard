from __future__ import annotations

import numpy as np
import pytest

from posture_guard.calibration import build_profile, collect_samples, verdict
from posture_guard.features import get_feature_set
from posture_guard.scoring import CalibrationError, score
from posture_guard.synth import Posture, synth_series

SIDE = get_feature_set("side")
FRONTAL = get_feature_set("frontal")


def series(view, angle, n=60, seed=0, **kw):
    base = Posture(yaw_deg=90.0 if view == "side" else 0.0)
    return synth_series(angle, n=n, seed=seed, posture=base, **kw)


def calibrate(view, good_angle=2.0, bad_angle=26.0):
    feature_set = get_feature_set(view)
    good = collect_samples(series(view, good_angle, seed=1), feature_set)
    bad = collect_samples(series(view, bad_angle, seed=2), feature_set)
    return feature_set, good, bad, build_profile(feature_set, good, bad)


class TestCollection:
    def test_it_keeps_the_usable_frames(self):
        collected = collect_samples(series("side", 10.0), SIDE)
        assert collected.accepted > 40
        assert collected.acceptance > 0.7
        assert collected.values.shape[1] == SIDE.n

    def test_missing_detections_are_counted_not_crashed_on(self):
        frames = list(series("side", 10.0, n=20)) + [None] * 5
        collected = collect_samples(frames, SIDE)
        assert collected.seen == 25
        assert collected.rejected["nobody detected"] == 5

    def test_rejection_reasons_are_reported(self):
        # Frontal frames fed to the side extractor: all rejected, with a reason.
        collected = collect_samples(series("frontal", 10.0, n=20), SIDE)
        assert collected.accepted == 0
        assert "side view" in collected.explain()

    def test_an_empty_run_yields_an_empty_matrix(self):
        collected = collect_samples([], SIDE)
        assert collected.values.shape == (0, SIDE.n)
        assert collected.explain() == "every frame usable"


class TestProfileBuilding:
    def test_the_side_view_calibrates_cleanly(self):
        feature_set, _, _, profile = calibrate("side")
        assert profile.view == "side"
        assert profile.weights.sum() == pytest.approx(1.0)
        # The direct protraction measures should carry most of the weight.
        assert sum(profile.weights[i] for i in feature_set.primary) > 0.4

    def test_the_frontal_view_calibrates_on_the_slouch_complex(self):
        _, _, _, profile = calibrate("frontal")
        assert profile.weights.sum() == pytest.approx(1.0)
        assert np.nanmax(profile.separation) > 1.5

    def test_anchors_land_where_they_should(self):
        _, good, bad, profile = calibrate("side")
        good_score = np.median([score(profile, v) for v in good.values])
        bad_score = np.median([score(profile, v) for v in bad.values])
        assert good_score == pytest.approx(0.0, abs=0.2)
        assert bad_score == pytest.approx(1.0, abs=0.2)

    def test_two_identical_poses_are_refused(self):
        feature_set = SIDE
        good = collect_samples(series("side", 12.0, seed=1), feature_set)
        bad = collect_samples(series("side", 12.0, seed=2), feature_set)
        with pytest.raises(CalibrationError, match="look the same"):
            build_profile(feature_set, good, bad)

    def test_thin_data_names_the_pose_that_failed(self):
        good = collect_samples(series("side", 2.0, n=60, seed=1), SIDE)
        bad = collect_samples([None] * 40, SIDE)
        with pytest.raises(CalibrationError, match="slouched"):
            build_profile(SIDE, good, bad)


class TestVerdict:
    def test_a_strong_side_profile_passes(self):
        feature_set, _, _, profile = calibrate("side")
        ok, note = verdict(profile, feature_set)
        assert ok
        assert "separation" in note

    def test_a_barely_different_pair_is_flagged(self):
        feature_set = SIDE
        good = collect_samples(series("side", 10.0, seed=1), feature_set)
        bad = collect_samples(series("side", 11.0, seed=2), feature_set)
        try:
            profile = build_profile(feature_set, good, bad, min_separation=0.1)
        except CalibrationError:
            return  # refusing outright is an acceptable outcome too
        ok, note = verdict(profile, feature_set)
        assert not ok
        assert "Weak separation" in note

    def test_a_frontal_profile_always_carries_the_caveat(self):
        """However good the numbers look head-on, they are not measuring shoulders."""
        feature_set, _, _, profile = calibrate("frontal")
        ok, note = verdict(profile, feature_set)
        assert ok
        assert "side camera" in note
        assert "perspective hides" in note

    def test_a_side_profile_carries_no_such_caveat(self):
        feature_set, _, _, profile = calibrate("side")
        _, note = verdict(profile, feature_set)
        assert "side camera" not in note
