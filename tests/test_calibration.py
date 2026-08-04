from __future__ import annotations

import numpy as np
import pytest

from posture_guard.calibration import build_profile, collect_samples, verdict
from posture_guard.features import get_feature_set
from posture_guard.scoring import CalibrationError, score
from posture_guard.synth import Posture, synth_series

SIDE = get_feature_set("side")
FRONTAL = get_feature_set("frontal")


def series(view, angle, n=60, seed=0, pose=None, **kw):
    base = Posture(yaw_deg=90.0 if view == "side" else 0.0, **(pose or {}))
    # A side camera placed properly has the hip in shot; a laptop webcam does not.
    kw.setdefault("hip_visibility", 0.95 if view == "side" else 0.25)
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

        # With the hips in frame the bulk of the weight should sit on the
        # measurements the head cannot fool.
        pelvis = sum(profile.weights[i] for i in feature_set.hip_dependent)
        assert pelvis > 0.5

        # And the shoulder-versus-pelvis measurement specifically should beat an
        # even split, rather than being drowned by the ambiguous majority.
        assert sum(profile.weights[i] for i in feature_set.primary) > 1 / feature_set.n

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

    def test_a_side_profile_without_hips_is_warned_about(self):
        """Ear-referenced features alone cannot separate protraction from a craning neck."""
        feature_set = SIDE
        good = collect_samples(series("side", 2.0, seed=1, hip_visibility=0.1), feature_set)
        bad = collect_samples(series("side", 26.0, seed=2, hip_visibility=0.1), feature_set)
        profile = build_profile(feature_set, good, bad)

        assert all(profile.weights[i] == 0 for i in feature_set.hip_dependent)
        ok, note = verdict(profile, feature_set)
        assert ok
        assert "hips were not in frame" in note
        assert "chin back" in note


class TestHeadPositionEndToEnd:
    """The question the design has to answer: does it notice the head at all?"""

    def test_forward_head_posture_is_not_mistaken_for_good_posture(self):
        feature_set, _, _, profile = calibrate("side")
        craning = collect_samples(
            series("side", 2.0, seed=9, pose={"head_forward_m": 0.07}), feature_set
        )
        value = float(np.median([score(profile, v) for v in craning.values]))
        assert value > 0.55, f"shoulders fine, head 7cm forward scored {value:.2f}"

    def test_a_tucked_chin_does_not_excuse_forward_shoulders(self):
        feature_set, _, _, profile = calibrate("side")
        tucked = collect_samples(
            series("side", 26.0, seed=10, pose={"head_forward_m": -0.06}), feature_set
        )
        value = float(np.median([score(profile, v) for v in tucked.values]))
        assert value > 0.8, f"shoulders still slouched but chin tucked scored {value:.2f}"

    def test_on_axis_postures_keep_their_smooth_ladder(self):
        """The off-axis penalty must not disturb ordinary slouching."""
        feature_set, _, _, profile = calibrate("side")
        ladder = []
        for angle in (2.0, 8.0, 14.0, 20.0, 26.0):
            samples = collect_samples(series("side", angle, n=40, seed=11), feature_set)
            ladder.append(float(np.median([score(profile, v) for v in samples.values])))
        assert all(b > a for a, b in zip(ladder, ladder[1:])), ladder
        assert ladder[0] < 0.15 and ladder[-1] > 0.85
