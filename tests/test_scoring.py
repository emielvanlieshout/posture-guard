from __future__ import annotations

import numpy as np
import pytest

from posture_guard.scoring import (
    CalibrationError,
    Profile,
    Scorer,
    fit_profile,
    score,
)

NAMES = ("a", "b", "c")


def make(good_means, bad_means, n=40, spread=0.05, seed=0):
    rng = np.random.default_rng(seed)
    good = rng.normal(good_means, spread, (n, len(good_means)))
    bad = rng.normal(bad_means, spread, (n, len(bad_means)))
    return fit_profile("side", NAMES, good, bad)


class TestFitting:
    def test_separating_features_get_the_weight(self):
        # 'a' separates by a mile, 'b' a little, 'c' not at all.
        profile = make([10.0, 2.0, 5.0], [5.0, 1.9, 5.0])
        assert profile.weights[0] > profile.weights[1] > profile.weights[2]
        assert profile.weights[2] == 0.0
        assert profile.weights.sum() == pytest.approx(1.0)

    def test_identical_postures_are_refused(self):
        with pytest.raises(CalibrationError, match="look the same"):
            make([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])

    def test_too_few_frames_is_refused(self):
        with pytest.raises(CalibrationError, match="not enough usable frames"):
            fit_profile("side", NAMES, np.ones((3, 3)), np.zeros((3, 3)))

    def test_a_feature_that_is_mostly_nan_is_dropped(self):
        rng = np.random.default_rng(1)
        good = rng.normal([10.0, 2.0, 5.0], 0.05, (40, 3))
        bad = rng.normal([5.0, 1.0, 5.0], 0.05, (40, 3))
        good[:35, 1] = np.nan  # 'b' available in only 12% of frames
        bad[:35, 1] = np.nan
        profile = fit_profile("side", NAMES, good, bad)
        assert profile.weights[1] == 0.0

    def test_noise_floor_stops_a_constant_feature_dominating(self):
        good = np.tile([1.0, 5.0, 5.0], (40, 1))
        bad = np.tile([1.0001, 5.0, 5.0], (40, 1))
        # 'a' moves a hair with zero variance; without a noise floor that would
        # read as infinite discriminability.
        with pytest.raises(CalibrationError):
            fit_profile("side", NAMES, good, bad)

    def test_mismatched_shape_is_rejected(self):
        with pytest.raises(ValueError, match="do not match"):
            fit_profile("side", NAMES, np.ones((10, 2)), np.zeros((10, 2)))


class TestScoring:
    def test_anchors_land_on_zero_and_one(self):
        profile = make([10.0, 2.0, 5.0], [5.0, 1.0, 5.0])
        assert score(profile, np.array([10.0, 2.0, 5.0])) == pytest.approx(0.0, abs=0.05)
        assert score(profile, np.array([5.0, 1.0, 5.0])) == pytest.approx(1.0, abs=0.05)

    def test_halfway_scores_halfway(self):
        profile = make([10.0, 2.0, 5.0], [5.0, 1.0, 5.0])
        assert score(profile, np.array([7.5, 1.5, 5.0])) == pytest.approx(0.5, abs=0.05)

    def test_beyond_the_anchors_is_allowed_but_bounded(self):
        profile = make([10.0, 2.0, 5.0], [5.0, 1.0, 5.0])
        assert score(profile, np.array([100.0, 50.0, 5.0])) == pytest.approx(-0.5, abs=1e-6)
        assert score(profile, np.array([-100.0, -50.0, 5.0])) == pytest.approx(1.5, abs=1e-6)

    def test_missing_features_are_renormalised(self):
        profile = make([10.0, 2.0, 5.0], [5.0, 1.0, 5.0])
        partial = np.array([7.5, np.nan, np.nan])
        assert score(profile, partial) == pytest.approx(0.5, abs=0.05)

    def test_too_little_weight_returns_none_rather_than_a_guess(self):
        rng = np.random.default_rng(2)
        good = rng.normal([10.0, 2.0, 5.0], 0.05, (60, 3))
        bad = rng.normal([2.0, 1.98, 5.0], 0.05, (60, 3))
        profile = fit_profile("side", NAMES, good, bad)
        assert profile.weights[0] > 0.9  # 'a' carries nearly everything
        assert score(profile, np.array([np.nan, 2.0, 5.0])) is None

    def test_all_nan_returns_none(self):
        profile = make([10.0, 2.0, 5.0], [5.0, 1.0, 5.0])
        assert score(profile, np.full(3, np.nan)) is None


class TestSmoothing:
    def test_first_reading_is_taken_as_is(self):
        profile = make([10.0, 2.0, 5.0], [5.0, 1.0, 5.0])
        scorer = Scorer(profile, tau=2.0)
        assert scorer.update(100.0, np.array([5.0, 1.0, 5.0])) == pytest.approx(1.0, abs=0.05)

    def test_it_lags_then_catches_up(self):
        profile = make([10.0, 2.0, 5.0], [5.0, 1.0, 5.0])
        scorer = Scorer(profile, tau=2.0)
        scorer.update(0.0, np.array([10.0, 2.0, 5.0]))
        after_one = scorer.update(1.0, np.array([5.0, 1.0, 5.0]))
        assert 0.2 < after_one < 0.7, "one time constant in, roughly half way"

        t = 1.0
        for _ in range(30):
            t += 0.5
            value = scorer.update(t, np.array([5.0, 1.0, 5.0]))
        assert value == pytest.approx(1.0, abs=0.05)

    def test_unscoreable_frames_do_not_move_the_average(self):
        profile = make([10.0, 2.0, 5.0], [5.0, 1.0, 5.0])
        scorer = Scorer(profile, tau=2.0)
        scorer.update(0.0, np.array([10.0, 2.0, 5.0]))
        assert scorer.update(1.0, np.full(3, np.nan)) is None
        assert scorer.value == pytest.approx(0.0, abs=0.05)


class TestSerialisation:
    def test_round_trip(self):
        profile = make([10.0, 2.0, 5.0], [5.0, 1.0, 5.0])
        restored = Profile.from_json(profile.to_json())
        assert restored.names == profile.names
        assert restored.view == profile.view
        np.testing.assert_allclose(restored.weights, profile.weights)
        np.testing.assert_allclose(restored.good_med, profile.good_med)

    def test_nan_survives_the_round_trip(self):
        profile = make([10.0, 2.0, 5.0], [5.0, 1.0, 5.0])
        profile.good_med[2] = np.nan
        restored = Profile.from_json(profile.to_json())
        assert np.isnan(restored.good_med[2])

    def test_a_future_version_is_refused_loudly(self):
        profile = make([10.0, 2.0, 5.0], [5.0, 1.0, 5.0])
        raw = profile.to_json().replace('"version": 1', '"version": 99')
        with pytest.raises(ValueError, match="calibrate"):
            Profile.from_json(raw)

    def test_describe_mentions_every_feature(self):
        profile = make([10.0, 2.0, 5.0], [5.0, 1.0, 5.0])
        text = profile.describe()
        for name in NAMES:
            assert name in text
