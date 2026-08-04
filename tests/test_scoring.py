from __future__ import annotations

import numpy as np
import pytest

from posture_guard.scoring import (
    CalibrationError,
    Profile,
    Scorer,
    fit_profile,
    score,
    score_parts,
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


class TestOffAxisPostures:
    """A posture nobody demonstrated must not be assumed to be a good one.

    Averaging the features places you on the line between your two calibrated
    poses. Step off that line -- shoulders back but head craning forward -- and
    the features start contradicting each other, one reporting better than
    perfect while another reports fully slouched. The average hides it; the
    spread does not.
    """

    def sep(self):
        return make([10.0, 2.0, 3.0], [5.0, 1.0, 6.0])

    def test_features_that_agree_are_left_alone(self):
        profile = self.sep()
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            values = np.array([10 - 5 * fraction, 2 - fraction, 3 + 3 * fraction])
            parts = score_parts(profile, values)
            assert parts.disagreement < 0.05
            assert parts.penalty == 0.0
            assert parts.value == pytest.approx(parts.axis)

    def test_contradicting_features_push_the_score_up(self):
        profile = self.sep()
        # a and b say "good posture", c says "fully slouched".
        parts = score_parts(profile, np.array([10.0, 2.0, 6.0]))
        assert parts.axis == pytest.approx(1 / 3, abs=0.05)
        assert parts.disagreement > 0.4
        assert parts.penalty > 0.0
        assert parts.value > parts.axis

    def test_the_penalty_cannot_run_away(self):
        profile = self.sep()
        parts = score_parts(profile, np.array([1e6, -1e6, 6.0]))
        assert parts.value <= 1.5

    def test_the_reference_is_measured_during_calibration(self):
        """A user whose features are noisy gets a correspondingly higher bar."""
        rng = np.random.default_rng(3)
        tidy = fit_profile(
            "side", NAMES,
            rng.normal([10.0, 2.0, 3.0], 0.02, (60, 3)),
            rng.normal([5.0, 1.0, 6.0], 0.02, (60, 3)),
        )
        assert tidy.disagreement_ref == pytest.approx(0.35), "floored, never below"

    def test_parts_expose_the_per_feature_verdicts(self):
        profile = self.sep()
        parts = score_parts(profile, np.array([10.0, 2.0, 6.0]))
        assert parts.per_feature.shape == (3,)
        assert parts.per_feature[0] == pytest.approx(0.0, abs=0.05)
        assert parts.per_feature[2] == pytest.approx(1.0, abs=0.05)

    def test_an_unscoreable_frame_has_no_parts(self):
        assert score_parts(self.sep(), np.full(3, np.nan)) is None

    def test_a_feature_set_of_the_wrong_size_is_refused(self):
        with pytest.raises(ValueError, match="recalibrate"):
            score_parts(self.sep(), np.array([1.0, 2.0]))


class TestPrior:
    """Separation says how well a feature splits your poses, not whether it means
    what its name says. The prior is where that second question gets a vote."""

    def test_it_shifts_weight_between_equally_separating_features(self):
        rng = np.random.default_rng(4)
        good = rng.normal([10.0, 10.0, 10.0], 0.05, (40, 3))
        bad = rng.normal([5.0, 5.0, 5.0], 0.05, (40, 3))
        flat = fit_profile("side", NAMES, good, bad)
        weighted = fit_profile("side", NAMES, good, bad, prior=(1.0, 0.5, 0.5))

        assert flat.weights == pytest.approx(flat.weights[0], rel=0.01)
        assert weighted.weights[0] > weighted.weights[1]
        assert weighted.weights.sum() == pytest.approx(1.0)

    def test_a_zero_prior_removes_a_feature(self):
        profile = make([10.0, 2.0, 3.0], [5.0, 1.0, 6.0])
        muted = fit_profile(
            "side", NAMES,
            np.random.default_rng(5).normal([10.0, 2.0, 3.0], 0.05, (40, 3)),
            np.random.default_rng(6).normal([5.0, 1.0, 6.0], 0.05, (40, 3)),
            prior=(1.0, 1.0, 0.0),
        )
        assert profile.weights[2] > 0
        assert muted.weights[2] == 0.0

    def test_a_mismatched_prior_is_rejected(self):
        with pytest.raises(ValueError, match="prior does not match"):
            fit_profile(
                "side", NAMES,
                np.random.default_rng(7).normal([10.0, 2.0, 3.0], 0.05, (40, 3)),
                np.random.default_rng(8).normal([5.0, 1.0, 6.0], 0.05, (40, 3)),
                prior=(1.0, 1.0),
            )


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
        raw = profile.to_json().replace('"version": 2', '"version": 99')
        with pytest.raises(ValueError, match="calibrate"):
            Profile.from_json(raw)

    def test_describe_mentions_every_feature(self):
        profile = make([10.0, 2.0, 5.0], [5.0, 1.0, 5.0])
        text = profile.describe()
        for name in NAMES:
            assert name in text
