"""Feature behaviour against the synthetic torso.

The first test is the one that shaped the whole design: it pins down that a
frontal camera at laptop distance cannot see isolated protraction, so nobody
later "fixes" the side view back out of the project.
"""

from __future__ import annotations

import numpy as np
import pytest

from posture_guard.features import (
    FRONTAL_FEATURES,
    SIDE_FEATURES,
    QualityLimits,
    extract_frontal,
    extract_side,
    get_feature_set,
)
from posture_guard.synth import Camera, Posture, synth_frame

STILL = dict(rise_coupling=0.0, forward_coupling=0.0, pitch_coupling=0.0)


def frontal(angle, **kw):
    return extract_frontal(synth_frame(posture=Posture(protraction_deg=angle, **kw)))


def side(angle, camera=None, hips=0.25, **kw):
    return extract_side(
        synth_frame(
            posture=Posture(protraction_deg=angle, yaw_deg=90.0, **kw),
            camera=camera,
            hip_visibility=hips,
        )
    )


# Indices into SIDE_FEATURES.
SHOULDER_AHEAD, NECK_INCLINE, TRUNK_INCLINE, HEAD_OVER_HIP = 0, 1, 5, 6


class TestFrontalLimits:
    def test_isolated_protraction_is_nearly_invisible_head_on(self):
        """Perspective magnification cancels the cosine narrowing at desk distance.

        Rotating the shoulders forward shrinks their projected width by cos(theta)
        but also brings them closer to the lens. Below roughly 17 degrees the
        second effect wins, so the ratio rises before it falls and a neutral
        posture reads much like a badly slouched one.
        """
        ratios = [frontal(a, **STILL).values[0] for a in (0, 10, 20, 30, 35)]

        assert ratios[1] > ratios[0], "should widen first, not narrow"
        assert max(ratios) == pytest.approx(ratios[2], rel=0.02), "peak sits near 20 degrees"
        assert abs(ratios[-1] - ratios[0]) / ratios[0] < 0.03, (
            "0 and 35 degrees end up within 3% of each other: not a usable signal"
        )

    def test_a_distant_camera_would_see_it(self):
        """Same geometry at 2.5 m is monotone, which confirms the cause is perspective."""
        far = [
            extract_frontal(
                synth_frame(posture=Posture(protraction_deg=a, **STILL), camera=Camera(distance=2.5))
            ).values[0]
            for a in (0, 10, 20, 30)
        ]
        assert all(b < a for a, b in zip(far, far[1:]))

    def test_the_slouch_complex_is_visible(self):
        """With the usual couplings the companion features separate cleanly."""
        neutral, slouched = frontal(0), frontal(25)
        assert slouched.values[1] > neutral.values[1] * 1.2
        assert slouched.values[3] < neutral.values[3] - 0.2


class TestSideView:
    def test_shoulder_offset_is_monotonic(self):
        values = [side(a).values[0] for a in (0, 5, 10, 15, 20, 25, 30)]
        assert all(b > a for a, b in zip(values, values[1:]))
        assert values[0] == pytest.approx(0.0, abs=0.02)
        assert values[-1] > 0.4

    def test_neck_incline_tracks_the_angle(self):
        angles = [side(a).values[1] for a in (0, 10, 20, 30)]
        assert all(b > a for a, b in zip(angles, angles[1:]))

    def test_survives_a_change_of_camera_distance(self):
        far = side(20, camera=Camera(distance=0.9)).values[0]
        near = side(20, camera=Camera(distance=0.5)).values[0]
        assert abs(far - near) / max(near, 1e-6) < 0.25

    def test_facing_direction_does_not_matter(self):
        left = side(20).values[0]
        right = extract_side(
            synth_frame(posture=Posture(protraction_deg=20, yaw_deg=-90.0))
        ).values[0]
        assert left == pytest.approx(right, abs=0.02)

    def test_rejects_a_frontal_pose(self):
        sample = extract_side(synth_frame(posture=Posture(protraction_deg=10)))
        assert not sample.ok
        assert "side view" in sample.reason


class TestProtractionVersusHead:
    """The reason the hips have to be in frame.

    Forward head posture moves the ear the same way protraction moves the
    shoulder, so anything measured between the two conflates them. Anything
    measured against the pelvis does not.
    """

    def test_ear_referenced_features_invert_when_the_head_leads(self):
        pure = side(30, **STILL).values[SHOULDER_AHEAD]
        led = side(30, head_forward_m=0.18, **STILL).values[SHOULDER_AHEAD]

        assert pure > 0.9, "shoulders 30 degrees forward, head still: reads strongly forward"
        assert led < -0.5, (
            "same shoulders, head further forward still: the sign flips and the feature "
            "now claims the shoulders are back"
        )

    def test_the_two_motions_nearly_cancel_in_a_realistic_slouch(self):
        pure = side(30, **STILL).values[SHOULDER_AHEAD]
        together = side(30, head_forward_m=0.09, **STILL).values[SHOULDER_AHEAD]
        assert together < pure / 5, "most of the signal is eaten by the head moving too"

    def test_trunk_incline_ignores_the_head_entirely(self):
        still = side(20, hips=0.95, **STILL).values[TRUNK_INCLINE]
        craning = side(20, hips=0.95, head_forward_m=0.12, **STILL).values[TRUNK_INCLINE]
        assert still == pytest.approx(craning, abs=0.01)
        assert still > 5.0, "and it still responds to the shoulders"

    def test_head_over_hip_ignores_the_shoulders_entirely(self):
        values = [side(a, hips=0.95, **STILL).values[HEAD_OVER_HIP] for a in (0, 15, 30)]
        assert values[0] == pytest.approx(values[-1], abs=0.01)

    def test_head_over_hip_tracks_the_head(self):
        values = [
            side(0, hips=0.95, head_forward_m=h, **STILL).values[HEAD_OVER_HIP]
            for h in (0.0, 0.04, 0.08, 0.12)
        ]
        assert all(b > a for a, b in zip(values, values[1:]))

    def test_the_pelvis_referenced_pair_is_lost_without_hips(self):
        sample = side(20, hips=0.25)
        assert np.isnan(sample.values[TRUNK_INCLINE])
        assert np.isnan(sample.values[HEAD_OVER_HIP])
        assert np.isfinite(sample.values[SHOULDER_AHEAD]), "the ambiguous ones remain"


class TestQualityGates:
    @pytest.mark.parametrize("yaw", [10, 20, 35, 60])
    def test_turned_head_is_rejected_not_mismeasured(self, yaw):
        assert not frontal(5, yaw_deg=yaw).ok

    @pytest.mark.parametrize("roll", [20, 35])
    def test_tilted_torso_is_rejected(self, roll):
        assert not frontal(5, roll_deg=roll).ok

    def test_small_movements_are_still_accepted(self):
        assert frontal(5, yaw_deg=4, roll_deg=6).ok

    def test_invisible_landmarks_are_rejected(self):
        frame = synth_frame(posture=Posture(protraction_deg=5))
        frame.visibility[:] = 0.1
        assert not extract_frontal(frame).ok

    def test_hip_features_are_nan_when_hips_are_out_of_frame(self):
        sample = frontal(10)
        assert np.isnan(sample.values[4]), "torso_incline needs hips"
        assert np.isnan(sample.values[5]), "shoulder_depth needs hips"

    def test_hip_features_appear_when_hips_are_visible(self):
        frame = synth_frame(posture=Posture(protraction_deg=10), hip_visibility=0.95)
        sample = extract_frontal(frame)
        assert np.isfinite(sample.values[4])
        assert np.isfinite(sample.values[5])

    def test_limits_are_configurable(self):
        strict = QualityLimits(max_yaw=0.05)
        assert not extract_frontal(
            synth_frame(posture=Posture(protraction_deg=5, yaw_deg=4)), strict
        ).ok


class TestFeatureSets:
    def test_registry_matches_the_modules(self):
        assert get_feature_set("side").names == SIDE_FEATURES
        assert get_feature_set("frontal").names == FRONTAL_FEATURES

    def test_unknown_view_is_rejected(self):
        with pytest.raises(ValueError, match="unknown view"):
            get_feature_set("diagonal")

    def test_rejected_samples_carry_no_numbers(self):
        sample = frontal(5, yaw_deg=45)
        assert not sample.ok
        assert np.all(np.isnan(sample.values)), "a rejected frame must not leak values"
