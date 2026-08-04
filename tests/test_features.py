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


def side(angle, camera=None, **kw):
    return extract_side(
        synth_frame(posture=Posture(protraction_deg=angle, yaw_deg=90.0, **kw), camera=camera)
    )


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
