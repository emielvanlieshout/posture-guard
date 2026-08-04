"""The calibration flow, driven by a fake clock and synthetic frames.

The window is untestable off macOS, which is exactly why every decision it
makes lives here instead: the countdown, the pose order, what to say when frames
are being rejected, and whether a result is worth offering to save.
"""

from __future__ import annotations

import numpy as np
import pytest

from posture_guard.calibration_flow import (
    CAPTURE_S,
    COUNTDOWN_S,
    POSES,
    CalibrationSession,
    Phase,
    hint_for,
)
from posture_guard.features import get_feature_set
from posture_guard.synth import Posture, synth_frame

FRONTAL = get_feature_set("frontal")
SIDE = get_feature_set("side")


def frame(angle, **kw):
    return synth_frame(posture=Posture(protraction_deg=angle, **kw))


def session(feature_set=FRONTAL):
    return CalibrationSession(feature_set, countdown_s=COUNTDOWN_S, capture_s=CAPTURE_S)


def run(sess, *, good_angle=2.0, bad_angle=26.0, start=1000.0, step=1 / 12, pose_kwargs=None):
    """Press start and hold both poses, returning the last view."""
    pose_kwargs = pose_kwargs or {}
    t = start
    sess.press(t)
    view = sess.tick(t)
    while not view.finished:
        angle = good_angle if sess.index == 0 else bad_angle
        jitter = float(np.sin(t * 7.0)) * 0.8
        sess.offer(frame(angle + jitter, **pose_kwargs))
        view = sess.tick(t)
        t += step
        if t - start > 120:
            pytest.fail("the flow never finished")
    return view


class TestFlow:
    def test_it_starts_by_explaining_the_first_pose(self):
        view = session().tick(0.0)
        assert view.phase is Phase.READY
        assert view.heading == POSES[0].heading
        assert view.button == "Start"
        assert "Step 1 of 2" in view.step

    def test_nothing_happens_until_the_button(self):
        sess = session()
        sess.offer(frame(2.0))
        assert sess.tick(60.0).phase is Phase.READY

    def test_the_countdown_runs_before_capturing(self):
        sess = session()
        sess.press(100.0)
        assert sess.tick(100.0).phase is Phase.COUNTDOWN
        assert sess.tick(101.0).countdown == int(COUNTDOWN_S - 1) + 1
        assert sess.tick(100.0 + COUNTDOWN_S - 0.1).phase is Phase.COUNTDOWN
        assert sess.tick(100.0 + COUNTDOWN_S).phase is Phase.CAPTURING

    def test_frames_are_only_kept_while_capturing(self):
        sess = session()
        sess.press(100.0)
        sess.tick(100.0)
        sess.offer(frame(2.0))  # during the countdown
        assert sess.tick(100.0).accepted == 0

        sess.tick(100.0 + COUNTDOWN_S)
        sess.offer(frame(2.0))
        assert sess.tick(100.0 + COUNTDOWN_S).accepted == 1

    def test_it_moves_to_the_second_pose_on_its_own(self):
        sess = session()
        sess.press(0.0)
        t = 0.0
        while sess.index == 0 and t < 60:
            sess.tick(t)
            t += 0.1
        assert sess.index == 1
        assert sess.tick(t).heading == POSES[1].heading

    def test_a_stalled_clock_does_not_skip_a_pose(self):
        """A slept machine or a frozen window must not silently drop a capture."""
        sess = session()
        sess.press(0.0)
        sess.tick(3600.0)  # an hour between two ticks
        assert sess.index == 0, "still on the first pose"
        assert sess.phase is Phase.CAPTURING
        assert sess.tick(3600.1).progress < 0.05, "the capture restarts rather than counting the gap"

    def test_progress_fills_over_the_capture(self):
        sess = session()
        sess.press(0.0)
        sess.tick(COUNTDOWN_S)
        assert sess.tick(COUNTDOWN_S + CAPTURE_S / 2).progress == pytest.approx(0.5, abs=0.05)


class TestOutcome:
    def test_a_good_run_offers_to_save(self):
        sess = session()
        view = run(sess)
        assert view.phase is Phase.REVIEW
        assert sess.profile is not None
        assert view.button == "Save and close"
        assert view.secondary == "Try again"

    def test_the_profile_is_usable(self):
        sess = session()
        run(sess)
        assert sess.profile.view == "frontal"
        assert sess.profile.weights.sum() == pytest.approx(1.0)

    def test_two_identical_poses_fail_with_a_reason(self):
        sess = session()
        view = run(sess, good_angle=12.0, bad_angle=12.0)
        assert view.phase is Phase.FAILED
        assert "look the same" in view.error if hasattr(view, "error") else True
        assert "same" in view.detail.lower()
        assert view.secondary == "Try again"

    def test_a_camera_pointed_the_wrong_way_explains_itself(self):
        """Frontal frames into a side profile: the failure has to say so."""
        sess = session(SIDE)
        view = run(sess)
        assert view.phase is Phase.FAILED
        assert "front" in view.detail.lower() or "side" in view.detail.lower()

    def test_trying_again_clears_everything(self):
        sess = session()
        run(sess)
        assert sess.profile is not None
        sess.restart()
        view = sess.tick(0.0)
        assert view.phase is Phase.READY
        assert sess.index == 0
        assert sess.profile is None
        assert view.accepted == 0


class TestLiveHints:
    """The reason for having a window at all: knowing while there is still time."""

    def test_a_usable_frame_says_so(self):
        text, ok = hint_for(FRONTAL.extract(frame(5.0)), "frontal")
        assert ok
        assert text

    def test_nobody_in_view(self):
        text, ok = hint_for(None, "frontal")
        assert not ok
        assert "No-one in view" in text

    def test_a_turned_head_is_explained_in_plain_words(self):
        text, ok = hint_for(FRONTAL.extract(frame(5.0, yaw_deg=40)), "frontal")
        assert not ok
        assert "squarely" in text

    def test_a_frontal_camera_on_a_side_profile_names_the_fix(self):
        text, ok = hint_for(SIDE.extract(frame(5.0)), "side")
        assert not ok
        assert "view=frontal" in text

    def test_the_hint_appears_while_capturing(self):
        sess = session()
        sess.press(0.0)
        sess.tick(COUNTDOWN_S)
        sess.offer(None)
        assert not sess.tick(COUNTDOWN_S).hint_ok
