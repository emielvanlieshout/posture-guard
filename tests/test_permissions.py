"""Camera authorisation handling.

AVFoundation does not exist off macOS, so the framework itself is stubbed. What
is under test is the logic around it: that statuses map correctly, that a
refusal produces instructions rather than a probe, and that everything degrades
to "just try the camera" where the framework is absent.
"""

from __future__ import annotations

import pytest

from posture_guard import permissions
from posture_guard.permissions import (
    AUTHORIZED,
    DENIED,
    NOT_DETERMINED,
    RESTRICTED,
    UNAVAILABLE,
    camera_status,
    describe,
    list_video_devices,
    request_camera_access,
    responsible_app,
)

MEDIA_TYPE = "vide"


class FakeDevice:
    def __init__(self, name):
        self._name = name

    def localizedName(self):  # noqa: N802 - mirrors the Objective-C selector
        return self._name


class FakeAVCaptureDevice:
    """Stands in for AVCaptureDevice, including its request-then-answer dance."""

    def __init__(self, status_code=0, grant=True, devices=(), explode=False):
        self.status_code = status_code
        self.grant = grant
        self.devices = [FakeDevice(n) for n in devices]
        self.explode = explode
        self.requests = 0

    def authorizationStatusForMediaType_(self, media_type):  # noqa: N802
        if self.explode:
            raise RuntimeError("objc bridge blew up")
        return self.status_code

    def requestAccessForMediaType_completionHandler_(self, media_type, handler):  # noqa: N802
        self.requests += 1
        self.status_code = 3 if self.grant else 2
        handler(self.grant)

    def devicesWithMediaType_(self, media_type):  # noqa: N802
        if self.explode:
            raise RuntimeError("objc bridge blew up")
        return self.devices


@pytest.fixture
def fake(monkeypatch):
    def install(**kwargs):
        device = FakeAVCaptureDevice(**kwargs)
        monkeypatch.setattr(permissions, "_avcapture", lambda: (device, MEDIA_TYPE))
        return device

    return install


class TestWithoutAvFoundation:
    """A Linux box, or a Mac missing the bindings, must still be able to try."""

    def test_status_is_unavailable(self, monkeypatch):
        monkeypatch.setattr(permissions, "_avcapture", lambda: None)
        assert camera_status() == UNAVAILABLE

    def test_requesting_is_a_no_op(self, monkeypatch):
        monkeypatch.setattr(permissions, "_avcapture", lambda: None)
        assert request_camera_access() == UNAVAILABLE

    def test_no_devices_are_listed(self, monkeypatch):
        monkeypatch.setattr(permissions, "_avcapture", lambda: None)
        assert list_video_devices() == []

    def test_unavailable_suggests_the_bindings(self):
        assert "pyobjc-framework-AVFoundation" in describe(UNAVAILABLE)


class TestStatus:
    @pytest.mark.parametrize(
        "code,expected",
        [(0, NOT_DETERMINED), (1, RESTRICTED), (2, DENIED), (3, AUTHORIZED)],
    )
    def test_codes_map_to_names(self, fake, code, expected):
        fake(status_code=code)
        assert camera_status() == expected

    def test_an_unknown_code_is_not_guessed_at(self, fake):
        fake(status_code=99)
        assert camera_status() == UNAVAILABLE

    def test_a_broken_bridge_does_not_crash_the_diagnostic(self, fake):
        fake(explode=True)
        assert camera_status() == UNAVAILABLE
        assert list_video_devices() == []


class TestRequesting:
    def test_an_unanswered_prompt_is_raised(self, fake):
        device = fake(status_code=0, grant=True)
        assert request_camera_access() == AUTHORIZED
        assert device.requests == 1

    def test_a_refusal_comes_back_as_denied(self, fake):
        device = fake(status_code=0, grant=False)
        assert request_camera_access() == DENIED
        assert device.requests == 1

    @pytest.mark.parametrize("code,expected", [(2, DENIED), (3, AUTHORIZED), (1, RESTRICTED)])
    def test_an_answered_question_is_not_asked_again(self, fake, code, expected):
        """macOS only ever prompts once; re-asking would be a silent no-op."""
        device = fake(status_code=code)
        assert request_camera_access() == expected
        assert device.requests == 0


class TestGuidance:
    def test_authorised_says_nothing(self):
        assert describe(AUTHORIZED) == ""

    def test_denied_names_the_app_and_the_restart(self, monkeypatch):
        monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
        text = describe(DENIED)
        assert "iTerm" in text
        assert "Privacy & Security" in text
        assert "Cmd-Q" in text, "the restart is the step everyone skips"
        assert "already running" in text

    def test_not_determined_points_at_doctor(self):
        assert "posture-guard doctor" in describe(NOT_DETERMINED)

    def test_restricted_blames_policy_not_the_user(self):
        text = describe(RESTRICTED)
        assert "Screen Time" in text or "device-management" in text

    @pytest.mark.parametrize(
        "term,expected",
        [
            ("Apple_Terminal", "Terminal"),
            ("iTerm.app", "iTerm"),
            ("vscode", "Visual Studio Code"),
            ("WarpTerminal", "Warp"),
        ],
    )
    def test_known_terminals_are_named(self, monkeypatch, term, expected):
        monkeypatch.setenv("TERM_PROGRAM", term)
        assert responsible_app() == expected

    def test_an_unknown_terminal_falls_back_gracefully(self, monkeypatch):
        monkeypatch.delenv("TERM_PROGRAM", raising=False)
        assert responsible_app() == "your terminal"


class TestDevices:
    def test_devices_are_listed_with_indices(self, fake):
        fake(status_code=3, devices=["FaceTime HD Camera", "Emiel's iPhone"])
        assert list_video_devices() == [(0, "FaceTime HD Camera"), (1, "Emiel's iPhone")]


class TestProbing:
    def test_probing_is_skipped_when_access_is_refused(self, monkeypatch):
        """Otherwise doctor prints a backend error per index and learns nothing."""
        from posture_guard import capture

        monkeypatch.setattr(permissions, "camera_status", lambda: DENIED)

        def explode(*args, **kwargs):
            raise AssertionError("must not touch cv2 when access is denied")

        monkeypatch.setattr("cv2.VideoCapture", explode, raising=False)
        assert capture.list_cameras() == []

    def test_probing_still_happens_where_the_status_is_unknown(self, monkeypatch):
        from posture_guard import capture

        monkeypatch.setattr(permissions, "camera_status", lambda: UNAVAILABLE)
        tried = []

        class FakeCapture:
            def __init__(self, index, *args):
                tried.append(index)

            def isOpened(self):  # noqa: N802
                return False

            def read(self):
                return False, None

            def release(self):
                pass

        monkeypatch.setattr("cv2.VideoCapture", FakeCapture, raising=False)
        assert capture.list_cameras(limit=3) == []
        assert tried == [0, 1, 2]

    def test_only_the_known_indices_are_probed(self, monkeypatch):
        """Probing past the last real device prints an alarming non-error."""
        from posture_guard import capture

        monkeypatch.setattr(permissions, "camera_status", lambda: AUTHORIZED)
        tried = []

        class FakeCapture:
            def __init__(self, index, *args):
                tried.append(index)

            def isOpened(self):  # noqa: N802
                return True

            def read(self):
                return True, object()

            def release(self):
                pass

        monkeypatch.setattr("cv2.VideoCapture", FakeCapture, raising=False)
        assert capture.list_cameras(indices=[0, 1]) == [0, 1]
        assert tried == [0, 1], "must not wander past what the system reported"


class TestCliGate:
    def test_a_denied_camera_exits_with_instructions(self, monkeypatch):
        from posture_guard import cli

        monkeypatch.setattr(permissions, "camera_status", lambda: DENIED)
        with pytest.raises(SystemExit) as caught:
            cli._require_camera_permission()
        assert "Privacy & Security" in str(caught.value)

    @pytest.mark.parametrize("status", [AUTHORIZED, UNAVAILABLE])
    def test_a_usable_camera_passes_through(self, monkeypatch, status):
        from posture_guard import cli

        monkeypatch.setattr(permissions, "camera_status", lambda: status)
        cli._require_camera_permission()

    def test_an_unasked_permission_triggers_the_prompt(self, monkeypatch, capsys):
        from posture_guard import cli

        monkeypatch.setattr(permissions, "camera_status", lambda: NOT_DETERMINED)
        monkeypatch.setattr(permissions, "request_camera_access", lambda *a, **k: AUTHORIZED)
        cli._require_camera_permission()
        assert "approve the prompt" in capsys.readouterr().out
