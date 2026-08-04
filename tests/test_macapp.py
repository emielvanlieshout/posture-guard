"""The .app bundle.

Nothing here needs macOS: a bundle is a directory with a plist in it, and the
plist keys are the whole point. Whether Finder likes the result is something
only a Mac can answer, but whether the keys that make it behave like an app are
present is checkable anywhere.
"""

from __future__ import annotations

import plistlib
import stat

import pytest

from posture_guard.macapp import (
    BUNDLE_ID,
    BUNDLE_NAME,
    BundleError,
    build_app,
    default_destination,
    login_item_script,
)


@pytest.fixture
def bundle(tmp_path):
    return build_app(tmp_path / "PostureGuard.app", python="/opt/venv/bin/python")


def read_plist(bundle):
    with (bundle / "Contents" / "Info.plist").open("rb") as handle:
        return plistlib.load(handle)


class TestStructure:
    def test_it_looks_like_a_bundle(self, bundle):
        assert bundle.is_dir()
        assert (bundle / "Contents" / "Info.plist").is_file()
        assert (bundle / "Contents" / "MacOS" / BUNDLE_NAME).is_file()
        assert (bundle / "Contents" / "Resources").is_dir()

    def test_the_launcher_is_executable(self, bundle):
        mode = (bundle / "Contents" / "MacOS" / BUNDLE_NAME).stat().st_mode
        assert mode & stat.S_IXUSR

    def test_the_launcher_uses_an_absolute_interpreter(self, bundle):
        """Finder gives an app almost no PATH, so a bare `python` would not resolve."""
        script = (bundle / "Contents" / "MacOS" / BUNDLE_NAME).read_text()
        assert "/opt/venv/bin/python" in script
        assert "-m posture_guard run" in script

    def test_it_logs_somewhere_findable(self, bundle):
        script = (bundle / "Contents" / "MacOS" / BUNDLE_NAME).read_text()
        assert "Library/Logs/posture-guard.log" in script

    def test_a_missing_extension_is_added(self, tmp_path):
        bundle = build_app(tmp_path / "somewhere", python="/usr/bin/python3")
        assert bundle.name == f"{BUNDLE_NAME}.app"

    def test_appending_the_suffix_is_not_doubled(self, tmp_path):
        bundle = build_app(tmp_path / "X.app", python="/usr/bin/python3")
        assert bundle.name == "X.app"


class TestInfoPlist:
    def test_no_dock_icon(self, bundle):
        """A menu bar tool with a Dock icon is a menu bar tool done wrong."""
        assert read_plist(bundle)["LSUIElement"] is True

    def test_the_camera_prompt_can_explain_itself(self, bundle):
        reason = read_plist(bundle)["NSCameraUsageDescription"]
        assert "camera" in reason.lower()
        assert "discarded" in reason or "nothing is recorded" in reason

    def test_continuity_cameras_are_declared(self, bundle):
        """The key macOS asks for in that deprecation warning."""
        assert read_plist(bundle)["NSCameraUseContinuityCameraDeviceType"] is True

    def test_identity_and_executable_line_up(self, bundle):
        plist = read_plist(bundle)
        assert plist["CFBundleIdentifier"] == BUNDLE_ID
        assert plist["CFBundleExecutable"] == BUNDLE_NAME
        assert (bundle / "Contents" / "MacOS" / plist["CFBundleExecutable"]).exists()

    def test_it_carries_a_version(self, bundle):
        from posture_guard import __version__

        assert read_plist(bundle)["CFBundleShortVersionString"] == __version__


class TestRebuilding:
    def test_rebuilding_replaces_cleanly(self, tmp_path):
        first = build_app(tmp_path / "A.app", python="/one/python")
        (first / "Contents" / "stale.txt").write_text("left over")
        second = build_app(tmp_path / "A.app", python="/two/python")
        assert not (second / "Contents" / "stale.txt").exists()
        assert "/two/python" in (second / "Contents" / "MacOS" / BUNDLE_NAME).read_text()

    def test_it_refuses_to_delete_something_that_is_not_a_bundle(self, tmp_path):
        target = tmp_path / "Precious.app"
        target.mkdir()
        (target / "important.txt").write_text("do not delete me")
        with pytest.raises(BundleError, match="not an app bundle"):
            build_app(target)
        assert (target / "important.txt").exists()


class TestLoginItem:
    def test_adding_names_the_bundle(self, bundle):
        script = login_item_script(bundle, add=True)
        assert str(bundle) in script
        assert "make login item" in script
        assert "hidden:true" in script

    def test_removing_uses_the_name(self, bundle):
        assert "delete login item" in login_item_script(bundle, add=False)


def test_the_default_home_needs_no_admin_rights():
    assert "Applications" in str(default_destination())
    assert str(default_destination()).startswith(str(__import__("pathlib").Path.home()))


class TestFatalErrorsAreVisible:
    """Launched from Finder there is no terminal, so stderr goes to a log file
    nobody thinks to open and the app simply appears not to start."""

    def test_a_message_is_shown_when_there_is_no_terminal(self, monkeypatch):
        from posture_guard import cli

        calls = []
        monkeypatch.setattr(cli.sys, "platform", "darwin")
        monkeypatch.setattr(cli.sys.stderr, "isatty", lambda: False, raising=False)
        monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/osascript")
        monkeypatch.setattr("subprocess.run", lambda cmd, **kw: calls.append(cmd))

        cli._surface("pose model missing")
        assert calls, "nothing was shown"
        script = calls[0][-1]
        assert "display alert" in script
        assert "pose model missing" in script

    def test_a_terminal_is_left_alone(self, monkeypatch):
        from posture_guard import cli

        calls = []
        monkeypatch.setattr(cli.sys, "platform", "darwin")
        monkeypatch.setattr(cli.sys.stderr, "isatty", lambda: True, raising=False)
        monkeypatch.setattr("subprocess.run", lambda cmd, **kw: calls.append(cmd))

        cli._surface("already printed to stderr")
        assert not calls, "would be shown twice"

    def test_quotes_cannot_break_out_of_the_script(self, monkeypatch):
        from posture_guard import cli

        calls = []
        monkeypatch.setattr(cli.sys, "platform", "darwin")
        monkeypatch.setattr(cli.sys.stderr, "isatty", lambda: False, raising=False)
        monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/osascript")
        monkeypatch.setattr("subprocess.run", lambda cmd, **kw: calls.append(cmd))

        cli._surface('he said "no" \\ and left')
        script = calls[0][-1]
        assert script.count('"') % 2 == 0
        assert '\\"no\\"' in script


class TestLauncherDebuggability:
    """An app that will not start and an app that started invisibly look the
    same from outside, so the launcher has to make them distinguishable."""

    def test_a_terminal_gets_the_output_not_the_log(self, bundle):
        script = (bundle / "Contents" / "MacOS" / BUNDLE_NAME).read_text()
        assert "[ -t 1 ]" in script, "must notice it is being run by hand"
        interactive = script.split("[ -t 1 ]")[1].split("fi")[0]
        assert ">>" not in interactive, "redirecting would hide the very thing being looked for"

    def test_finder_gets_the_log_with_a_dated_marker(self, bundle):
        script = (bundle / "Contents" / "MacOS" / BUNDLE_NAME).read_text()
        assert "starting ===" in script
        assert 'date ' in script


class TestDescribeInstall:
    def test_a_missing_bundle_says_how_to_get_one(self, tmp_path):
        from posture_guard.macapp import describe_install

        lines = describe_install(tmp_path / "nothing.app")
        assert "not installed" in lines[0]
        assert "install-app" in lines[0]

    def test_it_reports_the_interpreter(self, bundle):
        from posture_guard.macapp import describe_install

        text = "\n".join(describe_install(bundle))
        assert "/opt/venv/bin/python" in text

    def test_a_vanished_virtualenv_is_called_out(self, bundle):
        from posture_guard.macapp import describe_install

        text = "\n".join(describe_install(bundle))
        assert "MISSING" in text, "/opt/venv does not exist here"
        assert "rebuild" in text

    def test_it_points_at_the_launcher_for_a_manual_run(self, bundle):
        from posture_guard.macapp import describe_install

        assert str(bundle / "Contents" / "MacOS" / BUNDLE_NAME) in "\n".join(
            describe_install(bundle)
        )


class TestLauncherOutputIsUsable:
    def test_python_is_unbuffered(self, bundle):
        """Writing to a file, Python block-buffers stdout: a log that only
        appears once the process exits cannot explain why it exited."""
        script = (bundle / "Contents" / "MacOS" / BUNDLE_NAME).read_text()
        assert "PYTHONUNBUFFERED=1" in script
        assert script.index("PYTHONUNBUFFERED") < script.index("exec $RUN")
