"""Architecture mismatches.

A universal Python inherits its architecture from whatever launched it, so the
same virtualenv can work from one shell and fail from another with nothing but
a page of numpy text that never mentions Rosetta. These make the diagnosis, and
the diagnosis itself, survivable.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from posture_guard.arch import describe, explain_import_error, running_arch, under_rosetta

CLI = Path(__file__).resolve().parent.parent / "posture_guard" / "cli.py"

REAL_NUMPY_ERROR = """
Original error was: dlopen(/Users/x/.venv/lib/python3.11/site-packages/numpy/_core/
_multiarray_umath.cpython-311-darwin.so, 0x0002): tried: '...' (mach-o file, but is
an incompatible architecture (have 'arm64', need 'x86_64'))
"""


class TestDetection:
    def test_the_running_architecture_is_reported(self):
        assert running_arch() in {"arm64", "x86_64", "aarch64", "amd64", "arm"}

    def test_describe_mentions_rosetta_only_when_translated(self):
        text = describe()
        assert running_arch() in text
        if not under_rosetta():
            assert "Rosetta" not in text

    def test_rosetta_detection_is_false_off_macos(self, monkeypatch):
        monkeypatch.setattr("sys.platform", "linux")
        assert under_rosetta() is False


class TestExplanation:
    def test_the_real_numpy_error_is_recognised(self):
        explanation = explain_import_error(ImportError(REAL_NUMPY_ERROR))
        assert explanation is not None
        assert "Nothing is corrupt" in explanation
        assert "Rosetta" in explanation
        assert "arch -" in explanation, "must give the command that proves it"

    def test_it_names_the_architecture_to_switch_to(self, monkeypatch):
        monkeypatch.setattr("posture_guard.arch.running_arch", lambda: "x86_64")
        assert "arch -arm64" in explain_import_error(ImportError(REAL_NUMPY_ERROR))

        monkeypatch.setattr("posture_guard.arch.running_arch", lambda: "arm64")
        assert "arch -x86_64" in explain_import_error(ImportError(REAL_NUMPY_ERROR))

    def test_an_ordinary_missing_module_is_left_alone(self):
        assert explain_import_error(ImportError("No module named 'rumps'")) is None


class TestDiagnosticsSurviveABrokenEnvironment:
    """doctor and app-log are what you reach for when the environment is broken.

    They died on the same numpy import that broke it, which is how a wall of
    text about C extensions became the answer to "why will the app not start".
    """

    NUMPY_DEPENDENT = {"numpy", "calibration", "features", "landmarks", "scoring", "synth"}

    def test_the_cli_module_imports_nothing_that_needs_numpy(self):
        tree = ast.parse(CLI.read_text())
        offending = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                offending += [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                offending.append(node.module.split(".")[0])
        assert not (set(offending) & self.NUMPY_DEPENDENT), (
            f"cli.py imports {set(offending) & self.NUMPY_DEPENDENT} at module level; "
            "defer it into the command that needs it"
        )

    def test_the_cli_can_be_imported_without_numpy(self, monkeypatch):
        import builtins
        import importlib
        import sys

        real_import = builtins.__import__

        def no_numpy(name, *args, **kwargs):
            if name.split(".")[0] == "numpy":
                raise ImportError("mach-o file, but is an incompatible architecture")
            return real_import(name, *args, **kwargs)

        for module in [m for m in sys.modules if m.startswith("posture_guard")]:
            monkeypatch.delitem(sys.modules, module, raising=False)
        monkeypatch.setattr(builtins, "__import__", no_numpy)

        cli = importlib.import_module("posture_guard.cli")
        assert hasattr(cli, "main")

    def test_doctor_and_app_log_do_not_touch_numpy_dependent_modules(self):
        """Whatever else changes, these two have to keep working."""
        source = CLI.read_text()
        tree = ast.parse(source)
        for name in ("cmd_doctor", "cmd_app_log"):
            func = next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == name
            )
            imported = set()
            for node in ast.walk(func):
                if isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".")[0])
                elif isinstance(node, ast.Import):
                    imported |= {a.name.split(".")[0] for a in node.names}
            # doctor imports numpy deliberately, inside a try, to report on it.
            allowed = {"numpy"} if name == "cmd_doctor" else set()
            assert not (imported & self.NUMPY_DEPENDENT - allowed), (
                f"{name} imports {imported & self.NUMPY_DEPENDENT - allowed}"
            )


class TestLauncherPinning:
    def test_the_launcher_pins_the_architecture(self, tmp_path):
        from posture_guard.macapp import build_app

        bundle = build_app(tmp_path / "P.app", python="/opt/py", arch="arm64")
        script = (bundle / "Contents" / "MacOS" / "PostureGuard").read_text()
        assert 'ARCH="arm64"' in script
        assert "/usr/bin/arch -$ARCH" in script

    def test_it_still_runs_where_arch_is_unavailable(self, tmp_path):
        from posture_guard.macapp import build_app

        bundle = build_app(tmp_path / "P.app", python="/opt/py", arch="arm64")
        script = (bundle / "Contents" / "MacOS" / "PostureGuard").read_text()
        assert 'RUN="$PY"' in script, "must fall back rather than fail"

    def test_the_pin_defaults_to_what_is_working_now(self, tmp_path):
        import platform

        from posture_guard.macapp import build_app

        bundle = build_app(tmp_path / "P.app", python="/opt/py")
        script = (bundle / "Contents" / "MacOS" / "PostureGuard").read_text()
        assert f'ARCH="{platform.machine()}"' in script

    def test_doctor_reports_the_pin(self, tmp_path):
        from posture_guard.macapp import build_app, describe_install

        bundle = build_app(tmp_path / "P.app", python="/opt/py", arch="x86_64")
        assert "pinned to x86_64" in "\n".join(describe_install(bundle))
