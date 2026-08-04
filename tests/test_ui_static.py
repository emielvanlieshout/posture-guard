"""Static checks on the AppKit code, which cannot be executed off macOS.

Running the window is impossible here, so the next best thing is to enforce the
PyObjC rules that a normal test run would have caught. Each of these stands for
a mistake that reached the user's machine before it was noticed.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

UI = Path(__file__).resolve().parent.parent / "posture_guard" / "ui"
MODULES = sorted(UI.glob("*.py"))


def objc_classes(tree: ast.AST) -> list[ast.ClassDef]:
    """Classes deriving from an Objective-C base, by name."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        bases = [b.id if isinstance(b, ast.Name) else getattr(b, "attr", "") for b in node.bases]
        if any(base.startswith("NS") for base in bases):
            found.append(node)
    return found


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_objc_subclasses_do_not_call_plain_super(path):
    """`super()` in an Objective-C subclass raises at runtime; objc.super is needed.

    Simply not overriding init at all is better still, which is what these do.
    """
    tree = ast.parse(path.read_text())
    for cls in objc_classes(tree):
        for node in ast.walk(cls):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "super"
            ):
                pytest.fail(
                    f"{path.name}:{node.lineno} {cls.name} calls plain super(); "
                    "use objc.super or do not override the method"
                )


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_appkit_is_never_imported_at_module_level(path):
    """Importing AppKit at the top would make the package unimportable elsewhere."""
    tree = ast.parse(path.read_text())
    for node in tree.body:
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for name in names:
            root = name.split(".")[0]
            assert root not in {"AppKit", "Foundation", "Quartz", "objc", "rumps"}, (
                f"{path.name} imports {root} at module level; defer it into the function"
            )


def test_selectors_that_appkit_calls_are_spelled_correctly():
    """A method wired up with setAction_ has to exist, with the trailing underscore."""
    source = (UI / "calibrate_window.py").read_text()
    tree = ast.parse(source)

    wired = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"setAction_"}
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            wired.add(str(node.args[0].value))

    defined = {
        node.name
        for cls in objc_classes(tree)
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
    }
    assert wired, "no actions found; this test would pass vacuously"
    for selector in wired:
        assert selector.replace(":", "_") in defined, f"setAction_({selector!r}) has no method"


def test_the_timer_selector_exists():
    source = (UI / "calibrate_window.py").read_text()
    assert '"tick:"' in source
    assert "def tick_(" in source


def test_the_window_delegate_method_exists():
    """Without it, closing the standalone window leaves the run loop spinning."""
    source = (UI / "calibrate_window.py").read_text()
    assert "setDelegate_" in source
    assert "def windowWillClose_(" in source
