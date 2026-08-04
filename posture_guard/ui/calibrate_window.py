"""A calibration window: live camera, an instruction, a bar, and one button.

The terminal flow asks you to hold a pose at a camera you cannot see, and only
tells you afterwards that every frame was rejected. This shows the picture while
you pose, says in plain words what is wrong with the current frame, and counts
usable frames as they arrive -- so a badly placed camera is something you fix
during the countdown rather than discover twelve seconds too late.

Every decision lives in ``calibration_flow``. This file draws a
:class:`~posture_guard.calibration_flow.View` and forwards two button presses.

Threading: AppKit owns the main thread, so the camera runs on a worker that
keeps only its newest frame and a main-thread timer collects it. Same
arrangement as the monitor, for the same reason.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from ..calibration_flow import CalibrationSession, Phase
from ..config import Config
from ..features import get_feature_set

WINDOW_W, WINDOW_H = 720, 640
PREVIEW_H = 320
PREVIEW_FPS = 12.0
UI_HZ = 15.0

# Windows opened from the menu bar outlive this function, and nothing on the
# Python side would otherwise hold a reference to them.
_OPEN: list = []


class _Camera(threading.Thread):
    """Grabs frames continuously, keeping only the newest."""

    daemon = True

    def __init__(self, cfg: Config, model: Path):
        super().__init__(name="calibration-camera")
        self.cfg = cfg
        self.model = model
        self.stop_event = threading.Event()
        self.error: str | None = None
        self._lock = threading.Lock()
        self._latest = None

    def run(self) -> None:
        import cv2

        from ..capture import PoseSource

        source = PoseSource(
            self.model,
            camera_index=self.cfg.camera_index,
            width=self.cfg.camera_width,
            height=self.cfg.camera_height,
            fps=PREVIEW_FPS,
            keep_frame=True,
        )
        try:
            with source:
                for frame in source.frames():
                    if self.stop_event.is_set():
                        return
                    image = source.last_frame
                    encoded = None
                    if image is not None:
                        # Mirrored: posing in front of an unmirrored image of
                        # yourself is disorienting.
                        ok, buffer = cv2.imencode(
                            ".jpg", cv2.flip(image, 1), [cv2.IMWRITE_JPEG_QUALITY, 70]
                        )
                        encoded = buffer.tobytes() if ok else None
                    with self._lock:
                        self._latest = (frame, encoded)
        except Exception as exc:  # noqa: BLE001 - shown in the window instead
            self.error = str(exc)

    def take(self):
        with self._lock:
            return self._latest

    def stop(self) -> None:
        self.stop_event.set()


def run_calibration_window(cfg: Config, model: Path, on_saved=None) -> bool:
    """Open the window and block until it closes. True when a profile was saved."""
    import AppKit
    from Foundation import NSData, NSMakeRect, NSObject, NSTimer

    feature_set = get_feature_set(cfg.view)
    session = CalibrationSession(feature_set, cfg.quality_limits())
    camera = _Camera(cfg, model)
    saved = {"value": False}

    def const(*names, default=0):
        """Constants that have been renamed across macOS versions."""
        for name in names:
            if hasattr(AppKit, name):
                return getattr(AppKit, name)
        return default

    def label(rect, size, bold=False, grey=False, lines=1):
        field = AppKit.NSTextField.alloc().initWithFrame_(rect)
        field.setBezeled_(False)
        field.setDrawsBackground_(False)
        field.setEditable_(False)
        field.setSelectable_(True)
        field.setFont_(
            AppKit.NSFont.boldSystemFontOfSize_(size)
            if bold
            else AppKit.NSFont.systemFontOfSize_(size)
        )
        if grey:
            field.setTextColor_(AppKit.NSColor.secondaryLabelColor())
        if lines != 1:
            field.setUsesSingleLineMode_(False)
            field.cell().setWraps_(True)
        return field

    class Controller(NSObject):
        """Target for the buttons, the timer and the window delegate.

        No custom init: an Objective-C subclass would need ``objc.super`` rather
        than Python's ``super``, and there is nothing here worth that footgun.
        Attributes are set after alloc/init instead.
        """

        # -- buttons ---------------------------------------------------------

        def primary_(self, sender):
            if session.phase is Phase.REVIEW and session.profile is not None:
                self._save()
            else:
                session.press(time.time())

        def secondary_(self, sender):
            if session.phase in (Phase.REVIEW, Phase.FAILED):
                session.restart()
            else:
                self.window.performClose_(None)

        def _save(self):
            from .. import config as cfgmod
            from ..storage import Store

            profile = session.profile
            if profile is None:
                return
            cfgmod.profile_path().parent.mkdir(parents=True, exist_ok=True)
            cfgmod.profile_path().write_text(profile.to_json())
            with Store(cfgmod.database_path()) as store:
                store.save_profile(profile.view, profile.to_json(), profile.created)
                store.log_event(profile.created, "calibrated", {"view": profile.view})
            saved["value"] = True
            if on_saved:
                on_saved(profile)
            self.window.performClose_(None)

        # -- window ----------------------------------------------------------

        def windowWillClose_(self, notification):
            camera.stop()
            if self.standalone:
                AppKit.NSApplication.sharedApplication().stop_(None)

        # -- the loop --------------------------------------------------------

        def tick_(self, timer):
            latest = camera.take()
            if latest is not None:
                frame, jpeg = latest
                session.offer(frame)
                if jpeg:
                    image = AppKit.NSImage.alloc().initWithData_(
                        NSData.dataWithBytes_length_(jpeg, len(jpeg))
                    )
                    if image is not None:
                        self.preview.setImage_(image)

            view = session.tick(time.time())
            self.step.setStringValue_(view.step)
            self.heading.setStringValue_(view.heading)
            self.detail.setStringValue_(view.detail)

            if camera.error:
                hint = f"Camera stopped: {camera.error}"
                ok = False
            elif view.countdown is not None:
                hint, ok = f"Starting in {view.countdown}…    {view.hint}", view.hint_ok
            elif view.phase is Phase.CAPTURING:
                hint = f"{view.hint}    {view.accepted} usable frames so far"
                ok = view.hint_ok
            else:
                hint, ok = view.hint, view.hint_ok
            self.hint.setStringValue_(hint)
            self.hint.setTextColor_(
                AppKit.NSColor.secondaryLabelColor() if ok else AppKit.NSColor.systemOrangeColor()
            )

            self.bar.setDoubleValue_(view.progress * 100.0)
            self.primary.setHidden_(view.button is None)
            if view.button:
                self.primary.setTitle_(view.button)
            self.secondary.setHidden_(view.secondary is None)
            if view.secondary:
                self.secondary.setTitle_(view.secondary)

    app = AppKit.NSApplication.sharedApplication()
    # rumps has already started a run loop when this is opened from the menu
    # bar; starting a second one would deadlock. Detect which case we are in.
    standalone = not app.isRunning()

    style = const("NSWindowStyleMaskTitled", default=1) | const(
        "NSWindowStyleMaskClosable", default=2
    )
    window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        NSMakeRect(0, 0, WINDOW_W, WINDOW_H), style, AppKit.NSBackingStoreBuffered, False
    )
    window.setTitle_("posture-guard — calibration")
    window.setReleasedWhenClosed_(False)
    window.center()

    controller = Controller.alloc().init()
    controller.window = window
    controller.standalone = standalone
    window.setDelegate_(controller)

    content = window.contentView()
    top = WINDOW_H

    preview = AppKit.NSImageView.alloc().initWithFrame_(
        NSMakeRect(20, top - PREVIEW_H - 20, WINDOW_W - 40, PREVIEW_H)
    )
    preview.setImageScaling_(const("NSImageScaleProportionallyUpOrDown", default=3))
    content.addSubview_(preview)

    y = top - PREVIEW_H - 50
    step = label(NSMakeRect(22, y, WINDOW_W - 44, 16), 11, grey=True)
    y -= 42
    heading = label(NSMakeRect(20, y, WINDOW_W - 40, 30), 22, bold=True)
    y -= 62
    detail = label(NSMakeRect(22, y, WINDOW_W - 44, 56), 13, lines=3)
    y -= 46
    hint = label(NSMakeRect(22, y, WINDOW_W - 44, 38), 12, grey=True, lines=2)
    y -= 28
    bar = AppKit.NSProgressIndicator.alloc().initWithFrame_(
        NSMakeRect(22, y, WINDOW_W - 44, 10)
    )
    bar.setIndeterminate_(False)
    bar.setMinValue_(0.0)
    bar.setMaxValue_(100.0)
    for widget in (step, heading, detail, hint, bar):
        content.addSubview_(widget)

    rounded = const("NSBezelStyleRounded", "NSRoundedBezelStyle", default=1)
    primary = AppKit.NSButton.alloc().initWithFrame_(NSMakeRect(WINDOW_W - 190, 22, 170, 32))
    primary.setBezelStyle_(rounded)
    primary.setTarget_(controller)
    primary.setAction_("primary:")
    primary.setKeyEquivalent_("\r")
    secondary = AppKit.NSButton.alloc().initWithFrame_(NSMakeRect(WINDOW_W - 350, 22, 150, 32))
    secondary.setBezelStyle_(rounded)
    secondary.setTarget_(controller)
    secondary.setAction_("secondary:")
    for widget in (primary, secondary):
        content.addSubview_(widget)

    controller.preview = preview
    controller.step = step
    controller.heading = heading
    controller.detail = detail
    controller.hint = hint
    controller.bar = bar
    controller.primary = primary
    controller.secondary = secondary

    timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
        1.0 / UI_HZ, controller, "tick:", None, True
    )

    if standalone:
        # Started from a shell there is no bundle to inherit an activation
        # policy from, and without one the window opens behind everything or
        # not at all.
        app.setActivationPolicy_(const("NSApplicationActivationPolicyRegular", default=0))

    camera.start()
    window.makeKeyAndOrderFront_(None)
    app.activateIgnoringOtherApps_(True)

    if not standalone:
        # The menu bar's run loop drives the timer; keep the pieces alive.
        controller.timer = timer
        _OPEN.append((window, controller, camera))
        return False

    try:
        app.run()
    finally:
        timer.invalidate()
        camera.stop()
        camera.join(timeout=2.0)
    return saved["value"]
