"""Command line interface."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import webbrowser
from dataclasses import asdict
from pathlib import Path

from . import __version__
from . import config as cfgmod
from . import permissions
from .config import Config
from .storage import Store

# Everything that pulls numpy is imported inside the command that needs it.
# `doctor` and `app-log` are the commands you reach for when the environment is
# broken, and they must not die on the same import that broke it.

VIEWS = ("side", "frontal")


# --------------------------------------------------------------------------
# helpers


def _load_config() -> Config:
    try:
        return Config.load()
    except (ValueError, OSError) as exc:
        raise SystemExit(f"could not read {cfgmod.config_path()}: {exc}") from None


def _load_profile():
    from .scoring import Profile

    path = cfgmod.profile_path()
    if not path.exists():
        raise SystemExit("no calibration yet. Run `posture-guard calibrate` first.")
    try:
        return Profile.from_json(path.read_text())
    except (ValueError, OSError) as exc:
        raise SystemExit(f"could not read {path}: {exc}") from None


def _require_camera_permission(allow_prompt: bool = True) -> None:
    """Fail with instructions rather than an opaque backend error.

    Called before anything opens a camera. ``allow_prompt=False`` for the
    monitor, which asks later from its own run loop.
    """
    status = permissions.camera_status()
    if status in (permissions.AUTHORIZED, permissions.UNAVAILABLE):
        return
    if status == permissions.NOT_DETERMINED:
        # Not fatal, and deliberately not asked for here. A dialog needs a
        # running application, and `run` has not started one yet; the Runner
        # asks from its own loop instead. The commands that block on a camera
        # ask here because by then they are about to open one.
        if allow_prompt:
            print("asking macOS for camera access, approve the prompt…")
            status = permissions.request_camera_access()
        if status in (permissions.AUTHORIZED, permissions.UNAVAILABLE, permissions.NOT_DETERMINED):
            return
    raise SystemExit(f"camera access is {status}.\n{permissions.describe(status)}")


def _require_model() -> Path:
    from .model import manual_download_hint

    path = cfgmod.model_path()
    if not path.exists():
        raise SystemExit(
            "pose model missing. Run `posture-guard setup` first, or fetch it by hand:\n"
            f"  {manual_download_hint(path)}"
        )
    return path


def _bounded(frames, seconds: float):
    """Take frames until the wall clock runs out."""
    deadline = time.monotonic() + seconds
    for frame in frames:
        yield frame
        if time.monotonic() >= deadline:
            return


def _countdown(message: str, seconds: int) -> None:
    for remaining in range(seconds, 0, -1):
        print(f"\r{message} {remaining}… ", end="", flush=True)
        time.sleep(1)
    print("\rgo!" + " " * 40, flush=True)


# --------------------------------------------------------------------------
# commands


def cmd_setup(args) -> int:
    from .model import ModelDownloadError, ensure_model

    directory = cfgmod.app_dir()
    directory.mkdir(parents=True, exist_ok=True)

    cfg = _load_config()
    if args.view:
        cfg.view = args.view
    if not cfgmod.config_path().exists() or args.view:
        cfg.validate()
        cfg.save()

    print(f"data directory: {directory}")
    print("downloading the pose model (this is the only network request there is)…")
    try:
        path = ensure_model(cfgmod.model_path(), force=args.force)
    except ModelDownloadError as exc:
        print(f"\ndownload failed: {exc}", file=sys.stderr)
        print("\nRe-run `posture-guard setup` once the model is in place.", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"\ndownload failed: {exc}", file=sys.stderr)
        return 1
    print(f"model ready: {path} ({path.stat().st_size / 1e6:.1f} MB)")
    print()
    print("next:")
    print("  posture-guard doctor      check the camera and permissions")
    print(f"  posture-guard calibrate   teach it your two postures ({cfg.view} view)")
    print("  posture-guard run         start monitoring")
    return 0


def cmd_doctor(args) -> int:
    cfg = _load_config()
    # Something being broken and something not being done yet are different
    # answers, and a first run is all of the latter. Reporting "some checks
    # failed" because you have not calibrated is just wrong.
    broken: list[str] = []
    pending: list[str] = []

    from .arch import describe as describe_arch
    from .arch import under_rosetta

    print(f"platform          {sys.platform}")
    print(f"python            {sys.version.split()[0]}")
    print(f"architecture      {describe_arch()}")
    print(f"data directory    {cfgmod.app_dir()}")
    if under_rosetta():
        broken.append("run it natively instead of under Rosetta")

    from .arch import explain_import_error

    for label, module in (("mediapipe", "mediapipe"), ("opencv", "cv2"), ("numpy", "numpy")):
        try:
            mod = __import__(module)
            print(f"{label:<18}{getattr(mod, '__version__', 'installed')}")
        except ImportError as exc:
            explanation = explain_import_error(exc)
            if explanation:
                print(f"{label:<18}WRONG ARCHITECTURE")
                for line in explanation.splitlines():
                    print(f"{'':<18}{line}")
                broken.append("fix the architecture mismatch")
            else:
                print(f"{label:<18}MISSING  (pip install 'posture-guard[macos]')")
                broken.append(f"install {label}")

    if sys.platform == "darwin":
        for label, module in (("rumps", "rumps"), ("pyobjc AppKit", "AppKit")):
            try:
                __import__(module)
                print(f"{label:<18}installed")
            except ImportError:
                print(f"{label:<18}MISSING  (needed for the menu bar and the dim overlay)")
                broken.append(f"install {label}")

    model = cfgmod.model_path()
    if model.exists():
        print(f"{'pose model':<18}{model.stat().st_size / 1e6:.1f} MB")
    else:
        from .model import manual_download_hint

        print(f"{'pose model':<18}MISSING  (run `posture-guard setup`)")
        try:
            import certifi

            print(f"{'':<18}certifi {certifi.__version__} present, TLS should verify")
        except ImportError:
            print(
                f"{'':<18}certifi missing, so the download will fail TLS verification "
                "on a python.org or conda build: pip install certifi"
            )
        print(f"{'':<18}by hand: {manual_download_hint(model)}")
        pending.append("posture-guard setup")

    profile_file = cfgmod.profile_path()
    if profile_file.exists():
        # Read as plain JSON rather than through Profile: parsing it properly
        # needs numpy, and numpy is one of the things that might be broken.
        try:
            payload = json.loads(profile_file.read_text())
            age = (time.time() - float(payload.get("created", 0))) / 86400
            print(f"{'calibration':<18}{payload.get('view', '?')} view, {age:.0f} days old")
            if age > cfg.recalibrate_after_days:
                print(
                    f"{'':<18}older than {cfg.recalibrate_after_days} days — "
                    "recalibrate so the baseline keeps up with you"
                )
        except (ValueError, TypeError) as exc:
            print(f"{'calibration':<18}unreadable: {exc}")
            broken.append("posture-guard calibrate")
    else:
        print(f"{'calibration':<18}not done yet")
        pending.append("posture-guard calibrate")

    if sys.platform == "darwin":
        from .macapp import describe_install

        for i, line in enumerate(describe_install()):
            print(f"{'app bundle' if i == 0 else '':<18}{line}")

    devices: list[tuple[int, str]] = []
    if sys.platform == "darwin":
        # Asking outright beats probing and guessing from the failure. If macOS
        # has not put the question yet, this is where the prompt appears.
        status = permissions.camera_status()
        if status == permissions.NOT_DETERMINED:
            print(f"{'camera access':<18}asking macOS now, approve the prompt…")
            status = permissions.request_camera_access()
        print(f"{'camera access':<18}{status}")
        guidance = permissions.describe(status)
        if guidance:
            if status not in (permissions.AUTHORIZED, permissions.UNAVAILABLE):
                broken.append("grant camera access")
            for line in guidance.splitlines():
                print(f"{'':<18}{line}")

        # Enumerated once and reused: on a Mac with a Continuity Camera this
        # call makes macOS log a deprecation notice, and doing it twice per run
        # would print it twice.
        devices = permissions.list_video_devices()
        for index, name in devices:
            marker = "  <- config uses this" if index == cfg.camera_index else ""
            print(f"{'':<18}[{index}] {name}{marker}")

    try:
        from .capture import list_cameras

        # Probe the devices macOS actually reports rather than a blind range;
        # anything past the end just prints "out device of bound" and worries you.
        known = [index for index, _ in devices]
        cameras = list_cameras(indices=known or None)
        if cameras:
            print(f"{'cameras opened':<18}indices {cameras} (config uses {cfg.camera_index})")
            if cfg.camera_index not in cameras:
                print(f"{'':<18}camera {cfg.camera_index} did not open")
                broken.append(f"pick a camera that opens: config --set camera_index=<n>")
        else:
            print(f"{'cameras opened':<18}none")
            if sys.platform != "darwin":
                print(f"{'':<18}check that a camera is connected and not in use")
            broken.append("get a camera working")
    except ImportError:
        print(f"{'cameras opened':<18}skipped (opencv missing)")
        broken.append("install opencv")

    print()
    if broken:
        print("problems to fix: " + ", ".join(broken))
        return 1
    if pending:
        print("nothing broken. next: " + "  then  ".join(pending))
        return 0
    print("all good")
    return 0


def cmd_calibrate(args) -> int:
    from .calibration import build_profile, collect_samples, verdict
    from .capture import PoseSource
    from .features import get_feature_set
    from .scoring import CalibrationError

    cfg = _load_config()
    view = args.view or cfg.view
    feature_set = get_feature_set(view)
    model = _require_model()
    _require_camera_permission()
    limits = cfg.quality_limits()

    # A window shows you the camera while you pose and says what is wrong with
    # the current frame, instead of reporting twelve seconds later that nothing
    # was usable. The terminal flow stays for headless machines and scripts.
    if not args.terminal and sys.platform == "darwin":
        if cfg.view != view:
            cfg.view = view
            cfg.save()
        try:
            from .ui.calibrate_window import run_calibration_window
        except ImportError as exc:
            print(f"the calibration window needs pyobjc ({exc}); using the terminal flow")
        else:
            saved = run_calibration_window(cfg, model)
            if saved:
                print("calibration saved. Start monitoring with: posture-guard run")
                return 0
            print("calibration was not saved")
            return 1

    print(f"Calibrating the {view} view — {feature_set.blurb}.")
    if view == "side":
        print(
            "Put the camera to one side of you, roughly at shoulder height, so it sees\n"
            "your profile. An iPhone on a stand via Continuity Camera works well.\n"
            "Get your hip in the shot: without it, every measurement is one part of you\n"
            "against another part that also moves, and craning your neck would satisfy\n"
            "the profile with your shoulders untouched."
        )
    else:
        print("Sit as you normally do, facing the camera, whole head and both shoulders in frame.")
    print()

    source = PoseSource(
        model,
        camera_index=args.camera if args.camera is not None else cfg.camera_index,
        width=cfg.camera_width,
        height=cfg.camera_height,
        fps=cfg.fps,
    )

    try:
        with source:
            print("POSE 1 of 2 — good posture.")
            print("  Shoulders rolled back and down, chest open, ears over your shoulders.")
            _countdown("Hold it in", 5)
            good = collect_samples(_bounded(source.frames(), args.seconds), feature_set, limits)
            print(f"  {good.explain()}")
            print()

            print("POSE 2 of 2 — your usual slouch.")
            print("  Let the shoulders roll forward, exactly how you catch yourself sitting.")
            _countdown("Hold it in", 5)
            bad = collect_samples(_bounded(source.frames(), args.seconds), feature_set, limits)
            print(f"  {bad.explain()}")
            print()
    except Exception as exc:  # noqa: BLE001
        print(f"capture failed: {exc}", file=sys.stderr)
        return 1

    try:
        profile = build_profile(
            feature_set, good, bad, min_separation=args.min_separation, limits=limits
        )
    except CalibrationError as exc:
        print(f"calibration failed: {exc}", file=sys.stderr)
        return 1

    print(profile.describe())
    print()
    usable, note = verdict(profile, feature_set)
    print(note)
    if not usable and not args.force:
        print("\nNot saving. Re-run with a clearer difference between the two poses, or --force.")
        return 1

    cfgmod.profile_path().parent.mkdir(parents=True, exist_ok=True)
    cfgmod.profile_path().write_text(profile.to_json())
    if cfg.view != view:
        cfg.view = view
        cfg.save()
    with Store(cfgmod.database_path()) as store:
        store.save_profile(view, profile.to_json(), profile.created)
        store.log_event(profile.created, "calibrated", {"view": view})

    print(f"\nsaved to {cfgmod.profile_path()}")
    print("start monitoring with: posture-guard run")
    return 0


def cmd_run(args) -> int:
    from .app import Runner, run_headless
    from .trace import Trace, announce_running, looks_bundled

    bundled = looks_bundled()
    trace = Trace(verbose=args.debug)
    trace.step(
        f"posture-guard {__version__} starting  pid={os.getpid()}  "
        f"{'from the app bundle' if bundled else 'from a shell'}"
    )
    trace.step(f"python {sys.version.split()[0]} at {sys.executable}")

    cfg = _load_config()
    trace.step(f"config: view={cfg.view} camera={cfg.camera_index} alerters={cfg.alerters}")
    profile = _load_profile()
    trace.step(
        f"calibration: {profile.view} view, "
        f"{(time.time() - profile.created) / 86400:.0f} days old"
    )
    model = _require_model()
    trace.step(f"model: {model.stat().st_size / 1e6:.1f} MB")
    trace.step(f"camera permission: {permissions.camera_status()}")
    _require_camera_permission(allow_prompt=False)

    if profile.view != cfg.view:
        print(
            f"note: config says view={cfg.view} but the calibration is {profile.view}; "
            f"using {profile.view}."
        )

    age_days = (time.time() - profile.created) / 86400
    if age_days > cfg.recalibrate_after_days:
        print(
            f"calibration is {age_days:.0f} days old. Recalibrate soon so your 'good' "
            "reference keeps up with the posture you can now hold."
        )

    store = Store(cfgmod.database_path())
    runner = Runner(cfg, profile, store, model)

    if args.verbose and "status" not in cfg.alerters:
        cfg.alerters = [*cfg.alerters, "status"]

    headless = args.headless or sys.platform != "darwin"
    if headless and "dim" in cfg.alerters and sys.platform != "darwin":
        print("the dim overlay needs macOS; falling back to console alerts")
        cfg.alerters = ["console"]
        runner = Runner(cfg, profile, store, model)

    print(f"monitoring ({profile.view} view, alerters: {', '.join(cfg.alerters) or 'none'})")
    try:
        if headless:
            print("ctrl-c to stop")
            run_headless(runner)
        else:
            from .tray import run_tray

            run_tray(runner, cfg, cfgmod.app_dir() / "report.html")
    finally:
        runner.stop()
        store.close()
    return 0


def cmd_preview(args) -> int:
    """Live view with landmarks and the score, for aiming the camera."""
    import cv2
    import numpy as np

    from .capture import PoseSource
    from .features import get_feature_set
    from .landmarks import LM
    from .scoring import Profile, Scorer, score_parts

    cfg = _load_config()
    model = _require_model()
    _require_camera_permission()
    profile = None
    if cfgmod.profile_path().exists():
        profile = Profile.from_json(cfgmod.profile_path().read_text())
    view = args.view or (profile.view if profile else cfg.view)
    feature_set = get_feature_set(view)
    scorer = Scorer(profile, tau=cfg.ema_tau) if profile else None

    print("press q or escape to close")
    source = PoseSource(
        model,
        camera_index=args.camera if args.camera is not None else cfg.camera_index,
        width=cfg.camera_width,
        height=cfg.camera_height,
        fps=15.0,
        keep_frame=True,
    )
    with source:
        for frame in source.frames():
            image = source.last_frame
            if image is None:
                continue
            height, width = image.shape[:2]
            lines = [f"view: {view}"]

            if frame is None:
                lines.append("no pose detected")
            else:
                sample = feature_set.extract(frame, cfg.quality_limits())
                for i in range(frame.xy.shape[0]):
                    if frame.visibility[i] < 0.5:
                        continue
                    x = int(frame.xy[i, 0] / (width / height) * width)
                    y = int(frame.xy[i, 1] * height)
                    cv2.circle(image, (x, y), 3, (80, 220, 160), -1)
                # Shown whether the frame passed or not: when nothing passes,
                # these numbers are the only way to tell a badly placed camera
                # from a threshold that wants nudging.
                for name, value in sorted(sample.metrics.items()):
                    lines.append(f"{name}: {value:.2f}")
                if not sample.ok:
                    lines.append(f"REJECTED: {sample.reason}")
                else:
                    if profile is not None:
                        parts = score_parts(profile, sample.values)
                        value = scorer.update(frame.ts, sample.values)
                        if parts is None or value is None:
                            lines.append("score: unavailable")
                        else:
                            lines.append(f"score: {value:.2f}  (axis {parts.axis:.2f})")
                            # High disagreement means the features contradict each
                            # other, i.e. you are in a posture neither calibration
                            # pose covers. Worth seeing while aiming the camera.
                            lines.append(
                                f"disagreement: {parts.disagreement:.2f}"
                                f"  (+{parts.penalty:.2f})"
                            )
                    hips = frame.vis(LM.LEFT_HIP, LM.RIGHT_HIP)
                    if view == "side" and hips < 0.6:
                        lines.append("hips not in frame - move the camera back")
                    for name, raw in zip(feature_set.names, sample.values):
                        if np.isfinite(raw):
                            lines.append(f"{name}: {raw:.3f}")

            for i, text in enumerate(lines):
                cv2.putText(
                    image, text, (12, 28 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 3, cv2.LINE_AA,
                )
                cv2.putText(
                    image, text, (12, 28 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA,
                )
            cv2.imshow("posture-guard preview", image)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    cv2.destroyAllWindows()
    return 0


def cmd_install_app(args) -> int:
    """Wrap the installed package in a real .app bundle."""
    from .macapp import BundleError, build_app, default_destination, register, set_login_item

    if sys.platform != "darwin":
        raise SystemExit("app bundles are a macOS thing")

    destination = Path(args.to).expanduser() if args.to else default_destination()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        bundle = build_app(destination)
    except BundleError as exc:
        raise SystemExit(str(exc)) from None
    register(bundle)

    from .arch import running_arch, under_rosetta

    print(f"built {bundle}")
    print(f"  runs {sys.executable} -m posture_guard run")
    print(f"  pinned to {running_arch()}")
    if under_rosetta():
        print(
            "  WARNING: this shell is translated by Rosetta, so the app is pinned to\n"
            "  x86_64. If the packages are arm64 it will not start. Recreate the\n"
            "  virtualenv natively and run install-app again."
        )
    print("  menu bar only, no Dock icon")
    print(f"  logs to {Path.home() / 'Library' / 'Logs' / 'posture-guard.log'}")

    if args.login_item:
        if set_login_item(bundle, add=True):
            print("  added to your login items")
        else:
            print("  could not add it to login items; do it in System Settings > General > Login Items")

    print()
    print("Open it once from Finder. macOS will ask for the camera in the app's own")
    print("name this time, rather than your terminal's.")
    return 0


def cmd_report(args) -> int:
    from .report import render_text, write_html

    cfg = _load_config()
    with Store(cfgmod.database_path()) as store:
        if args.html is not None:
            path = write_html(store, cfg, Path(args.html), days=args.days)
            print(f"wrote {path}")
            if args.open:
                webbrowser.open(path.as_uri())
        else:
            print(render_text(store, cfg, days=args.days))
    return 0


def cmd_app_log(args) -> int:
    """Show what the bundled app wrote, since it has no terminal to write to."""
    from .macapp import log_path

    path = log_path()
    if not path.exists():
        print(f"no log at {path}, so the app has never started")
        return 1
    if not args.follow:
        text = path.read_text(errors="replace").splitlines()
        for line in text[-args.lines :]:
            print(line)
        return 0

    print(f"following {path}, ctrl-c to stop")
    with path.open(errors="replace") as handle:
        handle.seek(0, 2)
        try:
            while True:
                line = handle.readline()
                if line:
                    print(line.rstrip())
                else:
                    time.sleep(0.3)
        except KeyboardInterrupt:
            return 0


def cmd_pause(args) -> int:
    from .app import _clear_pause_flag, _write_pause_flag

    if args.minutes <= 0:
        _clear_pause_flag()
        print("resumed")
    else:
        _write_pause_flag(time.time() + args.minutes * 60)
        print(f"paused for {args.minutes} minutes")
    return 0


def cmd_resume(args) -> int:
    from .app import _clear_pause_flag

    _clear_pause_flag()
    print("resumed")
    return 0


def cmd_selftest(args) -> int:
    from .selftest import run_selftest, selftest_passed

    cfg = _load_config()
    view = args.view or cfg.view
    print(f"running the pipeline on synthetic data ({view} view)\n")
    checks = run_selftest(view)
    for check in checks:
        print(check)
    passed = selftest_passed(checks)
    print("\n" + ("all checks passed" if passed else "some checks failed"))
    return 0 if passed else 1


def cmd_config(args) -> int:
    cfg = _load_config()
    if args.path:
        print(cfgmod.config_path())
        return 0
    if args.set:
        for pair in args.set:
            if "=" not in pair:
                raise SystemExit(f"expected key=value, got {pair!r}")
            key, value = pair.split("=", 1)
            try:
                cfg.set_from_string(key.strip(), value.strip())
            except ValueError as exc:
                raise SystemExit(str(exc)) from None
        cfg.save()
        print(f"saved {cfgmod.config_path()}")
    for key, value in asdict(cfg).items():
        print(f"{key:<24}{value}")
    return 0


# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="posture-guard",
        description="Local webcam posture monitor aimed at shoulder protraction.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("setup", help="create the data directory and download the pose model")
    p.add_argument("--view", choices=VIEWS, help="which camera placement you intend to use")
    p.add_argument("--force", action="store_true", help="re-download the model")
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("doctor", help="check dependencies, model, camera and calibration")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("calibrate", help="teach it your good posture and your slouch")
    p.add_argument("--view", choices=VIEWS)
    p.add_argument("--camera", type=int, help="camera index to use")
    p.add_argument("--seconds", type=float, default=12.0, help="how long to hold each pose")
    p.add_argument(
        "--min-separation",
        type=float,
        default=1.0,
        dest="min_separation",
        help="how many noise widths a feature must span to be used",
    )
    p.add_argument("--force", action="store_true", help="save even a weak profile")
    p.add_argument(
        "--terminal",
        action="store_true",
        help="use the countdown-in-the-terminal flow instead of the window",
    )
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("run", help="start monitoring")
    p.add_argument("--headless", action="store_true", help="no menu bar, console only")
    p.add_argument(
        "--verbose",
        action="store_true",
        help="print a live score readout, to see it reacting to you",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="log every frame's score or rejection reason, not just the startup steps",
    )
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("preview", help="live landmarks and features, for aiming the camera")
    p.add_argument("--view", choices=VIEWS)
    p.add_argument("--camera", type=int)
    p.set_defaults(func=cmd_preview)

    p = sub.add_parser("install-app", help="build a real macOS app bundle")
    p.add_argument("--to", help="where to put it (default: ~/Applications)")
    p.add_argument(
        "--login-item", action="store_true", help="also start it when you log in"
    )
    p.set_defaults(func=cmd_install_app)

    p = sub.add_parser("report", help="show your history")
    p.add_argument("--days", type=int, default=14)
    p.add_argument("--html", nargs="?", const="posture-report.html", help="write an HTML report")
    p.add_argument("--open", action="store_true", help="open the HTML report in a browser")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("app-log", help="show what the bundled app logged")
    p.add_argument("-f", "--follow", action="store_true", help="keep watching for new lines")
    p.add_argument("-n", "--lines", type=int, default=40, help="how many lines to show")
    p.set_defaults(func=cmd_app_log)

    p = sub.add_parser("pause", help="stop alerting for a while")
    p.add_argument("minutes", type=float, nargs="?", default=30.0)
    p.set_defaults(func=cmd_pause)

    p = sub.add_parser("resume", help="undo a pause")
    p.set_defaults(func=cmd_resume)

    p = sub.add_parser("selftest", help="verify the pipeline on synthetic data, no camera needed")
    p.add_argument("--view", choices=VIEWS)
    p.set_defaults(func=cmd_selftest)

    p = sub.add_parser("config", help="show or change settings")
    p.add_argument("--set", action="append", metavar="KEY=VALUE")
    p.add_argument("--path", action="store_true", help="print the config file path")
    p.set_defaults(func=cmd_config)

    return parser


def _surface(message: str) -> None:
    """Put a failure somewhere the user will actually see it.

    Launched from Finder there is no terminal, so a message on stderr goes into
    a log file nobody thinks to open and the app simply appears not to start.
    """
    if sys.stderr.isatty() or sys.platform != "darwin":
        return
    import shutil
    import subprocess

    osascript = shutil.which("osascript")
    if not osascript:
        return
    text = message.replace("\\", "\\\\").replace('"', '\\"')
    subprocess.run(  # noqa: S603 - fixed binary, escaped literal
        [osascript, "-e", f'display alert "posture-guard could not start" message "{text}"'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except SystemExit as exc:
        if isinstance(exc.code, str):
            _surface(exc.code)
        raise
    except ImportError as exc:
        from .arch import explain_import_error

        explanation = explain_import_error(exc)
        if explanation is None:
            _surface(f"{type(exc).__name__}: {exc}")
            raise
        # numpy's own message is a page long and never mentions Rosetta.
        print(f"\n{explanation}\n", file=sys.stderr)
        _surface(explanation)
        return 1
    except Exception as exc:  # noqa: BLE001 - re-raised after being surfaced
        _surface(f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
