"""Configuration, and where everything lives on disk.

One directory holds the lot: settings, calibration profile, the pose model and
the history database. Nothing is written anywhere else, so uninstalling is
`rm -rf` on a single path.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

APP_NAME = "posture-guard"
MODEL_FILENAME = "pose_landmarker_lite.task"


def app_dir() -> Path:
    """Per-user data directory, overridable with POSTURE_GUARD_HOME (used by tests)."""
    override = os.environ.get("POSTURE_GUARD_HOME")
    if override:
        return Path(override).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    base = os.environ.get("XDG_DATA_HOME")
    return (Path(base) if base else Path.home() / ".local" / "share") / APP_NAME


def config_path() -> Path:
    return app_dir() / "config.json"


def profile_path() -> Path:
    return app_dir() / "profile.json"


def database_path() -> Path:
    return app_dir() / "posture.db"


def model_path() -> Path:
    return app_dir() / MODEL_FILENAME


def pause_flag_path() -> Path:
    """Escape hatch: a running app polls this, so a terminal can always call it off."""
    return app_dir() / "paused-until"


@dataclass
class Config:
    # -- capture ------------------------------------------------------------
    view: str = "side"  # "side" measures protraction; "frontal" approximates it
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    fps: float = 6.0  # posture changes slowly; this keeps a core mostly idle

    # -- scoring ------------------------------------------------------------
    ema_tau: float = 1.5

    # -- alerting -----------------------------------------------------------
    enter: float = 0.55
    exit: float = 0.35
    dwell_s: float = 8.0
    ramp_s: float = 25.0
    release_s: float = 0.5
    absent_after_s: float = 5.0
    min_intensity: float = 0.15
    alerters: list[str] = field(default_factory=lambda: ["dim"])
    max_dim: float = 0.55  # hard ceiling on overlay opacity; never fully blind
    notify_cooldown_s: float = 300.0

    # -- schedule -----------------------------------------------------------
    quiet_start: str | None = None  # "HH:MM"; alerting off between these
    quiet_end: str | None = None

    # -- history and progression -------------------------------------------
    bucket_seconds: int = 30
    ratchet_enabled: bool = True
    ratchet_promote_at: float = 0.85
    ratchet_demote_at: float = 0.55
    ratchet_step: float = 0.92
    ratchet_min_enter: float = 0.25
    ratchet_max_enter: float = 0.75
    ratchet_min_days: int = 7
    ratchet_min_hours: float = 10.0
    recalibrate_after_days: int = 30

    def policy_kwargs(self) -> dict:
        return {
            "enter": self.enter,
            "exit": self.exit,
            "dwell_s": self.dwell_s,
            "ramp_s": self.ramp_s,
            "release_s": self.release_s,
            "absent_after_s": self.absent_after_s,
            "min_intensity": self.min_intensity,
        }

    def validate(self) -> None:
        if self.view not in ("side", "frontal"):
            raise ValueError(f"view must be 'side' or 'frontal', not {self.view!r}")
        if not 0.0 < self.exit < self.enter:
            raise ValueError("need 0 < exit < enter")
        if not 0.0 < self.max_dim <= 0.85:
            # Above ~0.85 the screen stops being usable, which turns a nudge
            # into something you have to fight. Refuse to go there.
            raise ValueError("max_dim must be in (0, 0.85]")
        if self.fps <= 0 or self.fps > 30:
            raise ValueError("fps must be in (0, 30]")
        for name in ("quiet_start", "quiet_end"):
            value = getattr(self, name)
            if value is not None:
                _parse_hhmm(value, name)

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        path = path or config_path()
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text())
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown settings in {path}: {', '.join(sorted(unknown))}")
        cfg = cls(**raw)
        cfg.validate()
        return cfg

    def save(self, path: Path | None = None) -> Path:
        path = path or config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")
        return path

    def set_from_string(self, key: str, value: str) -> None:
        """Apply one ``key=value`` pair coming from the command line."""
        known = {f.name: f for f in fields(self)}
        if key not in known:
            raise ValueError(f"unknown setting {key!r}")
        current = getattr(self, key)
        nulled = value.strip().lower() in ("", "none", "null")
        if key == "alerters":
            parsed: object = [v.strip() for v in value.split(",") if v.strip()]
        elif key.startswith("quiet_"):
            parsed = None if nulled else value
        elif isinstance(current, bool):
            parsed = value.lower() in ("1", "true", "yes", "on")
        elif isinstance(current, int):
            parsed = int(value)
        elif isinstance(current, float):
            parsed = float(value)
        else:
            parsed = value
        setattr(self, key, parsed)
        self.validate()


def _parse_hhmm(value: str, name: str) -> tuple[int, int]:
    try:
        hh, mm = value.split(":")
        h, m = int(hh), int(mm)
        if not (0 <= h < 24 and 0 <= m < 60):
            raise ValueError
        return h, m
    except (ValueError, AttributeError):
        raise ValueError(f"{name} must look like 'HH:MM', got {value!r}") from None


def in_quiet_hours(cfg: Config, local_hour: int, local_minute: int) -> bool:
    """True when alerting should stay silent. Handles windows crossing midnight."""
    if not cfg.quiet_start or not cfg.quiet_end:
        return False
    start = _parse_hhmm(cfg.quiet_start, "quiet_start")
    end = _parse_hhmm(cfg.quiet_end, "quiet_end")
    now = (local_hour, local_minute)
    if start <= end:
        return start <= now < end
    return now >= start or now < end
