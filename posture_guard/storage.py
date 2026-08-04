"""Local history in SQLite.

Only derived numbers are stored: a posture score, seconds spent above or below
threshold, and event markers. No images, no landmarks, nothing that could
reconstruct what the camera saw.

Samples are folded into fixed-length buckets (30 s by default) as they arrive, so
a year of continuous use is roughly a million rows and a few tens of megabytes.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS buckets (
    bucket_ts   INTEGER PRIMARY KEY,
    n_frames    INTEGER NOT NULL DEFAULT 0,
    n_valid     INTEGER NOT NULL DEFAULT 0,
    score_sum   REAL    NOT NULL DEFAULT 0,
    score_max   REAL    NOT NULL DEFAULT 0,
    secs_good   REAL    NOT NULL DEFAULT 0,
    secs_bad    REAL    NOT NULL DEFAULT 0,
    secs_absent REAL    NOT NULL DEFAULT 0,
    secs_alert  REAL    NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS events (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    ts     REAL NOT NULL,
    kind   TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);

CREATE TABLE IF NOT EXISTS profiles (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    created_ts REAL NOT NULL,
    view       TEXT NOT NULL,
    payload    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS app_state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class Bucket:
    bucket_ts: int
    n_frames: int
    n_valid: int
    score_sum: float
    score_max: float
    secs_good: float
    secs_bad: float
    secs_absent: float
    secs_alert: float

    @property
    def score_mean(self) -> float | None:
        return self.score_sum / self.n_valid if self.n_valid else None

    @property
    def secs_measured(self) -> float:
        return self.secs_good + self.secs_bad


class Store:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- writing -------------------------------------------------------------

    def add_bucket(
        self,
        bucket_ts: int,
        *,
        n_frames: int,
        n_valid: int,
        score_sum: float,
        score_max: float,
        secs_good: float,
        secs_bad: float,
        secs_absent: float,
        secs_alert: float,
    ) -> None:
        """Merge a bucket, adding to whatever is already there for that slot."""
        self.conn.execute(
            """
            INSERT INTO buckets (bucket_ts, n_frames, n_valid, score_sum, score_max,
                                 secs_good, secs_bad, secs_absent, secs_alert)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(bucket_ts) DO UPDATE SET
                n_frames    = n_frames    + excluded.n_frames,
                n_valid     = n_valid     + excluded.n_valid,
                score_sum   = score_sum   + excluded.score_sum,
                score_max   = MAX(score_max, excluded.score_max),
                secs_good   = secs_good   + excluded.secs_good,
                secs_bad    = secs_bad    + excluded.secs_bad,
                secs_absent = secs_absent + excluded.secs_absent,
                secs_alert  = secs_alert  + excluded.secs_alert
            """,
            (
                int(bucket_ts),
                n_frames,
                n_valid,
                score_sum,
                score_max,
                secs_good,
                secs_bad,
                secs_absent,
                secs_alert,
            ),
        )
        self.conn.commit()

    def log_event(self, ts: float, kind: str, detail: dict | None = None) -> None:
        self.conn.execute(
            "INSERT INTO events (ts, kind, detail) VALUES (?, ?, ?)",
            (float(ts), str(kind), json.dumps(detail or {})),
        )
        self.conn.commit()

    def save_profile(self, view: str, payload: str, created_ts: float | None = None) -> None:
        self.conn.execute(
            "INSERT INTO profiles (created_ts, view, payload) VALUES (?, ?, ?)",
            (float(created_ts if created_ts is not None else time.time()), view, payload),
        )
        self.conn.commit()

    def set_state(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO app_state (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, str(value)),
        )
        self.conn.commit()

    # -- reading -------------------------------------------------------------

    def get_state(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute("SELECT value FROM app_state WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def buckets_between(self, start_ts: float, end_ts: float) -> list[Bucket]:
        rows = self.conn.execute(
            "SELECT * FROM buckets WHERE bucket_ts >= ? AND bucket_ts < ? ORDER BY bucket_ts",
            (int(start_ts), int(end_ts)),
        ).fetchall()
        return [Bucket(**dict(r)) for r in rows]

    def events_between(self, start_ts: float, end_ts: float) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM events WHERE ts >= ? AND ts < ? ORDER BY ts",
            (float(start_ts), float(end_ts)),
        ).fetchall()

    def count_events(self, start_ts: float, end_ts: float, kind: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM events WHERE ts >= ? AND ts < ? AND kind = ?",
            (float(start_ts), float(end_ts), kind),
        ).fetchone()
        return int(row["n"])

    def first_bucket_ts(self) -> int | None:
        row = self.conn.execute("SELECT MIN(bucket_ts) AS t FROM buckets").fetchone()
        return int(row["t"]) if row and row["t"] is not None else None

    def profile_history(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT id, created_ts, view FROM profiles ORDER BY created_ts"
        ).fetchall()


class BucketAggregator:
    """Folds per-tick observations into buckets and writes them out as they close."""

    def __init__(self, store: Store, bucket_seconds: int = 30, good_below: float = 0.55):
        self.store = store
        self.bucket_seconds = max(1, int(bucket_seconds))
        self.good_below = good_below
        self._slot: int | None = None
        self._reset()

    def _reset(self) -> None:
        self.n_frames = 0
        self.n_valid = 0
        self.score_sum = 0.0
        self.score_max = 0.0
        self.secs_good = 0.0
        self.secs_bad = 0.0
        self.secs_absent = 0.0
        self.secs_alert = 0.0

    def _slot_for(self, ts: float) -> int:
        return int(ts // self.bucket_seconds) * self.bucket_seconds

    def add(self, ts: float, score: float | None, dt: float, alerting: bool = False) -> None:
        slot = self._slot_for(ts)
        if self._slot is None:
            self._slot = slot
        elif slot != self._slot:
            self.flush()
            self._slot = slot

        # A long gap means the machine slept or the app was stopped; that is not
        # time spent at the desk and must not dilute the day's statistics.
        dt = max(0.0, min(dt, self.bucket_seconds * 2.0))

        self.n_frames += 1
        if score is None:
            self.secs_absent += dt
        else:
            self.n_valid += 1
            self.score_sum += score
            self.score_max = max(self.score_max, score)
            if score < self.good_below:
                self.secs_good += dt
            else:
                self.secs_bad += dt
        if alerting:
            self.secs_alert += dt

    def flush(self) -> None:
        if self._slot is None or self.n_frames == 0:
            self._reset()
            return
        self.store.add_bucket(
            self._slot,
            n_frames=self.n_frames,
            n_valid=self.n_valid,
            score_sum=self.score_sum,
            score_max=self.score_max,
            secs_good=self.secs_good,
            secs_bad=self.secs_bad,
            secs_absent=self.secs_absent,
            secs_alert=self.secs_alert,
        )
        self._reset()
