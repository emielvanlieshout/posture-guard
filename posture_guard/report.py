"""Terminal and HTML views of the history.

The point of the report is not the daily number, which bounces around with how
long you sat down and what you were doing. It is the slope across weeks, read
next to the staircase of threshold changes: good posture holding steady while
the threshold keeps tightening is progress, even when the percentage looks flat.
"""

from __future__ import annotations

import html
import json
import time
from datetime import datetime
from pathlib import Path

from .config import Config
from .ratchet import EVENT_KIND, current_enter
from .stats import DayStats, daily_stats, trend, window_stats
from .storage import Store

BAR_CHARS = "░▏▎▍▌▋▊▉█"


def _bar(fraction: float | None, width: int = 20) -> str:
    if fraction is None:
        return "·" * width
    filled = max(0.0, min(1.0, fraction)) * width
    full = int(filled)
    out = "█" * full
    if full < width:
        remainder = filled - full
        out += BAR_CHARS[int(remainder * (len(BAR_CHARS) - 1))]
    return out.ljust(width, "░")[:width]


def _pct(fraction: float | None) -> str:
    return "  n/a" if fraction is None else f"{fraction:>4.0%}"


def _score(value: float | None) -> str:
    return " n/a" if value is None else f"{value:.2f}"


def _threshold_changes(store: Store, days: int) -> list[tuple[float, dict]]:
    now = time.time()
    rows = store.events_between(now - days * 86400, now + 1)
    return [(r["ts"], json.loads(r["detail"])) for r in rows if r["kind"] == EVENT_KIND]


def render_text(store: Store, cfg: Config, days: int = 14) -> str:
    now = time.time()
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()

    windows = {
        "today": window_stats(store, today_start, now + 1, cfg.bucket_seconds),
        "last 7 days": window_stats(store, now - 7 * 86400, now + 1, cfg.bucket_seconds),
        "last 30 days": window_stats(store, now - 30 * 86400, now + 1, cfg.bucket_seconds),
    }
    enter = current_enter(store, cfg)

    lines = [
        "posture-guard history",
        f"current threshold: {enter:.3f}   (started at {cfg.enter:.3f}; lower is stricter)",
        "",
        f"{'window':<14}{'good':>6}{'measured':>10}{'mean':>7}{'alerts':>8}{'best run':>10}",
    ]
    for label, w in windows.items():
        lines.append(
            f"{label:<14}{_pct(w.good_fraction):>6}{w.measured_hours:>9.1f}h"
            f"{_score(w.mean_score):>7}{w.alerts:>8}{w.longest_good_minutes:>8.0f}m"
        )

    history: list[DayStats] = daily_stats(store, days, cfg.bucket_seconds)
    lines += ["", f"day by day (last {days} days)"]
    for entry in history:
        s = entry.stats
        lines.append(
            f"{entry.day.isoformat()}  {_bar(s.good_fraction)}  {_pct(s.good_fraction)}"
            f"  {s.measured_hours:>4.1f}h  mean {_score(s.mean_score)}"
        )

    slope = trend([e.stats.mean_score for e in history])
    if slope is None:
        lines += ["", "trend: not enough days measured yet"]
    else:
        per_week = slope * 7
        if per_week < -0.005:
            verdict = "improving"
        elif per_week > 0.005:
            verdict = "sliding back"
        else:
            verdict = "flat"
        lines += ["", f"trend: {verdict} ({per_week:+.3f} mean score per week; lower is better)"]

    changes = _threshold_changes(store, max(days, 120))
    if changes:
        lines += ["", "threshold changes"]
        for ts, detail in changes:
            when = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
            lines.append(
                f"{when}  {detail.get('from', 0):.3f} -> {detail.get('to', 0):.3f}"
                f"   {detail.get('reason', '')}"
            )

    return "\n".join(lines)


def _svg_bars(history: list[DayStats]) -> str:
    width, height, gap = 760, 190, 4
    n = max(len(history), 1)
    bar_w = (width - gap * (n - 1)) / n
    parts = [
        f'<svg viewBox="0 0 {width} {height + 26}" role="img" '
        f'aria-label="Share of measured time in good posture, per day">'
    ]
    for i, entry in enumerate(history):
        frac = entry.stats.good_fraction
        x = i * (bar_w + gap)
        parts.append(
            f'<rect x="{x:.1f}" y="0" width="{bar_w:.1f}" height="{height}" '
            f'class="track" rx="3"/>'
        )
        if frac is not None:
            h = max(2.0, frac * height)
            parts.append(
                f'<rect x="{x:.1f}" y="{height - h:.1f}" width="{bar_w:.1f}" '
                f'height="{h:.1f}" class="bar" rx="3">'
                f"<title>{entry.day.isoformat()}: {frac:.0%} good, "
                f"{entry.stats.measured_hours:.1f}h measured</title></rect>"
            )
        if n <= 20 or i % 2 == 0:
            parts.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{height + 18}" class="tick" '
                f'text-anchor="middle">{entry.day.strftime("%d/%m")}</text>'
            )
    parts.append("</svg>")
    return "".join(parts)


def _svg_line(history: list[DayStats]) -> str:
    width, height = 760, 150
    values = [e.stats.mean_score for e in history]
    points = [(i, v) for i, v in enumerate(values) if v is not None]
    if len(points) < 2:
        return '<p class="empty">Not enough days measured to plot a trend yet.</p>'
    n = max(len(values) - 1, 1)
    hi = max(1.0, max(v for _, v in points))
    coords = " ".join(
        f"{(i / n) * width:.1f},{height - (v / hi) * height:.1f}" for i, v in points
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Mean posture score per day, lower is better">'
        f'<polyline points="{coords}" class="line"/></svg>'
    )


def render_html(store: Store, cfg: Config, days: int = 30) -> str:
    now = time.time()
    history = daily_stats(store, days, cfg.bucket_seconds)
    week = window_stats(store, now - 7 * 86400, now + 1, cfg.bucket_seconds)
    month = window_stats(store, now - 30 * 86400, now + 1, cfg.bucket_seconds)
    enter = current_enter(store, cfg)
    slope = trend([e.stats.mean_score for e in history])

    def card(label: str, value: str, sub: str) -> str:
        return (
            f'<div class="card"><div class="label">{html.escape(label)}</div>'
            f'<div class="value">{html.escape(value)}</div>'
            f'<div class="sub">{html.escape(sub)}</div></div>'
        )

    cards = "".join(
        [
            card(
                "Good posture, 7 days",
                _pct(week.good_fraction).strip(),
                f"{week.measured_hours:.1f} h measured",
            ),
            card(
                "Mean score, 7 days",
                _score(week.mean_score),
                "0 = your calibrated good, 1 = your slouch",
            ),
            card("Threshold", f"{enter:.3f}", f"started at {cfg.enter:.3f}, lower is stricter"),
            card(
                "Weekly trend",
                "n/a" if slope is None else f"{slope * 7:+.3f}",
                "mean score per week, lower is better",
            ),
            card("Alerts, 30 days", str(month.alerts), f"{month.alert_hours:.1f} h dimmed"),
            card(
                "Best unbroken stretch",
                f"{week.longest_good_minutes:.0f} min",
                "last 7 days",
            ),
        ]
    )

    changes = _threshold_changes(store, 365)
    if changes:
        rows = "".join(
            f"<tr><td>{datetime.fromtimestamp(ts).strftime('%Y-%m-%d')}</td>"
            f"<td>{d.get('from', 0):.3f} &rarr; {d.get('to', 0):.3f}</td>"
            f"<td>{html.escape(str(d.get('reason', '')))}</td></tr>"
            for ts, d in changes
        )
        table = f"<h2>Threshold changes</h2><table><tbody>{rows}</tbody></table>"
    else:
        table = (
            "<h2>Threshold changes</h2>"
            '<p class="empty">No changes yet. The first review happens after a full week '
            f"with at least {cfg.ratchet_min_hours:.0f} hours measured.</p>"
        )

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>posture-guard report</title>
<style>
  :root {{ color-scheme: light dark;
    --bg:#fbfbfa; --fg:#1a1a19; --muted:#6b6b68; --line:#dedddb; --accent:#2f7d63; --track:#ececeb; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#16161a; --fg:#ececef; --muted:#9a9aa2; --line:#2c2c33; --accent:#5fb99a; --track:#232329; }}
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; padding:2.5rem 1.25rem 4rem; background:var(--bg); color:var(--fg);
    font:16px/1.55 ui-sans-serif,-apple-system,Segoe UI,Roboto,sans-serif; }}
  main {{ max-width: 820px; margin: 0 auto; }}
  h1 {{ font-size:1.5rem; margin:0 0 .25rem; }}
  h2 {{ font-size:1.05rem; margin:2.5rem 0 .75rem; font-weight:600; }}
  .meta {{ color:var(--muted); font-size:.875rem; margin:0 0 2rem; }}
  .grid {{ display:grid; gap:.75rem; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); }}
  .card {{ border:1px solid var(--line); border-radius:10px; padding:.9rem 1rem; }}
  .label {{ font-size:.75rem; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); }}
  .value {{ font-size:1.75rem; font-weight:600; margin:.2rem 0; font-variant-numeric:tabular-nums; }}
  .sub {{ font-size:.8rem; color:var(--muted); }}
  .chart {{ overflow-x:auto; }}
  svg {{ width:100%; height:auto; display:block; }}
  .track {{ fill:var(--track); }}
  .bar {{ fill:var(--accent); }}
  .line {{ fill:none; stroke:var(--accent); stroke-width:2.5; stroke-linejoin:round; }}
  .tick {{ fill:var(--muted); font-size:11px; }}
  table {{ border-collapse:collapse; width:100%; font-size:.9rem; }}
  td {{ border-top:1px solid var(--line); padding:.5rem .35rem; }}
  td:first-child {{ color:var(--muted); white-space:nowrap; }}
  .empty {{ color:var(--muted); font-size:.9rem; }}
  footer {{ margin-top:3rem; color:var(--muted); font-size:.8rem; border-top:1px solid var(--line);
    padding-top:1rem; }}
</style></head>
<body><main>
<h1>posture-guard</h1>
<p class="meta">Generated {generated} &middot; last {days} days &middot; all data local to this machine</p>
<div class="grid">{cards}</div>
<h2>Share of measured time in good posture</h2>
<div class="chart">{_svg_bars(history)}</div>
<h2>Mean posture score (lower is better)</h2>
<div class="chart">{_svg_line(history)}</div>
{table}
<footer>Scores are relative to your own calibration, so they are comparable only for as long as
that profile stands. Recalibrating resets the baseline.</footer>
</main></body></html>
"""


def write_html(store: Store, cfg: Config, path: Path, days: int = 30) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(store, cfg, days))
    return path
