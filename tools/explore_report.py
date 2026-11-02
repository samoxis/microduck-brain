"""What exploration did in a recorded run: every bend and why it ended, every refusal and why, and
whether avoidances cluster after bends.

The recording is replayed through the current brain with the given config (the brain is a pure
function of its inputs). For a run recorded with this code and config the replay is exact and so is
the report; for an older run it shows what this code *would* have decided on the same inputs.

    uv run python tools/explore_report.py runs/<run>/run.jsonl.gz [-c brain.toml] [--window 6]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microduck_brain import config as config_mod  # noqa: E402
from microduck_brain.brain import Brain  # noqa: E402
from microduck_brain.record import read  # noqa: E402


def _reason_kind(reason: str) -> str:
    """Group reasons without their numbers: 'left half of the view clear only 0.52 m' -> '... clear only'."""
    return re.sub(r"\s*-?\d+(\.\d+)?\s*m?$", "", reason) or "-"


def report(path: str, cfg, window: float) -> dict:
    brain = Brain(cfg)
    events: list[tuple[float, str, float, str]] = []
    avoid_starts: list[float] = []
    differing = 0
    t0 = t_end = None
    for tick in read(path):
        before = brain.stats.avoids
        cmd = brain.tick(tick.inputs)
        if brain.stats.avoids > before:
            avoid_starts.append(tick.inputs.now)
        events.extend(brain.explore_log)
        brain.explore_log.clear()
        if cmd.kind != tick.cmd.kind or abs(cmd.vx - tick.cmd.vx) > 1e-4 or abs(cmd.vyaw - tick.cmd.vyaw) > 1e-4:
            differing += 1
        t0 = tick.inputs.now if t0 is None else t0
        t_end = tick.inputs.now
    duration = (t_end - t0) if t0 is not None else 0.0

    bends = [e for e in events if e[1] == "bend"]
    ends = [e for e in events if e[1] in ("completed", "interrupted", "aborted")]
    refused = Counter(_reason_kind(e[3]) for e in events if e[1] == "refused")
    ended = Counter(f"{e[1]}: {_reason_kind(e[3])}" if e[3] else e[1] for e in ends)

    # Avoidances starting within `window` s after a bend started, vs. the rate elsewhere.
    in_window = [a for a in avoid_starts if any(0 <= a - b[0] <= window for b in bends)]
    win_time = 0.0
    last = -math.inf
    for b in sorted(e[0] for e in bends):           # union of [b, b + window]
        win_time += min(window, b + window - max(b, last))
        last = max(last, b + window)
    out_time = max(1e-9, duration - win_time)
    rows = []
    for b in bends:
        end = next((e for e in ends if e[0] >= b[0]), None)
        rows.append({
            "t": round(b[0] - (t0 or 0.0), 1),
            "angle_deg": round(math.degrees(b[2])),
            "ended": end[1] if end else "still running",
            "after_s": round(end[0] - b[0], 1) if end else None,
            "why": end[3] if end else "",
            "avoids_within_window": sum(1 for a in avoid_starts if 0 <= a - b[0] <= window),
        })
    return {
        "replay_differing_commands": differing,
        "duration_s": round(duration, 1),
        "bends_started": len(bends),
        "bends_ended": dict(ended),
        "refusals": dict(refused),
        "avoidances": len(avoid_starts),
        "avoidances_per_min_after_bends": round(60 * len(in_window) / win_time, 2) if win_time else None,
        "avoidances_per_min_elsewhere": round(60 * (len(avoid_starts) - len(in_window)) / out_time, 2),
        "window_s": window,
        "bends": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    ap.add_argument("--config", "-c", help="TOML file (default: ./brain.toml if present, else defaults)")
    ap.add_argument("--window", type=float, default=6.0, help="seconds after a bend counted as 'after the bend'")
    args = ap.parse_args()
    path = args.config or ("brain.toml" if Path("brain.toml").exists() else None)
    cfg = config_mod.load(path)
    if not cfg.explore.enabled:
        print("note: exploration is disabled in this config; enable it to see bends", file=sys.stderr)
    print(json.dumps(report(args.run, cfg, args.window), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
