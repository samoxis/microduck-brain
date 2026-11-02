"""What exploration did in a recorded run, and where the avoidances happen.

* every bend and why it ended; every refused bend, grouped by reason;
* every side choice made by exploration at a turn in place (tie broken, agreed with the freer side,
  or overrode it), with the novelty and free space it saw;
* avoidance rates: within ``--window`` s after a bend, after a side choice, and after *every* turn in
  place. The last one exists in runs with exploration off too, so OFF and ON can be compared.

The recording is replayed through the current brain with the given config (the brain is a pure
function of its inputs). For a run recorded with this code and config the replay is exact
(``replay_differing_commands`` = 0) and so is the report.

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
    """Group reasons without their numbers: text before ':' if any, else without a trailing number."""
    if ":" in reason:
        return reason.split(":", 1)[0]
    return re.sub(r"\s*-?\d+(\.\d+)?\s*m?$", "", reason) or "-"


def _union_length(starts: list[float], window: float) -> float:
    total, last = 0.0, -math.inf
    for b in sorted(starts):
        total += max(0.0, b + window - max(b, last))
        last = max(last, b + window)
    return total


def _rate(avoids: list[float], starts: list[float], window: float, duration: float) -> dict:
    inside = [a for a in avoids if any(0 <= a - b <= window for b in starts)]
    t_in = _union_length(starts, window)
    t_out = max(1e-9, duration - t_in)
    return {
        "events": len(starts),
        "avoids_per_min_after": round(60 * len(inside) / t_in, 2) if t_in else None,
        "avoids_per_min_elsewhere": round(60 * (len(avoids) - len(inside)) / t_out, 2),
    }


def report(path: str, cfg, window: float) -> dict:
    brain = Brain(cfg)
    events: list[tuple[float, str, float, str]] = []
    avoid_starts: list[float] = []
    turn_ends: list[float] = []          # turn / bigturn -> forward
    differing = 0
    t0 = t_end = None
    for tick in read(path):
        before, state_before = brain.stats.avoids, brain.state_name
        cmd = brain.tick(tick.inputs)
        if brain.stats.avoids > before:
            avoid_starts.append(tick.inputs.now)
        if state_before in ("turn", "bigturn") and brain.state_name == "forward":
            turn_ends.append(tick.inputs.now)
        events.extend(brain.explore_log)
        brain.explore_log.clear()
        if cmd.kind != tick.cmd.kind or abs(cmd.vx - tick.cmd.vx) > 1e-4 or abs(cmd.vyaw - tick.cmd.vyaw) > 1e-4:
            differing += 1
        t0 = tick.inputs.now if t0 is None else t0
        t_end = tick.inputs.now
    duration = (t_end - t0) if t0 is not None else 0.0
    t0 = t0 or 0.0

    def within(t: float) -> int:
        return sum(1 for a in avoid_starts if 0 <= a - t <= window)

    bends = [e for e in events if e[1] == "bend"]
    ends = [e for e in events if e[1] in ("completed", "interrupted", "aborted")]
    sides = [e for e in events if e[1] == "side"]
    refusals = [e for e in events if e[1] == "refused"]
    bend_rows = []
    for b in bends:
        end = next((e for e in ends if e[0] >= b[0]), None)
        bend_rows.append({
            "t": round(b[0] - t0, 1), "angle_deg": round(math.degrees(b[2])),
            "ended": end[1] if end else "still running",
            "after_s": round(end[0] - b[0], 1) if end else None,
            "why": end[3] if end else "", "avoids_within_window": within(b[0]),
        })
    side_rows = [{
        "t": round(e[0] - t0, 1), "turned": "left" if e[2] > 0 else "right",
        "detail": e[3], "avoids_within_window": within(e[0]),
    } for e in sides]
    return {
        "replay_differing_commands": differing,
        "duration_s": round(duration, 1),
        "window_s": window,
        "avoidances": len(avoid_starts),
        "after_every_turn_in_place": _rate(avoid_starts, turn_ends, window, duration),
        "after_bends": _rate(avoid_starts, [b[0] for b in bends], window, duration),
        "after_side_choices": _rate(avoid_starts, [e[0] for e in sides], window, duration),
        "bends_started": len(bends),
        "bends_ended": dict(Counter(f"{e[1]}: {_reason_kind(e[3])}" if e[3] else e[1] for e in ends)),
        "refusals": dict(Counter(_reason_kind(e[3]) for e in refusals)),
        "refusals_by_bend_side": dict(Counter("left" if e[2] > 0 else "right" for e in refusals)),
        "side_choices": dict(Counter(f"{_reason_kind(e[3])} -> {'left' if e[2] > 0 else 'right'}" for e in sides)),
        "bends": bend_rows,
        "side_choice_list": side_rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    ap.add_argument("--config", "-c", help="TOML file (default: ./brain.toml if present, else defaults)")
    ap.add_argument("--window", type=float, default=6.0, help="seconds after an event counted as 'after' it")
    args = ap.parse_args()
    path = args.config or ("brain.toml" if Path("brain.toml").exists() else None)
    cfg = config_mod.load(path)
    print(json.dumps(report(args.run, cfg, args.window), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
