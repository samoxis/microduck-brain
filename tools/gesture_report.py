"""What the hand gesture did in a full-simulator run (tools/gesture_trials.sh).

    uv run python tools/gesture_report.py runs/<run>

For each time the hand was in front (frames the gesture calls "near"): how long, the state before
and 1.5 s after it was taken away; the gestures the brain counted; how far the trunk really moved
while paused (truth.txt, MEASURE=1); contacts. One line, then the details as JSON with --json.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microduck_brain.brain import Brain  # noqa: E402
from microduck_brain.config import load  # noqa: E402
from microduck_brain.gestures import HandGesture  # noqa: E402
from microduck_brain.record import read  # noqa: E402


def report(d: Path) -> dict:
    cfg = load(d / "brain.toml")
    ticks = list(read(d / "run.jsonl.gz"))
    t0 = ticks[0].inputs.now
    g = HandGesture(cfg)
    segs, cur = [], None
    for t in ticks:
        near = g.near(t.inputs.frame) if t.inputs.frame is not None else (cur is not None)
        if near and cur is None:
            cur = [t.inputs.now, t.inputs.now]
        elif near:
            cur[1] = t.inputs.now
        elif cur is not None and t.inputs.now - cur[1] > 0.3:
            segs.append(cur)
            cur = None
    if cur is not None:
        segs.append(cur)
    times = [t.inputs.now for t in ticks]

    def state_at(x):
        return ticks[min(len(ticks) - 1, bisect.bisect_left(times, x))].state

    b = Brain(cfg)
    for t in ticks:
        b.tick(t.inputs)
    truth = []
    if (d / "truth.txt").exists():
        truth = [tuple(float(v) for v in ln.split()) for ln in (d / "truth.txt").read_text().splitlines() if ln.strip()]
    tt = [r[0] for r in truth]

    def pos(x):
        r = truth[min(len(truth) - 1, bisect.bisect_left(tt, x))]
        return r[1], r[2]

    paused, start = [], None
    for t in ticks:
        if t.state == "paused" and start is None:
            start = t.inputs.now
        elif t.state != "paused" and start is not None:
            paused.append((start, t.inputs.now))
            start = None
    if start is not None:
        paused.append((start, ticks[-1].inputs.now))
    moved = []
    for a, z in paused:
        if truth and z - a > 1.5:
            (x0, y0), (x1, y1) = pos(a + 1.0), pos(z)
            moved.append(round(math.hypot(x1 - x0, y1 - y0), 3))
    contacts = 0
    if (d / "contacts.txt").exists():
        contacts = sum(1 for ln in (d / "contacts.txt").read_text().splitlines() if ln.split()[1:2] != ["hand"])
    return {
        "hand": [{"at_s": round(a - t0, 1), "held_s": round(z - a, 1), "before": state_at(a - 0.5),
                  "after": state_at(z + 1.5)} for a, z in segs],
        "gestures": b.stats.gestures,
        "paused_s": [round(z - a, 1) for a, z in paused],
        "moved_while_paused_m": moved,
        "contacts": contacts,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    ap.add_argument("--json")
    a = ap.parse_args()
    r = report(Path(a.run))
    hands = "; ".join(f"{h['held_s']} s at {h['at_s']} s: {h['before']} -> {h['after']}" for h in r["hand"])
    print(f"gestures {r['gestures']}, hand {hands or 'never seen'}, paused {r['paused_s']} s, "
          f"moved while paused {r['moved_while_paused_m']} m, contacts {r['contacts']}  ({a.run})")
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
