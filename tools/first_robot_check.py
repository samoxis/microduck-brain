"""Checks an observe-mode recording from a physical robot before it is allowed to walk.

    uv run microduck-brain --observe --duration 60 --record first.jsonl.gz   # on a stand / by hand
    uv run python tools/first_robot_check.py first.jsonl.gz [--config brain.toml] [--json out.json]

Each check prints PASS / WARN / FAIL with the number behind it; the exit code is 1 if any FAILs.
The thresholds are what the brain itself assumes (see brain.toml and the README's findings), so a
FAIL means the brain's arithmetic would be wrong on this robot, not just that the run was odd:

  frames       depth frame rate (the brain plans for ~15 Hz), fraction of valid zones
  distances    no zone reads closer than the sensor's minimum, nothing stuck at one value
  pose         robot.state gives the sensor pose and gravity (head-aware geometry, low objects)
  level        gravity ~ straight down while "safe", trunk height near the nominal one
  odometry     yaw and position present and finite (hazard memory, homing, localize)
  replay       replaying with the same config reproduces every command (determinism)
  decisions    what the brain would have done: states, low objects and drops it saw
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microduck_brain.config import load  # noqa: E402
from microduck_brain.record import read, replay  # noqa: E402

VALID = {5, 9}                 # VL53L5CX target status codes that mean "range valid"


def check(path: str, config: str | None) -> list[tuple[str, str, str]]:
    cfg = load(config)
    ticks = list(read(path))
    out: list[tuple[str, str, str]] = []

    def add(name, ok, warn, text):
        out.append((name, "PASS" if ok else ("WARN" if warn else "FAIL"), text))

    if not ticks:
        return [("recording", "FAIL", "no ticks")]
    t0, t1 = ticks[0].inputs.now, ticks[-1].inputs.now
    dur = t1 - t0
    add("recording", dur >= 20, False, f"{len(ticks)} ticks over {dur:.1f} s (need >= 20 s)")

    frames, seen = [], set()
    for t in ticks:
        f = t.inputs.frame
        if f is not None and id(f) not in seen and (not frames or f != frames[-1][1]):
            seen.add(id(f))
            frames.append((t.inputs.now, f))
    rate = (len(frames) - 1) / dur if dur > 0 and len(frames) > 1 else 0.0
    add("frames: rate", rate >= 10, rate >= 5, f"{rate:.1f} Hz (brain assumes ~15; < 10 warns, < 5 fails)")
    if frames:
        shape = Counter((f.rows, f.cols) for _, f in frames)
        add("frames: shape", set(shape) == {(8, 8)}, False, f"{dict(shape)} (need 8x8)")
        valid = [sum(s in VALID for s in f.status) / len(f.status) for _, f in frames]
        v = statistics.median(valid)
        add("frames: valid zones", v >= 0.6, v >= 0.3, f"median {v:.0%} of zones valid per frame")
        d = [mm for _, f in frames for mm, s in zip(f.distance_mm, f.status) if s in VALID]
        if d:
            too_close = sum(mm < 20 for mm in d) / len(d)
            add("distances: plausible", too_close < 0.01, too_close < 0.05,
                f"median {statistics.median(d):.0f} mm, {too_close:.1%} under 20 mm")
            zones = list(zip(*[f.distance_mm for _, f in frames]))
            stuck = sum(1 for z in zones if len(set(z)) == 1)
            add("distances: zones change", stuck <= 4, stuck <= 12,
                f"{stuck} of {len(zones)} zones never changed (a dead or covered zone)")
    else:
        add("frames: any", False, False, "no depth frame at all")

    poses = [t.inputs.pose for t in ticks if t.inputs.pose is not None]
    add("pose: present", len(poses) >= 0.9 * len(ticks), len(poses) > 0,
        f"{len(poses)} of {len(ticks)} ticks have the sensor pose")
    safe = [t for t in ticks if t.inputs.safe and t.inputs.pose is not None]
    if safe:
        tilts = [math.degrees(math.acos(max(-1.0, min(1.0, -p.pose.gravity[2] / max(1e-9, math.hypot(*p.pose.gravity))))))
                 for p in (t.inputs for t in safe)]
        tm = statistics.median(tilts)
        add("level: gravity", tm <= 8, tm <= 15, f"median tilt {tm:.1f} deg while safe (standing ~0-5)")
        zs = [t.inputs.pose.trunk_z for t in safe if t.inputs.pose.trunk_z is not None]
        if zs:
            z = statistics.median(zs)
            add("level: trunk height", 0.08 <= z <= 0.16, 0.05 <= z <= 0.2,
                f"median {z:.3f} m (simulator standing ~0.12)")
    reasons = Counter(t.inputs.safe_reason for t in ticks if not t.inputs.safe)
    unsafe = sum(reasons.values()) / len(ticks)
    add("safety: holds", unsafe <= 0.2, unsafe <= 0.5,
        f"{unsafe:.0%} of ticks not safe: {dict(reasons.most_common(3))}")

    yaws = [t.inputs.yaw for t in ticks if t.inputs.yaw is not None]
    pos = [t.inputs.pos for t in ticks if t.inputs.pos is not None]
    finite = all(math.isfinite(y) for y in yaws) and all(math.isfinite(a) and math.isfinite(b) for a, b in pos)
    add("odometry", finite and len(yaws) >= 0.9 * len(ticks) and len(pos) >= 0.9 * len(ticks), bool(yaws),
        f"yaw in {len(yaws)}, position in {len(pos)} of {len(ticks)} ticks, all finite: {finite}")

    rep = replay(path, cfg)
    add("replay", rep.differing == 0, False, f"{rep.differing} differing commands with this config")

    states = Counter(t.state for t in ticks)
    st = rep.brain.stats
    extra = f", low objects {st.low_seen}, drops {st.drops_seen}"
    out.append(("decisions", "INFO", f"states {dict(states.most_common())}{extra}"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    ap.add_argument("--config", "-c")
    ap.add_argument("--json")
    a = ap.parse_args()
    res = check(a.run, a.config)
    for name, verdict, text in res:
        print(f"{verdict:5} {name:24} {text}")
    if a.json:
        Path(a.json).write_text(json.dumps([dict(zip(("check", "verdict", "detail"), r)) for r in res], indent=2))
    return 1 if any(v == "FAIL" for _, v, _ in res) else 0


if __name__ == "__main__":
    sys.exit(main())
