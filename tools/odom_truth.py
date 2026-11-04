"""How far the odometry drifted from where the robot really was, in a full-simulator run.

Needs the run's recording and a truth file written by tools/push/sitecustomize.py (DUCK_TRUTH_FILE:
"monotonic_s x y yaw sim_time" of the trunk, every 0.25 s). The first column is the same clock as the
brain's (CLOCK_MONOTONIC, one machine), so the two line up in time directly; the truth is put into the
odometry's frame with the pose at the recording's first sample, where the odometry is still exact.

    uv run python tools/odom_truth.py runs/<run>/run.jsonl.gz runs/<run>/truth.txt [--json out.json]
    uv run python tools/odom_truth.py ... --localize     # also the corrected odometry ([localize])

With --localize the recording is replayed through a brain with [localize] on, and its corrected pose
is compared with the truth too. --drift YAW_RATE SCALE first degrades the recorded odometry the way a
slipping robot's would (its heading gains YAW_RATE rad/s, its distances are scaled by SCALE), on the
real frames and the real truth, to see what the correction does with drift the simulator lacks. The correction depends only on the recorded frames and odometry,
not on what the brain then did, so this measures it on the very same run.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microduck_brain.record import read  # noqa: E402


def _start(samples, eps=0.02):
    """Index of the first sample more than eps (m) from the first one."""
    x0, y0 = samples[0][1], samples[0][2]
    for i, s in enumerate(samples):
        if math.hypot(s[1] - x0, s[2] - y0) > eps:
            return max(0, i - 1)
    return 0


def align(odom, truth):
    """Truth samples in the odometry's frame: [(t, x, y, yaw)], posed on the first odometry sample."""
    to0, xo, yo, ao = odom[0]
    times = [s[0] for s in truth]
    it = min(len(truth) - 1, bisect.bisect_left(times, to0))
    _, xt, yt, at = truth[it][:4]
    d = ao - at
    c, s = math.cos(d), math.sin(d)
    out = []
    for t, x, y, a, *_ in truth:
        dx, dy = x - xt, y - yt
        out.append((t, xo + c * dx - s * dy, yo + s * dx + c * dy, a + d))
    return out


def drifted_ticks(run: str, yaw_rate: float, scale: float):
    """The recording's ticks with odometry degraded: re-integrated from its own increments."""
    from dataclasses import replace

    prev = None
    ox = oy = oa = None
    for t in read(run):
        inp = t.inputs
        if inp.pos is None or inp.yaw is None:
            yield replace(t, inputs=replace(inp, pos=None, yaw=None))
            continue
        if prev is None:
            ox, oy, oa = inp.pos[0], inp.pos[1], inp.yaw
        else:
            dx, dy = inp.pos[0] - prev[0], inp.pos[1] - prev[1]
            da = (inp.yaw - prev[2] + math.pi) % (2 * math.pi) - math.pi
            step = math.hypot(dx, dy) * scale
            oa_prev = oa
            oa = oa + da + yaw_rate * (inp.now - prev[3])
            # the step in the direction the true odometry moved, rotated by the accumulated error
            heading = math.atan2(dy, dx) + (oa_prev - prev[2]) if step > 0 else 0.0
            ox, oy = ox + step * math.cos(heading), oy + step * math.sin(heading)
        prev = (inp.pos[0], inp.pos[1], inp.yaw, inp.now)
        wy = (oa + math.pi) % (2 * math.pi) - math.pi
        yield replace(t, inputs=replace(inp, pos=(ox, oy), yaw=wy))


def corrected(run: str, ticks=None, settings: dict | None = None) -> list:
    from microduck_brain.brain import Brain
    from microduck_brain.config import from_dict

    brain = Brain(from_dict({"localize": {"enabled": True, **(settings or {})}}))
    out = []
    for t in (ticks if ticks is not None else read(run)):
        brain.tick(t.inputs)
        inp = t.inputs
        if inp.pos is None or inp.yaw is None:
            continue
        p, a = brain.localizer.correct(inp.pos, inp.yaw)
        out.append((inp.now, p[0], p[1], a))
    return out


def drift(run: str, truth_file: str, odom=None) -> dict:
    odom = odom or [(t.inputs.now, t.inputs.pos[0], t.inputs.pos[1], t.inputs.yaw) for t in read(run)
                    if t.inputs.pos is not None and t.inputs.yaw is not None]
    truth = [tuple(float(v) for v in line.split()) for line in Path(truth_file).read_text().splitlines() if line.strip()]
    tr = align(odom, truth)
    times = [s[0] for s in tr]
    errs, yaws, rows = [], [], []
    for i, (t, x, y, a) in enumerate(odom):
        j = bisect.bisect_left(times, t)
        if j <= 0 or j >= len(tr):
            continue
        _, tx, ty, ta = tr[j]
        e = math.hypot(x - tx, y - ty)
        ey = abs((a - ta + math.pi) % (2 * math.pi) - math.pi)
        errs.append(e)
        yaws.append(ey)
        if i % 200 == 0:
            rows.append({"t": round(t - odom[0][0], 1), "pos_err_m": round(e, 3), "yaw_err_deg": round(math.degrees(ey), 1)})
    if not errs:
        return {"error": "no overlap between the run and the truth file"}
    return {
        "samples": len(errs),
        "position_error_m": {"final": round(errs[-1], 3), "max": round(max(errs), 3), "mean": round(sum(errs) / len(errs), 3)},
        "yaw_error_deg": {"final": round(math.degrees(yaws[-1]), 1), "max": round(math.degrees(max(yaws)), 1)},
        "true_final_position_odom_frame": [round(v, 3) for v in tr[min(len(tr) - 1, bisect.bisect_left(times, odom[-1][0]))][1:3]],
        "every_10s": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    ap.add_argument("truth")
    ap.add_argument("--json")
    ap.add_argument("--localize", action="store_true", help="also the odometry corrected by [localize]")
    ap.add_argument("--settings", default="{}", help='[localize] settings for --localize, as JSON, e.g. {"method": "icp"}')
    ap.add_argument("--drift", nargs=2, type=float, metavar=("YAW_RATE", "SCALE"),
                    help="degrade the recorded odometry first (rad/s added to the heading, distance scale)")
    args = ap.parse_args()
    if args.drift:
        ticks = list(drifted_ticks(args.run, *args.drift))
        odom = [(t.inputs.now, t.inputs.pos[0], t.inputs.pos[1], t.inputs.yaw) for t in ticks
                if t.inputs.pos is not None and t.inputs.yaw is not None]
        res = drift(args.run, args.truth, odom)
        if args.localize:
            res["corrected"] = drift(args.run, args.truth, corrected(args.run, ticks, json.loads(args.settings)))
    else:
        res = drift(args.run, args.truth)
        if args.localize:
            res["corrected"] = drift(args.run, args.truth, corrected(args.run, None, json.loads(args.settings)))
    print(json.dumps({k: (v if k != "corrected" else {kk: vv for kk, vv in v.items() if kk != "every_10s"})
                      for k, v in res.items() if k != "every_10s"}, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
