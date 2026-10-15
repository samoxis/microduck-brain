"""Record a run tick by tick, replay it through the brain, and measure what it covered.

A recording is JSON Lines (optionally gzipped, by the ``.gz`` suffix): a header line, then one line
per brain tick with everything ``Brain.tick`` saw and what it answered. Because the brain is a pure
function of its inputs, a recording made on a real robot or in the full simulator can be replayed
offline with a different configuration — the question "what would this change have done on that
run?" gets answered without the robot.

To keep files small, a depth frame or a pose that did not change since the previous tick is not
written again (key omitted); ``null`` means "none".
"""

from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .brain import Brain, Command, Inputs
from .config import Config
from .depth import BEAMS, DepthFrame, Pose

FORMAT = 1
HEADER = {"kind": "microduck-brain-run", "format": FORMAT}


def _open(path: str | Path, mode: str):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def _pose_dict(p: Pose) -> dict:
    return {"pos": list(p.sensor_pos), "quat": list(p.sensor_quat), "g": list(p.gravity), "z": p.trunk_z}


class Recorder:
    """Writes one line per tick. ``Runner`` calls ``write`` and ``close``."""

    def __init__(self, path: str | Path, meta: dict | None = None) -> None:
        self._fh = _open(path, "w")
        self._fh.write(json.dumps({**HEADER, **(meta or {})}) + "\n")
        self._last_frame: object = None
        self._last_pose: object = None
        self._first = True

    def write(self, inp: Inputs, cmd: Command, state: str) -> None:
        rec: dict = {
            "t": round(inp.now, 4),
            "safe": inp.safe,
            "yaw": inp.yaw,
            "pos": list(inp.pos) if inp.pos is not None else None,
            "cmd": [cmd.kind, round(cmd.vx, 5), round(cmd.vyaw, 5)],
            "state": state,
        }
        if inp.safe_reason:
            rec["why"] = inp.safe_reason
        if self._first or inp.frame is not self._last_frame:
            rec["f"] = None if inp.frame is None else {"d": list(inp.frame.distance_mm), "s": list(inp.frame.status)}
            self._last_frame = inp.frame
        if self._first or inp.pose is not self._last_pose:
            rec["p"] = None if inp.pose is None else _pose_dict(inp.pose)
            self._last_pose = inp.pose
        self._first = False
        self._fh.write(json.dumps(rec, separators=(",", ":")) + "\n")

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


@dataclass(frozen=True)
class Tick:
    inputs: Inputs
    cmd: Command
    state: str


def read(path: str | Path) -> Iterator[Tick]:
    """The recorded ticks, with frames and poses restored."""
    frame: DepthFrame | None = None
    pose: Pose | None = None
    with _open(path, "r") as fh:
        header = json.loads(fh.readline())
        if header.get("kind") != HEADER["kind"] or header.get("format") != FORMAT:
            raise ValueError(f"{path}: not a microduck-brain recording (format {FORMAT})")
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if "f" in r:
                f = r["f"]
                frame = None if f is None else DepthFrame(8, 8, tuple(f["d"]), tuple(f["s"]))
            if "p" in r:
                p = r["p"]
                pose = None if p is None else Pose(tuple(p["pos"]), tuple(p["quat"]), tuple(p["g"]), p["z"])
            inp = Inputs(
                now=r["t"], frame=frame, safe=r["safe"], safe_reason=r.get("why", ""),
                yaw=r["yaw"], pose=pose, pos=tuple(r["pos"]) if r["pos"] is not None else None,
            )
            kind, vx, vyaw = r["cmd"]
            yield Tick(inp, Command(kind, vx, vyaw), r["state"])


@dataclass
class Replay:
    brain: Brain
    ticks: int = 0
    differing: int = 0                     # ticks where the replayed command differs from the recorded one
    first_difference: float | None = None  # time of the first one
    hazard_points: dict[str, set] = field(default_factory=dict)   # kind -> {(x, y)} remembered at any time


def replay(path: str | Path, cfg: Config) -> Replay:
    """Run a fresh brain with ``cfg`` over a recording and compare its commands with the recorded ones."""
    out = Replay(Brain(cfg))
    for tick in read(path):
        cmd = out.brain.tick(tick.inputs)
        out.ticks += 1
        same = cmd.kind == tick.cmd.kind and abs(cmd.vx - tick.cmd.vx) < 1e-4 and abs(cmd.vyaw - tick.cmd.vyaw) < 1e-4
        if not same:
            out.differing += 1
            if out.first_difference is None:
                out.first_difference = tick.inputs.now
        for kind, k in out.brain.hazards._kinds.items():
            out.hazard_points.setdefault(kind, set()).update((round(p[0], 2), round(p[1], 2)) for p in k.points)
    return out


def metrics(ticks: list[Tick], cell: float = 0.25) -> dict:
    """Distance walked, area covered (grid cells visited) and bounding box, from odometry."""
    distance = 0.0
    cells: set[tuple[int, int]] = set()
    xs, ys = [], []
    prev = None
    for t in ticks:
        p = t.inputs.pos
        if p is None:
            prev = None
            continue
        if prev is not None:
            step = math.hypot(p[0] - prev[0], p[1] - prev[1])
            if step < 0.5:                 # ignore jumps (odometry reset, reconnect)
                distance += step
        prev = p
        cells.add((math.floor(p[0] / cell), math.floor(p[1] / cell)))
        xs.append(p[0])
        ys.append(p[1])
    duration = ticks[-1].inputs.now - ticks[0].inputs.now if len(ticks) > 1 else 0.0
    return {
        "duration_s": round(duration, 1),
        "distance_m": round(distance, 2),
        "cells_visited": len(cells),
        "area_covered_m2": round(len(cells) * cell * cell, 2),
        "bounding_box_m": [round(max(xs) - min(xs), 2), round(max(ys) - min(ys), 2)] if xs else [0.0, 0.0],
    }


def wall_points(ticks: list[Tick], cfg: Config, every: int = 4) -> list[tuple[float, float]]:
    """Where the upper (wall) rows hit something, in the odometry frame — a rough map of the room
    for drawing. Uses the current heading and position; ignores the small head pitch."""
    pts = []
    last_frame = None
    n = 0
    for t in ticks:
        inp = t.inputs
        if inp.frame is None or inp.frame is last_frame or inp.pos is None or inp.yaw is None:
            continue
        last_frame = inp.frame
        n += 1
        if n % every:
            continue
        sx = inp.pose.sensor_pos[0] if inp.pose else cfg.low.nominal_sensor_forward
        c, s = math.cos(inp.yaw), math.sin(inp.yaw)
        o = cfg.obstacle
        for r in range(o.first_row, o.last_row + 1):
            for col in range(8):
                d = inp.frame.zone(r, col, math.inf)
                if d is None or not math.isfinite(d) or d > 3.0:
                    continue
                bx, by, _ = BEAMS[r * 8 + col]
                fx, fy = sx + d * bx, d * by
                pts.append((inp.pos[0] + c * fx - s * fy, inp.pos[1] + s * fx + c * fy))
    return pts
