"""A flat 2-D world for closed-loop runs without the simulator: walls as segments, an ideal floor,
the real 8x8 beam geometry and a plant with the measured real/commanded ratios.

It is deliberately simple (no balance, no slip, walls infinitely tall, the sensor level at its
nominal height), so it answers questions about *decisions* — where the brain goes, how much of a
room it covers — not about perception. Coverage uses the same metric as ``record.metrics``.

Benchmark (baseline vs. exploration)::

    uv run python -m tests.world2d --seconds 300
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, field

from microduck_brain.brain import Brain, Inputs
from microduck_brain.config import Config, from_dict
from microduck_brain.depth import BEAMS, DepthFrame

SENSOR_HEIGHT = 0.246
SENSOR_FORWARD = 0.064
SENSOR_LEFT = 0.0224
BODY_RADIUS = 0.09        # trunk-centre clearance to a wall: closer = contact, the move is refused
FORWARD_RATIO = 0.42      # real / commanded forward speed (0.24 commanded ≈ 0.10 m/s, measured)
TURN_RATIO = 0.55         # real / commanded turn rate (46–69% measured)
MAX_RANGE_MM = 4000

Seg = tuple[float, float, float, float]


def box(x0: float, y0: float, x1: float, y1: float) -> list[Seg]:
    return [(x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)]


def _ray_seg(px, py, dx, dy, s: Seg) -> float:
    """Ray parameter t (point = p + t·d) where it meets the segment, inf if it doesn't."""
    x0, y0, x1, y1 = s
    ex, ey = x1 - x0, y1 - y0
    den = dx * ey - dy * ex
    if abs(den) < 1e-12:
        return math.inf
    qx, qy = x0 - px, y0 - py
    t = (qx * ey - qy * ex) / den
    u = (qx * dy - qy * dx) / den
    return t if t > 1e-6 and 0.0 <= u <= 1.0 else math.inf


def _dist_seg(px, py, s: Seg) -> float:
    x0, y0, x1, y1 = s
    ex, ey = x1 - x0, y1 - y0
    L = ex * ex + ey * ey
    u = 0.0 if L == 0 else max(0.0, min(1.0, ((px - x0) * ex + (py - y0) * ey) / L))
    return math.hypot(px - (x0 + u * ex), py - (y0 + u * ey))


@dataclass
class World:
    name: str
    walls: list[Seg]
    start: tuple[float, float, float]           # x, y, yaw

    def render(self, x: float, y: float, yaw: float) -> DepthFrame:
        c, s = math.cos(yaw), math.sin(yaw)
        sx = x + c * SENSOR_FORWARD - s * SENSOR_LEFT
        sy = y + s * SENSOR_FORWARD + c * SENSOR_LEFT
        dist, status = [], []
        for bx, by, bz in BEAMS:
            dx, dy = c * bx - s * by, s * bx + c * by
            t = SENSOR_HEIGHT / -bz if bz < -1e-9 else math.inf
            for w in self.walls:
                t = min(t, _ray_seg(sx, sy, dx, dy, w))
            mm = int(round(t * 1000)) if math.isfinite(t) else MAX_RANGE_MM + 1
            if mm > MAX_RANGE_MM:
                dist.append(0)
                status.append(255)
            else:
                dist.append(mm)
                status.append(5)
        return DepthFrame(8, 8, tuple(dist), tuple(status))

    def clearance(self, x: float, y: float) -> float:
        return min(_dist_seg(x, y, w) for w in self.walls)


def worlds() -> list[World]:
    """Small apartment-like layouts, a few metres across, like the simulator's rooms."""
    out = [World("room 4x3", box(0, 0, 4, 3), (1.0, 1.0, 0.3))]
    # Two rooms joined by a 0.6 m door.
    w = box(0, 0, 6, 3) + [(3, 0, 3, 1.2), (3, 1.8, 3, 3)]
    out.append(World("two rooms + door", w, (1.0, 1.5, 0.0)))
    # L-shaped room.
    out.append(World("L-shaped", [(0, 0, 5, 0), (5, 0, 5, 2), (5, 2, 2, 2), (2, 2, 2, 4), (2, 4, 0, 4), (0, 4, 0, 0)],
                     (0.6, 0.6, 0.8)))
    # A living room with furniture (a sofa, a table, two stools).
    w = box(0, 0, 5, 4) + box(0.0, 3.2, 2.0, 4.0) + box(2.6, 1.4, 3.4, 2.2)
    w += box(1.2, 1.0, 1.45, 1.25) + box(4.0, 3.0, 4.25, 3.25)
    out.append(World("living room", w, (0.5, 0.5, 0.5)))
    return out


@dataclass
class Result:
    world: str
    distance: float = 0.0
    cells: set = field(default_factory=set)
    contacts: int = 0
    stats: object = None
    bend_open: bool = False

    @property
    def area(self) -> float:
        return len(self.cells) * 0.0625


def run(world: World, cfg: Config, seconds: float = 300.0, seed: int = 0, drift: float = 0.04,
        dt: float = 0.05, frame_every: int = 1, recorder=None) -> Result:
    """Closed loop: frame -> brain -> plant. ``drift`` = the gait's yaw drift while walking (rad/s),
    randomised in sign and size per seed, like the real policy's."""
    rng = random.Random(seed)
    x, y, yaw = world.start
    yaw += rng.uniform(-0.5, 0.5)
    bias = drift * rng.uniform(0.5, 1.5) * rng.choice((-1, 1))
    brain = Brain(cfg)
    res = Result(world.name)
    t, n, frame = 0.0, 0, None
    touching = False
    while t < seconds:
        if n % frame_every == 0:
            frame = world.render(x, y, yaw)
        n += 1
        wyaw = (yaw + math.pi) % (2 * math.pi) - math.pi
        inp = Inputs(t, frame, True, "", wyaw, None, (x, y))
        cmd = brain.tick(inp)
        if recorder is not None:
            recorder.write(inp, cmd, brain.state_name)
        if cmd.kind == "move":
            v = FORWARD_RATIO * cmd.vx
            yaw += (TURN_RATIO * cmd.vyaw + (bias if cmd.vx > 0 else 0.0)) * dt
            nx, ny = x + v * math.cos(yaw) * dt, y + v * math.sin(yaw) * dt
            if world.clearance(nx, ny) >= BODY_RADIUS:
                res.distance += math.hypot(nx - x, ny - y)
                x, y = nx, ny
                touching = False
            elif not touching:
                res.contacts += 1
                touching = True
        res.cells.add((math.floor(x / 0.25), math.floor(y / 0.25)))
        t += dt
    res.stats = brain.stats
    res.bend_open = brain.bend is not None
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Coverage benchmark in flat 2-D worlds")
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--seeds", type=int, default=4)
    args = ap.parse_args()
    configs = {"baseline": Config(), "explore": from_dict({"explore": {"enabled": True}})}
    total = {k: [0.0, 0.0, 0] for k in configs}
    for w in worlds():
        for name, cfg in configs.items():
            areas, dists, contacts = [], [], 0
            for seed in range(args.seeds):
                r = run(w, cfg, args.seconds, seed)
                areas.append(r.area)
                dists.append(r.distance)
                contacts += r.contacts
            a, d = sum(areas) / len(areas), sum(dists) / len(dists)
            total[name][0] += a
            total[name][1] += d
            total[name][2] += contacts
            print(f"{w.name:18s} {name:9s} area {a:5.2f} m²  (min {min(areas):5.2f})  walked {d:5.1f} m  contacts {contacts}")
    for name, (a, d, c) in total.items():
        print(f"{'TOTAL':18s} {name:9s} area {a:5.2f} m²  walked {d:5.1f} m  contacts {c}")


if __name__ == "__main__":
    main()
