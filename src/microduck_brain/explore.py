"""Exploration: prefer the directions the robot has not been to yet.

Without it the brain only reacts: it walks straight, and at each obstacle turns towards the freer
side. In a room that settles into a few lanes it keeps walking again and again. With it, the brain
remembers which grid cells (odometry frame) the trunk has passed through, and

* when both sides are about equally free, turns towards the side with more unvisited cells;
* while walking with a clear way, now and then bends its heading towards the least visited direction
  within ``max_bend_deg`` (the obstacle logic still decides whether that direction is walkable).

"Unvisited" means reachable and not walked yet: the wall returns of the upper rows are remembered as
blocked cells, and a probe stops at the first one. Without that, the cells behind a wall looked like
the most novel place of all, and the brain kept wanting to bend into the wall next to it (in the
2-D worlds, 94% of refused bends pointed at directions whose probed cells were all behind a wall).

Everything here is a function of ``odom.position`` and ``odom.yaw`` as they arrive in ``Inputs``,
so recordings still replay exactly. The map is forgotten after a fall, like the hazards: the
odometry's position can no longer be trusted.
"""

from __future__ import annotations

import math

MAX_CELLS = 20000            # bounded memory: the oldest cells go first
PROBE_DISTANCES = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
WALL_RANGE = 2.5             # wall returns further than this are not mapped (pose error grows with range)


class Explorer:
    def __init__(self, cfg) -> None:
        self.cfg = cfg.explore
        self._visited: dict[tuple[int, int], float] = {}
        self._blocked: dict[tuple[int, int], float] = {}
        self._last_frame: object = None

    def forget(self) -> None:
        self._visited.clear()
        self._blocked.clear()
        self._last_frame = None

    @property
    def blocked_cells(self) -> int:
        return len(self._blocked)

    def update_walls(self, frame, pos, yaw: float | None, sensor_forward: float, obstacle_cfg,
                     head: float = 0.0, geom=None) -> None:
        """Remember the cells where the wall rows of a new frame hit something. ``head`` = where the
        head looks, relative to the trunk (rad, + = left)."""
        if frame is None or frame is self._last_frame or pos is None or yaw is None:
            return
        self._last_frame = frame
        from .depth import BEAMS, rotate

        c, s = math.cos(yaw), math.sin(yaw)
        o = obstacle_cfg
        for r in range(o.first_row, o.last_row + 1):
            for col in range(frame.cols):
                d = frame.zone(r, col, math.inf)
                if d is None or not math.isfinite(d) or d > WALL_RANGE:
                    continue
                i = r * frame.cols + col
                if geom is not None:
                    if geom.not_an_obstacle(i, d):
                        continue                  # the floor, seen by a pitched head
                    bx, by, _ = geom.beams[i]
                    fx, fy = d * bx, d * by
                else:
                    bx, by, _ = BEAMS[i]
                    fx, fy = rotate(d * bx, d * by, head)
                fx, fy = sensor_forward + fx, fy + o.sensor_lateral
                key = self._cell(pos[0] + c * fx - s * fy, pos[1] + s * fx + c * fy)
                self._blocked.pop(key, None)
                self._blocked[key] = 0.0
                if len(self._blocked) > MAX_CELLS:
                    self._blocked.pop(next(iter(self._blocked)))

    @property
    def cells(self) -> int:
        return len(self._visited)

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        c = self.cfg.cell
        return (math.floor(x / c), math.floor(y / c))

    def update(self, pos, now: float) -> None:
        if pos is None:
            return
        key = self._cell(*pos)
        self._visited.pop(key, None)          # re-insert: the dict stays ordered by last visit
        self._visited[key] = now
        if len(self._visited) > MAX_CELLS:
            self._visited.pop(next(iter(self._visited)))

    def wall_beside(self, pos, yaw: float, side: float, reach: float = 0.45) -> bool:
        """A remembered wall cell on that side (+1 left, -1 right) within ``reach`` of the trunk,
        from a little behind to a little ahead: what a bend on that side would sweep the body over.
        The map outlasts the 3 s side memory and covers returns that were wider than its band."""
        cs = self.cfg.cell
        n = int(math.ceil(reach / cs)) + 1
        cx, cy = self._cell(*pos)
        c, s = math.cos(yaw), math.sin(yaw)
        for i in range(cx - n, cx + n + 1):
            for j in range(cy - n, cy + n + 1):
                if (i, j) not in self._blocked:
                    continue
                dx, dy = (i + 0.5) * cs - pos[0], (j + 0.5) * cs - pos[1]
                fwd, lat = c * dx + s * dy, -s * dx + c * dy
                if -0.2 < fwd < 0.5 and 0.0 < side * lat < reach:
                    return True
        return False

    def nearest_beside(self, pos, yaw: float, reach: float = 0.45) -> float | None:
        """Lateral offset (m, + = left) of the nearest remembered wall cell beside the trunk, from a
        little behind to a little ahead; None if there is none."""
        cs = self.cfg.cell
        n = int(math.ceil(reach / cs)) + 1
        cx, cy = self._cell(*pos)
        c, s = math.cos(yaw), math.sin(yaw)
        best = None
        for i in range(cx - n, cx + n + 1):
            for j in range(cy - n, cy + n + 1):
                if (i, j) not in self._blocked:
                    continue
                dx, dy = (i + 0.5) * cs - pos[0], (j + 0.5) * cs - pos[1]
                fwd, lat = c * dx + s * dy, -s * dx + c * dy
                if -0.2 < fwd < 0.5 and abs(lat) < reach:
                    d = math.hypot(fwd, lat)
                    if best is None or d < best[0]:
                        best = (d, lat)
        return None if best is None else best[1]

    def wall_on_course(self, pos, heading: float, half_width: float, reach: float = 0.75) -> bool:
        """A remembered wall cell in the body's corridor along ``heading``, within ``reach``: a bend
        onto that course would walk into something the sensor may see with a single zone only (the
        edge of a table corner, head on)."""
        c, s = math.cos(heading), math.sin(heading)
        d = 0.25
        while d <= reach + 1e-9:
            for off in (-half_width, 0.0, half_width):
                if self._cell(pos[0] + d * c - off * s, pos[1] + d * s + off * c) in self._blocked:
                    return True
            d += self.cfg.cell / 2
        return False

    def novelty(self, pos, yaw: float, angle: float) -> float:
        """Fraction of the probed cells along ``yaw + angle`` that are reachable and unvisited (0..1).
        A probe stops at the first blocked cell: what lies behind a wall counts as nothing."""
        a = yaw + angle
        c, s = math.cos(a), math.sin(a)
        here = self._cell(*pos)
        cells: list[tuple[int, int]] = []
        for d in PROBE_DISTANCES:
            k = self._cell(pos[0] + d * c, pos[1] + d * s)
            if k in cells:
                continue
            if k in self._blocked and k != here:
                break
            cells.append(k)
        n = len({self._cell(pos[0] + d * c, pos[1] + d * s) for d in PROBE_DISTANCES})
        return sum(1 for k in cells if k not in self._visited) / n

    def side_novelty(self, pos, yaw: float) -> tuple[float, float]:
        """Mean novelty to the left and to the right (30°..120° off the heading)."""
        angles = [math.radians(a) for a in (30, 60, 90, 120)]
        left = sum(self.novelty(pos, yaw, a) for a in angles) / len(angles)
        right = sum(self.novelty(pos, yaw, -a) for a in angles) / len(angles)
        return left, right

    def best_bend(self, pos, yaw: float) -> float | None:
        """The direction (rad, relative to the heading, within ±max_bend_deg) clearly more novel than
        straight ahead, or None if straight ahead is as good."""
        ahead = self.novelty(pos, yaw, 0.0)
        best, best_angle = ahead, None
        m = self.cfg.max_bend_deg
        for deg in (-m, -m / 2, m / 2, m):
            n = self.novelty(pos, yaw, math.radians(deg))
            if n > best + 1e-9:
                best, best_angle = n, math.radians(deg)
        if best_angle is None or best - ahead < self.cfg.novelty_margin:
            return None
        return best_angle
