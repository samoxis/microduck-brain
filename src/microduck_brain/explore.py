"""Exploration: prefer the directions the robot has not been to yet.

Without it the brain only reacts: it walks straight, and at each obstacle turns towards the freer
side. In a room that settles into a few lanes it keeps walking again and again. With it, the brain
remembers which grid cells (odometry frame) the trunk has passed through, and

* when both sides are about equally free, turns towards the side with more unvisited cells;
* while walking with a clear way, now and then bends its heading towards the least visited direction
  within ``max_bend_deg`` (the obstacle logic still decides whether that direction is walkable).

Everything here is a function of ``odom.position`` and ``odom.yaw`` as they arrive in ``Inputs``,
so recordings still replay exactly. The map is forgotten after a fall, like the hazards: the
odometry's position can no longer be trusted.
"""

from __future__ import annotations

import math

MAX_CELLS = 20000            # bounded memory: the oldest cells go first
PROBE_DISTANCES = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)


class Explorer:
    def __init__(self, cfg) -> None:
        self.cfg = cfg.explore
        self._visited: dict[tuple[int, int], float] = {}

    def forget(self) -> None:
        self._visited.clear()

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

    def novelty(self, pos, yaw: float, angle: float) -> float:
        """Fraction of unvisited cells along the direction ``yaw + angle`` (0..1)."""
        a = yaw + angle
        c, s = math.cos(a), math.sin(a)
        seen = {self._cell(pos[0] + d * c, pos[1] + d * s) for d in PROBE_DISTANCES}
        return sum(1 for k in seen if k not in self._visited) / len(seen)

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
