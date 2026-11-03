"""Going back to where the robot started.

The way back is planned only through cells the trunk has already walked through (explore.py keeps
them, in the odometry frame): every one of them was free at least once, so a path through them
exists by construction, without guessing what lies in the cells it has never seen. Remembered wall
cells are excluded, and a diagonal step needs both of its sides walked too, so the path never cuts a
corner. The brain then steers towards a waypoint a few cells along that path; the ordinary obstacle
logic stays in charge of anything in the way.

"Home" is where the odometry says the robot started. Positions come from odometry, which drifts, so
after a long walk home is only as good as the odometry; the map is forgotten after a fall, and with
it the way home.
"""

from __future__ import annotations

import math
from collections import deque

STEPS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]


def plan(visited, blocked, start: tuple[int, int], goal: tuple[int, int], max_cells: int = 20000,
         clearance: bool = True, _avoid=None):
    """Shortest 8-connected path of cells from ``start`` to ``goal`` through ``visited`` cells,
    avoiding ``blocked`` ones; None if there is none. Breadth-first, with diagonal steps costing the
    same as straight ones (good enough at this scale). With ``clearance``, cells next to a blocked one
    are avoided too, unless that leaves no way at all: a path along the centres of 25 cm cells next
    to a chair leg passes within a body's width of it."""
    # A cell the trunk walked through is free, whatever was mapped in it: with 25 cm cells, the walls
    # beside a path land in the very cells it went through (in the full simulator, 86 of 92 walked
    # cells were also "walls", and there was no way home at all).
    blocked = set(blocked) - set(visited)
    if clearance:
        near = {(c[0] + dx, c[1] + dy) for c in blocked for dx, dy in STEPS}
        path = plan(visited, blocked, start, goal, max_cells, clearance=False, _avoid=near - {start, goal})
        if path is not None:
            return path
    if _avoid:
        blocked = blocked | _avoid             # the clearance pass: walked or not, keep clear of walls
    if start == goal:
        return [start]

    def free(c):
        return (c in visited or c == goal or c == start) and c not in blocked

    prev = {start: None}
    q = deque([start])
    while q and len(prev) < max_cells:
        c = q.popleft()
        for dx, dy in STEPS:
            n = (c[0] + dx, c[1] + dy)
            if n in prev or not free(n):
                continue
            if dx and dy and not (free((c[0] + dx, c[1])) and free((c[0], c[1] + dy))):
                continue                                  # no corner cutting
            prev[n] = c
            if n == goal:
                path = [n]
                while prev[path[-1]] is not None:
                    path.append(prev[path[-1]])
                return path[::-1]
            q.append(n)
    return None


class Homing:
    def __init__(self, cfg) -> None:
        self.cfg = cfg.home
        self.cell = cfg.explore.cell
        self.reset()

    def reset(self) -> None:
        self.home: tuple[float, float] | None = None
        self.active = False
        self.arrived = False
        self.since = 0.0
        self.waypoint: tuple[float, float] | None = None
        self._planned_at = -1e9
        self.suspended_until = -1e9      # homing gives way to an avoidance until then
        self.yield_pending = False       # a stop happened: suspend once it walks again

    def distance(self, pos) -> float | None:
        if self.home is None or pos is None:
            return None
        return math.hypot(pos[0] - self.home[0], pos[1] - self.home[1])

    def bearing(self, explorer, pos, now: float) -> float | None:
        """World bearing (rad) to steer towards now, re-planned every ``replan_every`` seconds."""
        if now - self._planned_at >= self.cfg.replan_every or self.waypoint is None:
            self._planned_at = now
            c = self.cell
            here = (math.floor(pos[0] / c), math.floor(pos[1] / c))
            goal = (math.floor(self.home[0] / c), math.floor(self.home[1] / c))
            path = plan(explorer._visited, explorer._blocked, here, goal)
            if path is None:
                self.waypoint = self.home                  # no known way: head straight for it
            else:
                k = min(len(path) - 1, self.cfg.lookahead_cells)
                if k == len(path) - 1:
                    self.waypoint = self.home
                else:
                    self.waypoint = ((path[k][0] + 0.5) * c, (path[k][1] + 0.5) * c)
        return math.atan2(self.waypoint[1] - pos[1], self.waypoint[0] - pos[0])
