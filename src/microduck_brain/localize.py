"""Correcting odometry drift by matching what the wall rows see against the walls seen before.

The odometry (``odom.position``, ``odom.yaw``) drifts. Everything the brain remembers lives in its
frame, so after a long walk the map, the hazards and "home" drift with it. This keeps a fine grid
of where walls were seen, and for every few frames asks: which small shift and turn of the current
pose puts this frame's wall returns best onto walls seen *a while ago*? If one does clearly better
than no change, half of it is applied to a running correction:

    corrected position = R(d_yaw) · odom position + (dx, dy),   corrected yaw = odom yaw + d_yaw

Only walls seen at least ``min_age`` seconds earlier count, or a frame would always match the walls
it has just drawn. Like the rest of the brain, this is a function of its inputs, so it replays.
"""

from __future__ import annotations

import math

CELL = 0.05               # the matching grid (m)
CAP = 0.10                # a return further than this from any old wall counts as this far
MAX_CELLS = 60000
RANGE = 3.5               # wall returns further than this are not used (pose error grows with range)


class Localizer:
    def __init__(self, cfg) -> None:
        self.cfg = cfg.localize
        self.obstacle = cfg.obstacle
        self.reset()

    def reset(self) -> None:
        self.dx = self.dy = self.dyaw = 0.0
        self._map: dict[tuple[int, int], float] = {}   # cell -> when first seen
        self._last_frame: object = None
        self._frames = 0
        self.corrections = 0

    # --- the correction ------------------------------------------------------------
    def correct(self, pos, yaw):
        """(corrected position, corrected yaw) for raw odometry."""
        if pos is None:
            return None, (None if yaw is None else yaw + self.dyaw)
        c, s = math.cos(self.dyaw), math.sin(self.dyaw)
        p = (c * pos[0] - s * pos[1] + self.dx, s * pos[0] + c * pos[1] + self.dy)
        return p, (None if yaw is None else yaw + self.dyaw)

    # --- per frame ------------------------------------------------------------------
    def _points(self, frame, sensor_forward: float, head: float):
        """Wall-row returns in the trunk frame: (forward, left)."""
        from .depth import BEAMS, rotate

        o = self.obstacle
        out = []
        for r in range(o.first_row, o.last_row + 1):
            for col in range(frame.cols):
                d = frame.zone(r, col, math.inf)
                if d is None or not math.isfinite(d) or d > RANGE:
                    continue
                bx, by, _ = BEAMS[r * frame.cols + col]
                fx, fy = rotate(d * bx, d * by, head)
                out.append((sensor_forward + fx, fy + o.sensor_lateral))
        return out

    @staticmethod
    def _key(x: float, y: float) -> tuple[int, int]:
        return (math.floor(x / CELL), math.floor(y / CELL))

    def _cost(self, pts, px, py, yaw, now) -> tuple[float, int]:
        """Mean distance from each return to the nearest old wall cell, capped at CAP (chamfer), and
        how many returns had an old wall within reach. Flat along a wall, so sliding along one
        gains nothing: only what the walls actually pin down moves the correction."""
        c, s = math.cos(yaw), math.sin(yaw)
        old = now - self.cfg.min_age
        total, near = 0.0, 0
        for fx, fy in pts:
            wx, wy = px + c * fx - s * fy, py + s * fx + c * fy
            kx, ky = self._key(wx, wy)
            best = CAP
            for i in range(kx - 2, kx + 3):
                for j in range(ky - 2, ky + 3):
                    t = self._map.get((i, j))
                    if t is not None and t <= old:
                        d = math.hypot((i + 0.5) * CELL - wx, (j + 0.5) * CELL - wy)
                        if d < best:
                            best = d
            total += best
            if best < CAP:
                near += 1
        return total / len(pts), near

    def update(self, frame, pos, yaw, sensor_forward: float, now: float, head: float = 0.0) -> None:
        """Match a new frame (every ``every`` frames), then add its walls to the map."""
        if not self.cfg.enabled or frame is None or frame is self._last_frame or pos is None or yaw is None:
            return
        self._last_frame = frame
        self._frames += 1
        pts = self._points(frame, sensor_forward, head)
        if not pts:
            return
        (px, py), cyaw = self.correct(pos, yaw)
        if self._frames % self.cfg.every == 0:
            base, near = self._cost(pts, px, py, cyaw, now)
            if near >= self.cfg.min_points:      # enough of this frame lands near walls seen before
                best = (base, 0.0, 0.0, 0.0)
                step, turn = self.cfg.search_step, math.radians(self.cfg.search_turn_deg)
                for dth in (-turn, 0.0, turn):
                    for ex in (-step, 0.0, step):
                        for ey in (-step, 0.0, step):
                            if dth == ex == ey == 0.0:
                                continue
                            cost, _ = self._cost(pts, px + ex, py + ey, cyaw + dth, now)
                            if cost < best[0]:
                                best = (cost, ex, ey, dth)
                cost, ex, ey, dth = best
                if base - cost >= self.cfg.min_gain_m and base - cost >= self.cfg.min_gain_rel * base:
                    # Half of the step: the robot's corrected position moves by e/2 and its
                    # corrected yaw by dth/2; the correction (R(dyaw), d) is re-solved for that.
                    h = 0.5
                    self.dyaw += dth * h
                    c2, s2 = math.cos(self.dyaw), math.sin(self.dyaw)
                    tx, ty = px + ex * h, py + ey * h
                    self.dx = tx - (c2 * pos[0] - s2 * pos[1])
                    self.dy = ty - (s2 * pos[0] + c2 * pos[1])
                    self.corrections += 1
                    (px, py), cyaw = self.correct(pos, yaw)
        c, s = math.cos(cyaw), math.sin(cyaw)
        for fx, fy in pts:
            k = self._key(px + c * fx - s * fy, py + s * fx + c * fy)
            if k not in self._map:
                self._map[k] = now
                if len(self._map) > MAX_CELLS:
                    self._map.pop(next(iter(self._map)))
