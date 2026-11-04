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
        # cell -> [first seen, sum x, sum y, n]: distances are measured to the mean of the actual wall
        # points in a cell, not to its centre (that rewarded snapping points to the grid: with good
        # odometry, every "correction" made it worse)
        self._map: dict[tuple[int, int], list] = {}
        self._last_frame: object = None
        self._frames = 0
        self.corrections = 0
        self._prev: tuple[float, float] | None = None   # (t, yaw) of the last tick, for the turn rate
        self._pts: dict[tuple[int, int], list] = {}       # method "icp": 10 cm cell -> [(x, y, t)]

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
                    m = self._map.get((i, j))
                    if m is not None and m[0] <= old:
                        d = math.hypot(m[1] / m[3] - wx, m[2] / m[3] - wy)
                        if d < best:
                            best = d
            total += best
            if best < CAP:
                near += 1
        return total / len(pts), near

    def update(self, frame, pos, yaw, sensor_forward: float, now: float, head: float = 0.0) -> None:
        """Match a new frame (every ``every`` frames), then add its walls to the map."""
        if not self.cfg.enabled or pos is None or yaw is None:
            return
        prev, self._prev = self._prev, (now, yaw)
        if prev is not None and now > prev[0]:
            rate = abs((yaw - prev[1] + math.pi) % (2 * math.pi) - math.pi) / (now - prev[0])
            if rate > self.cfg.max_turn_rate:
                # Turning: the frame is up to a frame period older than the pose, so its walls come
                # out rotated by several degrees: neither matched nor drawn.
                self._last_frame = frame
                return
        if frame is None or frame is self._last_frame:
            return
        self._last_frame = frame
        self._frames += 1
        pts = self._points(frame, sensor_forward, head)
        if not pts:
            return
        (px, py), cyaw = self.correct(pos, yaw)
        if self.cfg.method == "icp":
            self._icp(pts, pos, yaw, px, py, cyaw, now)
            return
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
            wx, wy = px + c * fx - s * fy, py + s * fx + c * fy
            k = self._key(wx, wy)
            m = self._map.get(k)
            if m is None:
                self._map[k] = [now, wx, wy, 1]
                if len(self._map) > MAX_CELLS:
                    self._map.pop(next(iter(self._map)))
            elif m[3] < 50:
                m[1] += wx
                m[2] += wy
                m[3] += 1


    # --- method "icp": point to point, no grid ------------------------------------------------
    ICP_CELL = 0.10
    ICP_REACH = 0.15          # a new return pairs with the nearest old one within this (overridable)
    ICP_GAIN = 0.3            # fraction of the solved step applied
    ICP_MAX_STEP = 0.015      # m per update
    ICP_MAX_TURN = math.radians(0.5)

    def _icp(self, pts, pos, yaw, px, py, cyaw, now) -> None:
        """Pair each wall return with the nearest point seen at least min_age ago, solve the rigid
        motion that best aligns the pairs (closed form, least squares), apply a damped, clamped
        part of it -- if it actually lowers the pairs' mean distance -- then store the returns."""
        cs = self.ICP_CELL
        old = now - self.cfg.min_age
        c, s = math.cos(cyaw), math.sin(cyaw)
        world = [(px + c * fx - s * fy, py + s * fx + c * fy) for fx, fy in pts]
        if self._frames % self.cfg.every == 0:
            pairs = []
            for wx, wy in world:
                kx, ky = math.floor(wx / cs), math.floor(wy / cs)
                best, bq = getattr(self.cfg, 'icp_reach', self.ICP_REACH), None
                for i in (kx - 1, kx, kx + 1):
                    for j in (ky - 1, ky, ky + 1):
                        for qx, qy, t in self._pts.get((i, j), ()):
                            if t <= old:
                                d = math.hypot(qx - wx, qy - wy)
                                if d < best:
                                    best, bq = d, (qx, qy)
                if bq is not None:
                    pairs.append(((wx, wy), bq))
            if len(pairs) >= self.cfg.min_points:
                n = len(pairs)
                mx = sum(a[0] for a, _ in pairs) / n
                my = sum(a[1] for a, _ in pairs) / n
                nx = sum(b[0] for _, b in pairs) / n
                ny = sum(b[1] for _, b in pairs) / n
                sxx = sum((a[0] - mx) * (b[0] - nx) for a, b in pairs)
                syy = sum((a[1] - my) * (b[1] - ny) for a, b in pairs)
                sxy = sum((a[0] - mx) * (b[1] - ny) for a, b in pairs)
                syx = sum((a[1] - my) * (b[0] - nx) for a, b in pairs)
                th = math.atan2(sxy - syx, sxx + syy)
                ct, st = math.cos(th), math.sin(th)
                tx, ty = nx - (ct * mx - st * my), ny - (st * mx + ct * my)

                def cost(a_th, a_tx, a_ty):
                    ca, sa = math.cos(a_th), math.sin(a_th)
                    return sum(math.hypot(ca * a[0] - sa * a[1] + a_tx - b[0], sa * a[0] + ca * a[1] + a_ty - b[1])
                               for a, b in pairs) / n

                # The damped step, as a motion of the robot: rotate by g*th about the robot, then
                # move it as the solved transform moves the robot's own position, scaled by g.
                g = getattr(self.cfg, 'icp_gain', self.ICP_GAIN)
                dth = max(-self.ICP_MAX_TURN, min(self.ICP_MAX_TURN, g * th))
                rx, ry = ct * px - st * py + tx - px, st * px + ct * py + ty - py
                mv = math.hypot(rx, ry) * g
                if mv > self.ICP_MAX_STEP:
                    rx, ry = rx * self.ICP_MAX_STEP / (mv / g), ry * self.ICP_MAX_STEP / (mv / g)
                else:
                    rx, ry = rx * g, ry * g
                # the same small motion applied to the frame's points, to check it helps
                cd, sd = math.cos(dth), math.sin(dth)
                a_tx = px + rx - (cd * px - sd * py)
                a_ty = py + ry - (sd * px + cd * py)
                if cost(dth, a_tx, a_ty) + self.cfg.min_gain_m * 0.25 < cost(0.0, 0.0, 0.0):
                    self.dyaw += dth
                    c2, s2 = math.cos(self.dyaw), math.sin(self.dyaw)
                    self.dx = px + rx - (c2 * pos[0] - s2 * pos[1])
                    self.dy = py + ry - (s2 * pos[0] + c2 * pos[1])
                    self.corrections += 1
                    (px, py), cyaw = self.correct(pos, yaw)
                    c, s = math.cos(cyaw), math.sin(cyaw)
                    world = [(px + c * fx - s * fy, py + s * fx + c * fy) for fx, fy in pts]
        for wx, wy in world:
            k = (math.floor(wx / cs), math.floor(wy / cs))
            cell = self._pts.setdefault(k, [])
            if len(cell) < 8:
                cell.append((wx, wy, now))
        if len(self._pts) > MAX_CELLS:
            self._pts.pop(next(iter(self._pts)))
