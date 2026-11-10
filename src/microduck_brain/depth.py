"""8x8 depth frames -> what the robot sees ahead.

Conventions come from ``tof/src/lib.rs`` and from the simulator (``sim/tof.py``): row-major
arrays, row 0 at the top, column 0 on the robot's LEFT. ``status`` 5 or 9 = valid range (mm, > 0),
255 = "nothing in range" (free space), anything else = failed measurement (says nothing about
what is out there).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import Obstacle

STATUS_VALID = (5, 9)
STATUS_NO_TARGET = 255

FOV_DEG = 45.0
# Height of the ToF sensor above the trunk origin when standing (MJCF `tof` site, STAND keyframe).
SENSOR_ABOVE_TRUNK = 0.126


@dataclass(frozen=True)
class DepthFrame:
    """One raw frame, as it arrives in ``tof.frame``."""

    rows: int
    cols: int
    distance_mm: tuple[int, ...]
    status: tuple[int, ...]

    @classmethod
    def from_params(cls, params: dict) -> "DepthFrame":
        rows, cols = int(params["rows"]), int(params["cols"])
        dist = tuple(int(v) for v in params["distance_mm"])
        status = tuple(int(v) for v in params["status"])
        if len(dist) != rows * cols or len(status) != rows * cols:
            raise ValueError("depth frame with inconsistent dimensions")
        return cls(rows, cols, dist, status)

    def zone(self, row: int, col: int, max_range: float) -> float | None:
        """Range in metres; ``max_range`` if nothing is in range; ``None`` if the measurement failed."""
        i = row * self.cols + col
        status = self.status[i]
        if status in STATUS_VALID:
            d = self.distance_mm[i]
            return d / 1000.0 if d > 0 else None
        if status == STATUS_NO_TARGET:
            return max_range
        return None


@dataclass(frozen=True)
class View:
    """What the brain decides from a frame."""

    blocked: bool          # obstacle in the corridor ahead (or the sensor is blind)
    blind: bool            # too few zones measured to trust the frame
    nearest_front: float | None
    front: float | None    # robust distance ahead: the min_zones-th smallest (rejects single-zone noise)
    left_free: float       # how free the left half is (m, larger = freer)
    right_free: float

    @property
    def freer_side(self) -> str | None:
        """'left', 'right', or None if they are about equal."""
        if abs(self.left_free - self.right_free) < 0.05:
            return None
        return "left" if self.left_free > self.right_free else "right"


def head_yaw(pose: "Pose | None") -> float:
    """How far the sensor looks to the left of the trunk's axis (rad), from frames.tof's quaternion:
    the yaw of the sensor's forward axis in the trunk frame. 0 without a pose."""
    if pose is None:
        return 0.0
    w, x, y, z = pose.sensor_quat
    n = w * w + x * x + y * y + z * z
    return math.atan2(2 * (x * y + w * z) / n, 1 - 2 * (y * y + z * z) / n)


def rotate(f: float, l: float, yaw: float) -> tuple[float, float]:
    """A point (forward, left) in the sensor's horizontal frame, in the trunk's. ``yaw`` = 0 returns
    it unchanged, bit for bit (cos 0 = 1, sin 0 = 0), which keeps old recordings replaying exactly."""
    if yaw == 0.0:
        return f, l
    c, s = math.cos(yaw), math.sin(yaw)
    return c * f - s * l, s * f + c * l


def column_azimuth(col: int) -> float:
    """Azimuth of a column's centre in the sensor frame (rad, + = left); column 0 is the left."""
    half = math.radians(FOV_DEG / 2.0 - FOV_DEG / 8 / 2.0)
    return half - col * (2.0 * half / 7.0)


def _column_minima(frame: DepthFrame, cfg: Obstacle, cols: range) -> list[float]:
    out = []
    for c in cols:
        vals = [
            d
            for r in range(cfg.first_row, cfg.last_row + 1)
            if (d := frame.zone(r, c, cfg.max_range)) is not None
        ]
        if vals:
            out.append(min(vals))
    return out


def analyse(frame: DepthFrame, cfg: Obstacle, threshold: float | None = None, yaw: float = 0.0,
            geom: "Geometry | None" = None) -> View:
    """Analyse the wall rows of a frame. ``threshold`` = distance below which a zone is "near"
    (``stop_distance`` by default; ``clear_distance`` when deciding a turn is finished). ``yaw`` =
    where the head looks (rad, + = left, see ``head_yaw``): the returns are rotated into the trunk's
    frame before deciding what is "ahead" and which side a column is on.

    With ``geom`` (the head pitched, see ``Geometry``), every return is placed with the sensor's full
    pose instead, and a wall-row return that lands on the floor is the floor, not a wall."""
    if frame.rows != 8 or frame.cols != 8:
        raise ValueError(f"expected an 8x8 frame, got {frame.rows}x{frame.cols}")
    if geom is not None:
        return _analyse_tilted(frame, cfg, cfg.stop_distance if threshold is None else threshold, geom)
    threshold = cfg.stop_distance if threshold is None else threshold

    rows = range(cfg.first_row, cfg.last_row + 1)
    front_cols = range(cfg.front_first_col, cfg.front_last_col + 1)

    # Blindness is judged on the central columns: are there enough measurements to trust the frame?
    measured = sum(1 for r in rows for c in front_cols if frame.zone(r, c, cfg.max_range) is not None)
    blind = measured < cfg.blind_fraction * len(rows) * len(front_cols)

    # "Ahead" is geometric, not a set of columns: every wall-row return whose lateral position, relative
    # to the robot's axis, falls within the body's swept corridor counts, at its forward distance. The
    # sensor sits ``sensor_lateral`` to the left of the axis, so the right side needs the outer columns.
    ranges: list[float] = []
    for r in rows:
        for c in range(frame.cols):
            d = frame.zone(r, c, cfg.max_range)
            if d is None:
                continue
            bx, by, _ = BEAMS[r * frame.cols + c]
            fx, fy = rotate(d * bx, d * by, yaw)
            if abs(fy + cfg.sensor_lateral) <= cfg.corridor_half_width:
                ranges.append(fx)
    ranges.sort()
    nearest = ranges[0] if ranges else None
    k = max(1, cfg.min_zones)
    # The k-th smallest range: "obstacle" means at least k zones closer than the threshold.
    front = ranges[k - 1] if len(ranges) >= k else (cfg.max_range if ranges else None)
    blocked = blind or (front is not None and front < threshold)

    # How free each half is: the mean of the per-column nearest ranges. Failed zones don't count;
    # a half with no measurement at all scores 0 (never chosen).
    if yaw == 0.0:
        left = _column_minima(frame, cfg, range(0, 4))
        right = _column_minima(frame, cfg, range(4, 8))
    else:   # a column belongs to the side it points to, in the trunk's frame
        left_cols = [c for c in range(8) if column_azimuth(c) + yaw > 0]
        right_cols = [c for c in range(8) if column_azimuth(c) + yaw <= 0]
        left = _column_minima(frame, cfg, left_cols)
        right = _column_minima(frame, cfg, right_cols)
    left_free = sum(left) / len(left) if left else 0.0
    right_free = sum(right) / len(right) if right else 0.0
    return View(blocked, blind, nearest, front, left_free, right_free)


def _analyse_tilted(frame: DepthFrame, cfg: Obstacle, threshold: float, geom: "Geometry") -> View:
    rows = range(cfg.first_row, cfg.last_row + 1)
    front_cols = range(cfg.front_first_col, cfg.front_last_col + 1)
    measured = sum(1 for r in rows for c in front_cols if frame.zone(r, c, cfg.max_range) is not None)
    blind = measured < cfg.blind_fraction * len(rows) * len(front_cols)
    ranges: list[float] = []
    col_min: dict[int, float] = {}
    for r in rows:
        for c in range(frame.cols):
            d = frame.zone(r, c, cfg.max_range)
            if d is None:
                continue
            i = r * frame.cols + c
            if geom.not_an_obstacle(i, d):
                d = cfg.max_range                  # the floor (or a drop): free as far as walls go
            col_min[c] = min(col_min.get(c, math.inf), d)
            if d >= cfg.max_range:
                continue
            vx, vy, _ = geom.beams[i]
            if abs(d * vy + cfg.sensor_lateral) <= cfg.corridor_half_width:
                ranges.append(d * vx)
    ranges.sort()
    nearest = ranges[0] if ranges else None
    k = max(1, cfg.min_zones)
    front = ranges[k - 1] if len(ranges) >= k else (cfg.max_range if col_min else None)
    blocked = blind or (front is not None and front < threshold)
    left = [v for c, v in col_min.items() if geom.column_side(c) > 0]
    right = [v for c, v in col_min.items() if geom.column_side(c) <= 0]
    return View(blocked, blind, nearest, front,
                sum(left) / len(left) if left else 0.0, sum(right) / len(right) if right else 0.0)


# ── Floor geometry ──────────────────────────────────────────────────────────────────
#
# In the bottom rows the sensor also sees the floor. With the geometry known (the sensor's height
# above the floor and where "down" is relative to each beam) we can compute the range each zone
# would read on an empty floor: a clearly shorter reading is an object, a clearly longer one (or
# none at all) is a drop. The geometry comes from robot.state (frames.tof, safety.gravity,
# odom.position[2]) — the same inputs robotd's own floor filter uses (kinematics/src/tof.rs) —
# or, when missing, from the nominal values in [low].


def _beams() -> list[tuple[float, float, float]]:
    """Beam directions in the sensor frame (+x forward, +y left, +z up); row 0 top, column 0 left.
    The same zone centres as sim/tof.py and kinematics/src/tof.rs (45° FOV)."""
    half = math.radians(FOV_DEG / 2.0 - FOV_DEG / 8 / 2.0)
    step = 2.0 * half / 7.0
    out = []
    for r in range(8):
        e = half - r * step
        for c in range(8):
            a = half - c * step
            out.append((math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)))
    return out


BEAMS = _beams()


@dataclass(frozen=True)
class Pose:
    """The sensor's pose, as robot.state reports it."""

    sensor_pos: tuple[float, float, float]            # frames.tof.pos, in the trunk frame (m)
    sensor_quat: tuple[float, float, float, float]    # frames.tof.quat, [w,x,y,z], sensor -> trunk
    gravity: tuple[float, float, float]               # safety.gravity (pointing down; standing ≈ [0,0,-1])
    trunk_z: float | None                              # odom.position[2], trunk height above the floor

    @classmethod
    def from_state(cls, params: dict) -> "Pose | None":
        try:
            tof = params["frames"]["tof"]
            pos = tuple(float(v) for v in tof["pos"])
            quat = tuple(float(v) for v in tof["quat"])
            grav = tuple(float(v) for v in params["safety"]["gravity"])
            if len(pos) != 3 or len(quat) != 4 or len(grav) != 3:
                return None
            if math.sqrt(sum(v * v for v in quat)) < 0.5 or math.sqrt(sum(v * v for v in grav)) < 0.5:
                return None   # no pose yet / IMU not converged: nothing is better than something random
            z = (params.get("odom") or {}).get("position")
            trunk_z = float(z[2]) if z and len(z) == 3 else None
            return cls(pos, quat, grav, trunk_z)  # type: ignore[arg-type]
        except (KeyError, TypeError, ValueError, IndexError):
            return None


def _rotate_inv(q: tuple[float, float, float, float], v: tuple[float, float, float]) -> tuple[float, float, float]:
    """Rotate v by the inverse of quaternion q = [w,x,y,z] (trunk -> sensor)."""
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, -x / n, -y / n, -z / n   # conjugate
    # v' = v + 2w(u × v) + 2 u × (u × v), u = (x,y,z)
    ux, uy, uz = x, y, z
    cx, cy, cz = uy * v[2] - uz * v[1], uz * v[0] - ux * v[2], ux * v[1] - uy * v[0]
    dx, dy, dz = uy * cz - uz * cy, uz * cx - ux * cz, ux * cy - uy * cx
    return (v[0] + 2 * (w * cx + dx), v[1] + 2 * (w * cy + dy), v[2] + 2 * (w * cz + dz))


@dataclass(frozen=True)
class FloorModel:
    """Where the floor would be, per zone."""

    height: float                    # sensor height above the floor (m)
    down_sin: tuple[float, ...]      # sin(beam angle below the WORLD horizontal), per zone (> 0 = looks down)
    horizontal: tuple[float, ...]    # |horizontal component| of each beam

    def expected(self, zone: int, tilt_deg: float = 0.0) -> float:
        """Slant range expected on an empty floor if the sensor were tilted a further ``tilt_deg``
        towards the floor (positive: the floor appears closer). ``inf`` = the beam never meets the floor."""
        s = self.down_sin[zone]
        alpha = math.asin(max(-1.0, min(1.0, s))) + math.radians(tilt_deg)
        if alpha <= 1e-3:
            return math.inf
        return self.height / math.sin(min(alpha, math.pi / 2))


def floor_model(pose: Pose | None, low) -> FloorModel:
    if pose is not None:
        gl = math.sqrt(sum(v * v for v in pose.gravity))
        g = tuple(v / gl for v in pose.gravity)                    # unit, pointing down, trunk frame
        g_s = _rotate_inv(pose.sensor_quat, g)                      # the same, in the sensor frame
        h = -sum(a * b for a, b in zip(pose.sensor_pos, g))         # "up" component of the sensor's offset
        if pose.trunk_z is not None and pose.trunk_z > 0.05:
            h += pose.trunk_z
        else:
            h += low.nominal_height - SENSOR_ABOVE_TRUNK             # no trustworthy z: nominal trunk height
        if not 0.05 < h < 0.6:
            h = low.nominal_height
    else:
        th = math.radians(low.nominal_pitch_deg)
        g_s = (math.sin(th), 0.0, -math.cos(th))                    # "down" in the sensor frame, pitched θ down
        h = low.nominal_height
    down_sin = tuple(sum(b * gg for b, gg in zip(beam, g_s)) for beam in BEAMS)
    horiz = tuple(math.sqrt(max(0.0, 1.0 - s * s)) for s in down_sin)
    return FloorModel(h, down_sin, horiz)


def _rotate(q: tuple[float, float, float, float], v: tuple[float, float, float]) -> tuple[float, float, float]:
    """Rotate v by quaternion q = [w,x,y,z] (sensor -> trunk)."""
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return _rotate_inv((w / n, -x / n, -y / n, -z / n), v)


class Geometry:
    """The sensor's full pose, for a head that is not level (``[head] pitch``): every beam in the
    trunk's frame, and the floor each zone would see. A return within the floor tolerance of the
    floor is the floor; one beyond it is a drop; neither is an obstacle, whichever row it is in."""

    def __init__(self, pose: Pose, cfg) -> None:
        self.beams = [_rotate(pose.sensor_quat, b) for b in BEAMS]
        self.floor = floor_model(pose, cfg.low)
        self.tilt = cfg.low.tilt_tolerance_deg
        self.margin = cfg.low.range_margin
        mid = 3 * 8                      # a column's side: where its middle row points, in the trunk
        self._side = [self.beams[mid + c][1] for c in range(8)]
        fwd = _rotate(pose.sensor_quat, (1.0, 0.0, 0.0))
        # Looking down more than 5 deg relative to the trunk: only then is the near-and-low gate on.
        self.pitched = math.atan2(-fwd[2], math.hypot(fwd[0], fwd[1])) > math.radians(5.0)

    def not_an_obstacle(self, i: int, d: float) -> bool:
        exp = self.floor.expected(i, self.tilt)
        return math.isfinite(exp) and d >= exp * (1.0 - self.margin)

    def column_side(self, c: int) -> float:
        return self._side[c]


def geometry(pose: Pose | None, cfg) -> Geometry | None:
    """A ``Geometry`` when the head may pitch (``[head] pitch``) and there is a pose; else None, and
    every caller keeps its level-head arithmetic, bit for bit."""
    if pose is None or not cfg.head.pitch:
        return None
    return Geometry(pose, cfg)


@dataclass(frozen=True)
class LowObstacle:
    row: int
    col: int
    slant: float        # measured range (m)
    forward: float      # ahead, in the sensor frame (m)
    lateral: float      # left +, in the sensor frame (m)
    ratio: float        # measured / expected floor range (< 1 = closer than the floor)


def low_obstacles(frame: DepthFrame, pose: Pose | None, cfg, geom: Geometry | None = None) -> list[LowObstacle]:
    """Bottom-row zones that read clearly shorter than an empty floor, under a clear wall row
    (otherwise it is the foot of a wall, not a low object).

    ``cfg`` is the whole Config. The tilt tolerance and range margin absorb pose error (timing,
    gait sway) and sensor noise; without a pose the tolerance is wider.

    With ``geom`` (the head pitched) every row that sees the floor is used, "the foot of a wall" is
    judged on the real obstacles above the zone in its column, and positions use the full pose."""
    if frame.rows != 8 or frame.cols != 8:
        return []
    if geom is not None:
        return _low_tilted(frame, cfg, geom)
    lo = cfg.low
    o = cfg.obstacle
    model = floor_model(pose, lo)
    tilt = lo.tilt_tolerance_deg if pose is not None else lo.tilt_tolerance_nopose_deg
    # Nearest return in the wall rows of each column: a short bottom-row reading under a wall that
    # close is just the foot of that wall, already handled by the wall rows — not a low object.
    upper = []
    for c in range(8):
        vals = [d for r in range(o.first_row, o.last_row + 1) if (d := frame.zone(r, c, o.max_range)) is not None]
        upper.append(min(vals) if vals else math.inf)
    out = []
    for r in range(o.last_row + 1, 8):
        for c in range(8):
            i = r * 8 + c
            if frame.status[i] not in STATUS_VALID or frame.distance_mm[i] <= 0:
                continue
            d = frame.distance_mm[i] / 1000.0
            exp_near = model.expected(i, tilt)
            if d >= exp_near * (1.0 - lo.range_margin):
                continue                                # floor (or further)
            if upper[c] < d + lo.wall_margin:
                continue                                # the foot of a wall the upper rows already see
            bx, by, _ = BEAMS[i]
            if d * bx < lo.min_range:
                continue                                # under ~10 cm the sensor reports phantom returns
            exp_nom = model.expected(i, 0.0)
            ratio = d / exp_nom if math.isfinite(exp_nom) else 0.0
            out.append(LowObstacle(r, c, d, d * bx, d * by, ratio))
    return out


# ── Drops (ledges, stairs going down) ───────────────────────────────────────────────────
#
# Where the floor should be clearly visible, a zone reading "nothing in range" (status 255) or a
# range clearly longer than the floor means the floor is missing there: a ledge, a staircase. In the
# simulator a beam entering the apartment's stairwell passes under the floor slab and returns 255;
# a real sensor would see the step, i.e. a longer range. Both count.


LOW_MAX_HEIGHT = 0.15    # pitched head: a return higher than this above the floor is a wall, not a low object
LOW_KEEP_HEIGHT = 0.10   # ... and only one this low, and within LOW_REACH ahead, is reported as a low object
LOW_REACH = 0.5          # the gap under the level beams (finding 3); further out the level head sees them
LOW_MIN_HEIGHT = 0.04    # pitched: a point lower than this is the floor read a little short (trunk lean)


def _low_tilted(frame: DepthFrame, cfg, geom: Geometry) -> list[LowObstacle]:
    """Pitched head: a low object is a return short of the floor whose point is within
    LOW_MAX_HEIGHT of the floor, with no taller return straight above it (at about the same
    horizontal distance: then it is the foot of a wall). Pitched, even the top row can meet the
    floor several metres out, so "which row" says nothing; the point's height does."""
    lo, o = cfg.low, cfg.obstacle
    out = []
    for c in range(8):
        above = math.inf                     # horizontal distance of the nearest tall return above, this column
        for r in range(8):
            i = r * 8 + c
            d = frame.zone(r, c, o.max_range)
            if d is None or d >= o.max_range:
                continue
            if geom.not_an_obstacle(i, d):
                continue                     # floor (or further)
            vx, vy, _ = geom.beams[i]
            horiz = d * math.hypot(vx, vy)
            height = geom.floor.height - d * geom.floor.down_sin[i]
            if height > LOW_MAX_HEIGHT:
                above = min(above, horiz)    # a wall or furniture: tall
                continue
            if above < horiz + lo.wall_margin or d * vx < lo.min_range:
                continue                     # the foot of a wall / phantom returns close in
            if geom.pitched and (d * vx > LOW_REACH or height > LOW_KEEP_HEIGHT or height < LOW_MIN_HEIGHT):
                # Far out, a few degrees of sway move the floor a lot: in the full simulator 85% of
                # pitched "low objects" were 0.5-4 m away, ~14 000 a run on furniture and wall feet.
                # And the false ones close in were mostly within 3 cm of the floor (the floor read a
                # little short with the trunk leaning 1-4 deg); cube and ball tops are 7-8 cm.
                continue
            exp_nom = geom.floor.expected(i, 0.0)
            out.append(LowObstacle(r, c, d, d * vx, d * vy, d / exp_nom if math.isfinite(exp_nom) else 0.0))
    return out


@dataclass(frozen=True)
class DropZone:
    row: int
    col: int
    forward: float      # where the floor should have been: ahead, in the sensor frame (m)
    lateral: float      # left +, in the sensor frame (m)
    no_target: bool     # True = 255 (nothing), False = a range longer than the floor


def drop_zones(frame: DepthFrame, pose: Pose | None, cfg, geom: Geometry | None = None) -> list[DropZone]:
    """Bottom-row zones where the floor is missing. Reported only if at least ``drop.min_zones``
    such zones are in the frame (a single missed zone is noise). With ``geom`` every row that sees
    the floor close enough is used, with positions from the full pose."""
    if frame.rows != 8 or frame.cols != 8:
        return []
    dr = cfg.drop
    model = geom.floor if geom is not None else floor_model(pose, cfg.low)
    tilt = cfg.low.tilt_tolerance_deg if pose is not None else cfg.low.tilt_tolerance_nopose_deg
    out = []
    for r in (range(8) if geom is not None else range(cfg.obstacle.last_row + 1, 8)):
        for c in range(8):
            i = r * 8 + c
            nominal = model.expected(i, 0.0)
            if not math.isfinite(nominal) or nominal > dr.max_expected:
                continue                     # floor too far away to be seen reliably
            far = model.expected(i, -tilt)   # the furthest the floor can plausibly be, head tilted up
            if not math.isfinite(far):
                continue
            status, d = frame.status[i], frame.distance_mm[i] / 1000.0
            if status == STATUS_NO_TARGET:
                no_target = True
            elif status in STATUS_VALID and d > far * (1.0 + dr.range_margin):
                no_target = False
            else:
                continue                     # floor (or closer: an object), or a failed measurement
            bx, by, _ = geom.beams[i] if geom is not None else BEAMS[i]
            out.append(DropZone(r, c, nominal * bx, nominal * by, no_target))
    return out if len(out) >= dr.min_zones else []
