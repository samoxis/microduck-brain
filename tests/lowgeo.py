"""Generator de cadre ToF sintetice: ray-cast pe o podea plată + un cub/o minge, cu aceleași raze ca
simulatorul (sim/tof.py). Folosit în teste ca să nu depindă doar de fixture-urile înregistrate."""

from __future__ import annotations

import math

from microduck_brain.depth import BEAMS, STATUS_NO_TARGET, DepthFrame, Pose

NOMINAL_H = 0.246
SENSOR_X = 0.0643        # senzorul față de originea trunchiului, înainte
MAX_RANGE = 4.0


SENSOR_POS = (SENSOR_X, 0.0224, NOMINAL_H - 0.12)   # senzorul în trunchi (trunchiul stă la z = 0.12)


def _trunk_rotation(pitch_deg: float, roll_deg: float):
    """Trunchi -> lume: R = Rx(roll) Ry(pitch); pitch + = nasul în jos (ca în fixture-uri)."""
    p, r = math.radians(pitch_deg), math.radians(roll_deg)
    cr, sr, cp, sp = math.cos(r), math.sin(r), math.cos(p), math.sin(p)
    Rx = [[1, 0, 0], [0, cr, -sr], [0, sr, cr]]
    Ry = [[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]
    return [[sum(Rx[i][k] * Ry[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def pose_for(pitch_deg: float = 0.0, roll_deg: float = 0.0, trunk_z: float = 0.12) -> Pose:
    """Poza pe care ar raporta-o robotul cu trunchiul înclinat (capul drept față de trunchi)."""
    R = _trunk_rotation(pitch_deg, roll_deg)
    gravity = tuple(-R[2][i] for i in range(3))             # R^T (0,0,-1)
    return Pose(SENSOR_POS, (1.0, 0.0, 0.0, 0.0), gravity, trunk_z)


def render(obj: tuple | None = None, noise: float = 0.0, rng=None,
           pitch_deg: float = 0.0, roll_deg: float = 0.0, trunk_z: float = 0.12) -> DepthFrame:
    """Cadru ToF pentru un trunchi înclinat cu pitch/roll, pe o podea plată (z = 0).

    obj = ("cube", x, size[, lateral]) sau ("ball", x, size[, lateral]); x = fața din față a obiectului,
    măsurată de la ORIGINEA TRUNCHIULUI (ca în fixture-uri; senzorul e cu 6.4 cm mai în față)."""
    R = _trunk_rotation(pitch_deg, roll_deg)
    off = tuple(sum(R[i][k] * SENSOR_POS[k] for k in range(3)) for i in range(3))
    origin = (off[0], off[1], trunk_z + off[2])
    dist, status = [], []
    for b in BEAMS:
        d = tuple(sum(R[i][k] * b[k] for k in range(3)) for i in range(3))
        hit = math.inf
        if d[2] < -1e-9:
            hit = origin[2] / -d[2]
        if obj is not None:
            kind, ox, size, *rest = obj
            lat = rest[0] if rest else 0.0
            if kind == "cube":
                t = _ray_box(origin, d, (ox, 0.0224 + lat - size / 2, 0.0), (ox + size, 0.0224 + lat + size / 2, size))
            else:
                t = _ray_sphere(origin, d, (ox + size / 2, 0.0224 + lat, size / 2), size / 2)
            if t is not None:
                hit = min(hit, t)
        if hit > MAX_RANGE or not math.isfinite(hit):
            dist.append(0), status.append(STATUS_NO_TARGET)
            continue
        if noise and rng is not None:
            hit = max(0.0, hit + rng.gauss(0.0, 0.003 + noise * (hit / MAX_RANGE)))
        dist.append(int(hit * 1000)), status.append(5)
    return DepthFrame(8, 8, tuple(dist), tuple(status))


def _ray_box(o, d, lo, hi):
    tmin, tmax = 0.0, math.inf
    for i in range(3):
        if abs(d[i]) < 1e-12:
            if o[i] < lo[i] or o[i] > hi[i]:
                return None
            continue
        t1, t2 = (lo[i] - o[i]) / d[i], (hi[i] - o[i]) / d[i]
        if t1 > t2:
            t1, t2 = t2, t1
        tmin, tmax = max(tmin, t1), min(tmax, t2)
        if tmin > tmax:
            return None
    return tmin


def _ray_sphere(o, d, c, rad):
    oc = tuple(o[i] - c[i] for i in range(3))
    b = sum(oc[i] * d[i] for i in range(3))
    cc = sum(v * v for v in oc) - rad * rad
    disc = b * b - cc
    if disc < 0:
        return None
    t = -b - math.sqrt(disc)
    return t if t > 0 else None
