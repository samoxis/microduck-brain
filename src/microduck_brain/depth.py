"""Cadrele de adâncime 8x8 -> ce vede robotul în față.

Convențiile vin din ``tof/src/lib.rs`` și din simulator (``sim/tof.py``): tablouri
row-major, rândul 0 sus, coloana 0 în STÂNGA robotului. ``status`` 5 sau 9 = distanță
valabilă (în mm, > 0), 255 = „nimic în rază" (spațiu liber), orice altceva = măsurătoare
eșuată (nu spune nimic despre ce e acolo).
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Obstacle

STATUS_VALID = (5, 9)
STATUS_NO_TARGET = 255


@dataclass(frozen=True)
class DepthFrame:
    """Un cadru brut, așa cum vine în ``tof.frame``."""

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
            raise ValueError("cadru de adâncime cu dimensiuni inconsistente")
        return cls(rows, cols, dist, status)

    def zone(self, row: int, col: int, max_range: float) -> float | None:
        """Distanța în metri; ``max_range`` dacă nu e nimic în rază; ``None`` dacă a eșuat."""
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
    """Ce hotărăște creierul din cadru."""

    blocked: bool          # obstacol în coridorul din față (sau senzor orb)
    blind: bool            # prea puține zone măsurate ca să ne încredem
    nearest_front: float | None
    front: float | None    # distanța „robustă" în față: a min_zones-a cea mai mică (taie zgomotul)
    left_free: float       # cât de liberă e partea stângă (m, mai mare = mai liber)
    right_free: float

    @property
    def freer_side(self) -> str | None:
        """'left', 'right' sau None dacă sunt la fel."""
        if abs(self.left_free - self.right_free) < 0.05:
            return None
        return "left" if self.left_free > self.right_free else "right"


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


def analyse(frame: DepthFrame, cfg: Obstacle, threshold: float | None = None) -> View:
    """Analizează un cadru. ``threshold`` = distanța sub care o zonă e „aproape"
    (implicit ``stop_distance``; la ieșirea din întoarcere se dă ``clear_distance``)."""
    if frame.rows != 8 or frame.cols != 8:
        raise ValueError(f"se așteaptă un cadru 8x8, am primit {frame.rows}x{frame.cols}")
    threshold = cfg.stop_distance if threshold is None else threshold

    rows = range(cfg.first_row, cfg.last_row + 1)
    front_cols = range(cfg.front_first_col, cfg.front_last_col + 1)

    ranges: list[float] = []
    for r in rows:
        for c in front_cols:
            d = frame.zone(r, c, cfg.max_range)
            if d is not None:
                ranges.append(d)
    ranges.sort()
    nearest = ranges[0] if ranges else None

    total = len(rows) * len(front_cols)
    blind = len(ranges) < cfg.blind_fraction * total
    k = max(1, cfg.min_zones)
    # a k-a cea mai mică distanță: „obstacol" înseamnă cel puțin k zone mai aproape decât pragul
    front = ranges[k - 1] if len(ranges) >= k else (cfg.max_range if ranges else None)
    blocked = blind or (front is not None and front < threshold)

    # Cât de liber e fiecare jumătate: media celor mai apropiate distanțe pe coloană.
    # Zonele eșuate nu contează; o jumătate fără nicio măsurătoare e 0 (nu o alegem).
    left = _column_minima(frame, cfg, range(0, 4))
    right = _column_minima(frame, cfg, range(4, 8))
    left_free = sum(left) / len(left) if left else 0.0
    right_free = sum(right) / len(right) if right else 0.0
    return View(blocked, blind, nearest, front, left_free, right_free)


# ── obiecte joase ────────────────────────────────────────────────────────────────
#
# În rândurile de jos senzorul vede și podeaua. Cu geometria cunoscută (înălțimea senzorului
# deasupra podelei și unde „în jos" e față de fiecare rază) se poate calcula ce distanță ar măsura
# pe podea goală; o citire vizibil mai scurtă e un obiect. Geometria vine din robot.state
# (frames.tof, safety.gravity, odom.position[2]) — aceeași ca în filtrul de podea al lui robotd
# (kinematics/src/tof.rs) — sau, dacă lipsește, din valorile nominale din [low].

import math

FOV_DEG = 45.0


def _beams() -> list[tuple[float, float, float]]:
    """Razele în cadrul senzorului (+x înainte, +y stânga, +z sus); rândul 0 sus, coloana 0 stânga.
    Aceleași ca în sim/tof.py și kinematics/src/tof.rs (centrele zonelor, FOV 45°)."""
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
    """Poza senzorului, așa cum o raportează robot.state."""

    sensor_pos: tuple[float, float, float]            # frames.tof.pos, în trunchi (m)
    sensor_quat: tuple[float, float, float, float]    # frames.tof.quat, [w,x,y,z], senzor -> trunchi
    gravity: tuple[float, float, float]               # safety.gravity (în jos; în picioare ≈ [0,0,-1])
    trunk_z: float | None                              # odom.position[2], înălțimea trunchiului

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
                return None   # poză negăsită / IMU neconvergent: mai bine nimic decât ceva aleator
            z = (params.get("odom") or {}).get("position")
            trunk_z = float(z[2]) if z and len(z) == 3 else None
            return cls(pos, quat, grav, trunk_z)  # type: ignore[arg-type]
        except (KeyError, TypeError, ValueError, IndexError):
            return None


def _rotate_inv(q: tuple[float, float, float, float], v: tuple[float, float, float]) -> tuple[float, float, float]:
    """Rotește v cu inversul cuaternionului q = [w,x,y,z] (trunchi -> senzor)."""
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, -x / n, -y / n, -z / n   # conjugat
    # v' = v + 2w(u × v) + 2 u × (u × v), u = (x,y,z)
    ux, uy, uz = x, y, z
    cx, cy, cz = uy * v[2] - uz * v[1], uz * v[0] - ux * v[2], ux * v[1] - uy * v[0]
    dx, dy, dz = uy * cz - uz * cy, uz * cx - ux * cz, ux * cy - uy * cx
    return (v[0] + 2 * (w * cx + dx), v[1] + 2 * (w * cy + dy), v[2] + 2 * (w * cz + dz))


@dataclass(frozen=True)
class FloorModel:
    """Unde ar fi podeaua pe fiecare zonă."""

    height: float                    # înălțimea senzorului deasupra podelei (m)
    down_sin: tuple[float, ...]      # sin(unghiul de coborâre față de orizontala LUMII), pe zonă (> 0 = privește în jos)
    horizontal: tuple[float, ...]    # |componenta orizontală| a razei (pentru distanța pe sol)

    def expected(self, zone: int, tilt_deg: float = 0.0) -> float:
        """Distanța înclinată așteptată pe podea goală, dacă senzorul ar fi înclinat în plus cu
        ``tilt_deg`` spre podea (cel mai rău caz: podeaua apare mai aproape). ``inf`` = raza nu atinge podeaua."""
        s = self.down_sin[zone]
        alpha = math.asin(max(-1.0, min(1.0, s))) + math.radians(tilt_deg)
        if alpha <= 1e-3:
            return math.inf
        return self.height / math.sin(min(alpha, math.pi / 2))


def floor_model(pose: Pose | None, low) -> FloorModel:
    if pose is not None:
        gl = math.sqrt(sum(v * v for v in pose.gravity))
        g = tuple(v / gl for v in pose.gravity)                    # unitar, în jos, în trunchi
        g_s = _rotate_inv(pose.sensor_quat, g)                      # în cadrul senzorului
        h = -sum(a * b for a, b in zip(pose.sensor_pos, g))         # componenta „sus" a poziției senzorului față de trunchi
        if pose.trunk_z is not None and pose.trunk_z > 0.05:
            h += pose.trunk_z
        else:
            h += low.nominal_height - 0.126                          # fără z de încredere: z nominal al trunchiului
        if not 0.05 < h < 0.6:
            h = low.nominal_height
    else:
        th = math.radians(low.nominal_pitch_deg)
        g_s = (math.sin(th), 0.0, -math.cos(th))                    # „în jos" în cadrul senzorului, pitch θ în jos
        h = low.nominal_height
    down_sin = tuple(sum(b * gg for b, gg in zip(beam, g_s)) for beam in BEAMS)
    horiz = tuple(math.sqrt(max(0.0, 1.0 - s * s)) for s in down_sin)
    return FloorModel(h, down_sin, horiz)


@dataclass(frozen=True)
class LowObstacle:
    row: int
    col: int
    slant: float        # distanța măsurată (m)
    forward: float      # înainte, în cadrul senzorului (m)
    lateral: float      # stânga +, în cadrul senzorului (m)
    ratio: float        # măsurat / așteptat pe podea (< 1 = mai aproape decât podeaua)


def low_obstacles(frame: DepthFrame, pose: Pose | None, cfg) -> list[LowObstacle]:
    """Zonele din rândurile de jos care măsoară vizibil mai aproape decât podeaua goală.

    ``cfg`` = Config. Toleranța la înclinare și marja de distanță acoperă eroarea de poză
    (sincronizare, legănat) și zgomotul senzorului; fără poză toleranța e mai mare."""
    if frame.rows != 8 or frame.cols != 8:
        return []
    lo = cfg.low
    model = floor_model(pose, lo)
    tilt = lo.tilt_tolerance_deg if pose is not None else lo.tilt_tolerance_nopose_deg
    out = []
    for r in range(cfg.obstacle.last_row + 1, 8):
        for c in range(8):
            i = r * 8 + c
            if frame.status[i] not in STATUS_VALID or frame.distance_mm[i] <= 0:
                continue
            d = frame.distance_mm[i] / 1000.0
            exp_near = model.expected(i, tilt)
            if d >= exp_near * (1.0 - lo.range_margin):
                continue                                # podea (sau mai departe)
            bx, by, _ = BEAMS[i]
            if d * bx < lo.min_range:
                continue                                # sub ~10 cm senzorul dă citiri false
            exp_nom = model.expected(i, 0.0)
            ratio = d / exp_nom if math.isfinite(exp_nom) else 0.0
            out.append(LowObstacle(r, c, d, d * bx, d * by, ratio))
    return out
