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

    measured = 0
    near = 0
    nearest: float | None = None
    for r in rows:
        for c in front_cols:
            d = frame.zone(r, c, cfg.max_range)
            if d is None:
                continue
            measured += 1
            nearest = d if nearest is None else min(nearest, d)
            if d < threshold:
                near += 1

    total = len(rows) * len(front_cols)
    blind = measured < cfg.blind_fraction * total
    blocked = blind or near >= cfg.min_zones

    # Cât de liber e fiecare jumătate: media celor mai apropiate distanțe pe coloană.
    # Zonele eșuate nu contează; o jumătate fără nicio măsurătoare e 0 (nu o alegem).
    left = _column_minima(frame, cfg, range(0, 4))
    right = _column_minima(frame, cfg, range(4, 8))
    left_free = sum(left) / len(left) if left else 0.0
    right_free = sum(right) / len(right) if right else 0.0
    return View(blocked, blind, nearest, left_free, right_free)
