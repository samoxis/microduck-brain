"""Pericolele care nu sunt pereți: obiecte joase (cuburi, mingi) și goluri (margini, scări în jos).

Amândouă se văd doar în rândurile de jos ale cadrului și doar pe o fereastră scurtă: un cub de 7 cm
dispare sub cea mai de jos rază când robotul se apropie la ~45 cm, iar o gaură îngustă dispare când
razele încep să aterizeze pe podeaua de dincolo de ea. De aceea fiecare pericol confirmat e ținut
minte în coordonatele odometriei (``odom.position``, ``odom.yaw``) și folosit și după ce nu se mai vede.

``HazardMap.update`` se cheamă o dată pe tick; ``HazardMap.merge`` adaugă pericolele la vederea din
față (``View``), ca stările să nu știe de unde vine un obstacol.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .depth import DepthFrame, Pose, View, drop_zones, low_obstacles

LOW = "low"
DROP = "drop"

MAX_POINTS = 400        # memorie limitată: cele mai vechi puncte pleacă primele


@dataclass(frozen=True)
class Hazard:
    forward: float      # înainte, în cadrul senzorului (m)
    lateral: float      # stânga +, în cadrul senzorului (m)
    kind: str           # LOW | DROP


@dataclass
class _Kind:
    """Confirmarea în timp și memoria pentru un fel de pericol."""

    kind: str
    first: float | None = None
    points: list | None = None      # [x, y, t] în lumea odometriei

    def __post_init__(self) -> None:
        self.points = []


class HazardMap:
    def __init__(self, cfg, stats) -> None:
        self.cfg = cfg
        self.stats = stats
        self._kinds = {LOW: _Kind(LOW), DROP: _Kind(DROP)}
        self.items: list[Hazard] = []

    # --- configurație pe fel ----------------------------------------------------
    def _enabled(self, kind: str) -> bool:
        o = self.cfg.obstacle
        return (o.low_obstacles if kind == LOW else o.drops) == "avoid"

    def _confirm_time(self, kind: str) -> float:
        return self.cfg.low.confirm_time if kind == LOW else self.cfg.drop.confirm_time

    def _memory_seconds(self, kind: str) -> float:
        return self.cfg.low.memory_seconds if kind == LOW else self.cfg.drop.memory_seconds

    def _clearance(self, kind: str) -> float:
        return self.cfg.low.extra_clearance if kind == LOW else self.cfg.drop.clearance

    def _half_width(self, kind: str) -> float:
        return self.cfg.low.corridor_half_width if kind == LOW else self.cfg.drop.corridor_half_width

    # --- memorie ----------------------------------------------------------------
    def forget(self) -> None:
        """După o cădere: poziția odometriei nu mai e de încredere."""
        for k in self._kinds.values():
            k.first = None
            k.points.clear()
        self.items = []

    def remembered(self, kind: str) -> int:
        return len(self._kinds[kind].points)

    def _remember(self, k: _Kind, pos, yaw: float, rx: float, ry: float, now: float) -> bool:
        c, s = math.cos(yaw), math.sin(yaw)
        wx = pos[0] + c * rx - s * ry
        wy = pos[1] + s * rx + c * ry
        for p in k.points:
            if abs(p[0] - wx) < 0.07 and abs(p[1] - wy) < 0.07:
                p[0], p[1], p[2] = wx, wy, now
                return False
        k.points.append([wx, wy, now])
        if len(k.points) > MAX_POINTS:
            k.points.pop(0)
        return True

    # --- pe tick ----------------------------------------------------------------
    def update(self, frame: DepthFrame | None, pose: Pose | None, pos, yaw: float | None,
               safe: bool, now: float) -> list[Hazard]:
        self.items = []
        if not safe:
            self.forget()
            return self.items
        sx = pose.sensor_pos[0] if pose else self.cfg.low.nominal_sensor_forward
        can_remember = pos is not None and yaw is not None

        for kind, k in self._kinds.items():
            if not self._enabled(kind):
                k.first = None
                k.points.clear()
                continue
            if frame is None:
                found = []
            elif kind == LOW:
                found = [(o.forward, o.lateral) for o in low_obstacles(frame, pose, self.cfg)]
            else:
                found = [(d.forward, d.lateral) for d in drop_zones(frame, pose, self.cfg)]

            if found:
                if k.first is None:
                    k.first = now
                confirmed = now - k.first >= self._confirm_time(kind)
            else:
                k.first = None
                confirmed = False

            memory = self._memory_seconds(kind) if can_remember else 0.0
            if confirmed:
                new = 0
                for f, lat in found:
                    self.items.append(Hazard(f, lat, kind))
                    if memory > 0 and self._remember(k, pos, yaw, f + sx, lat, now):
                        new += 1
                if kind == LOW:
                    self.stats.low_seen += new
                elif new and not any(now - p[2] < 1.0 for p in k.points[:-new]):
                    self.stats.drops_seen += 1      # un gol nou, nu încă un punct din același

            if memory <= 0:
                k.points.clear()
                continue
            c, s = math.cos(yaw), math.sin(yaw)
            keep = []
            for p in k.points:
                if now - p[2] > memory:
                    continue
                dx, dy = p[0] - pos[0], p[1] - pos[1]
                rx, ry = c * dx + s * dy, -s * dx + c * dy       # cadrul trunchiului
                fwd = rx - sx
                # Un obiect jos depășit nu mai contează. Un gol rămâne cât ține memoria: robotul se poate
                # întoarce spre el, iar o gaură nu pleacă nicăieri.
                if kind == LOW and (fwd < -0.15 or fwd > 2.0):
                    continue
                keep.append(p)
                if confirmed and any(abs(f - fwd) < 0.05 and abs(lat - ry) < 0.05 for f, lat in found):
                    continue                                        # deja în lista curentă
                self.items.append(Hazard(fwd, ry, kind))
            k.points = keep
        return self.items

    # --- în vederea din față ------------------------------------------------------
    def merge(self, view: View, threshold: float) -> View:
        """Adaugă pericolele din coridorul robotului la vederea din față."""
        if not self.items:
            return view
        front, blocked = view.front, view.blocked
        left_free, right_free = view.left_free, view.right_free
        for h in self.items:
            if abs(h.lateral) > self._half_width(h.kind) or h.forward < -0.1:
                continue
            # tratat ca mai aproape: reacție fermă, fiindcă pericolul se vede doar pe o fereastră scurtă
            f = h.forward - self._clearance(h.kind)
            front = f if front is None else min(front, f)
            blocked = blocked or f < threshold
            if h.lateral >= 0:
                left_free = min(left_free, f)
            else:
                right_free = min(right_free, f)
        return replace(view, blocked=blocked, front=front, left_free=left_free, right_free=right_free)
