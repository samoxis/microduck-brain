"""Configurația creierului, încărcată din TOML. Chei necunoscute sunt respinse."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path


@dataclass
class Sockets:
    robot: str = ""
    tof: str = ""


@dataclass
class Motion:
    forward_speed: float = 0.10
    turn_speed: float = 0.6
    big_turn_speed: float = 0.8


@dataclass
class Obstacle:
    stop_distance: float = 0.28
    clear_distance: float = 0.40
    first_row: int = 0
    last_row: int = 4
    front_first_col: int = 1
    front_last_col: int = 6
    min_zones: int = 2
    blind_fraction: float = 0.5
    max_range: float = 4.0
    stop_pause: float = 0.3
    min_turn_time: float = 0.5
    max_turn_time: float = 6.0
    default_turn: str = "left"


@dataclass
class Corner:
    turns_before_big_turn: int = 3
    progress_time: float = 3.0
    big_turn_degrees: float = 160.0


@dataclass
class Safety:
    tof_timeout: float = 0.5
    state_timeout: float = 1.0
    health_poll: float = 1.0
    resume_delay: float = 1.5


@dataclass
class Loop:
    rate_hz: float = 20.0
    log_every: float = 5.0


@dataclass
class Config:
    sockets: Sockets = field(default_factory=Sockets)
    motion: Motion = field(default_factory=Motion)
    obstacle: Obstacle = field(default_factory=Obstacle)
    corner: Corner = field(default_factory=Corner)
    safety: Safety = field(default_factory=Safety)
    loop: Loop = field(default_factory=Loop)

    def validate(self) -> None:
        o = self.obstacle
        if not 0 <= o.first_row <= o.last_row <= 7:
            raise ValueError("obstacle: first_row/last_row trebuie să fie 0 <= first <= last <= 7")
        if not 0 <= o.front_first_col <= o.front_last_col <= 7:
            raise ValueError("obstacle: front_first_col/front_last_col trebuie să fie în 0..7")
        if o.clear_distance < o.stop_distance:
            raise ValueError("obstacle: clear_distance trebuie să fie >= stop_distance")
        if o.default_turn not in ("left", "right"):
            raise ValueError("obstacle: default_turn trebuie să fie 'left' sau 'right'")
        if self.motion.forward_speed < 0 or self.motion.turn_speed <= 0 or self.motion.big_turn_speed <= 0:
            raise ValueError("motion: vitezele trebuie să fie pozitive")
        if self.loop.rate_hz <= 0:
            raise ValueError("loop: rate_hz trebuie să fie > 0")
        if self.corner.turns_before_big_turn < 1:
            raise ValueError("corner: turns_before_big_turn trebuie să fie >= 1")


def _fill(obj, data: dict, path: str) -> None:
    known = {f.name: f for f in fields(obj)}
    for key, value in data.items():
        if key not in known:
            raise ValueError(f"cheie necunoscută în configurație: {path}{key}")
        current = getattr(obj, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"{path}{key} trebuie să fie o secțiune")
            _fill(current, value, f"{path}{key}.")
        else:
            if isinstance(current, bool) or not isinstance(value, type(current)):
                # un întreg e acceptat acolo unde se așteaptă un float
                if not (isinstance(current, float) and isinstance(value, int) and not isinstance(value, bool)):
                    raise ValueError(
                        f"{path}{key}: se aștepta {type(current).__name__}, am primit {value!r}"
                    )
                value = float(value)
            setattr(obj, key, value)


def from_dict(data: dict) -> Config:
    cfg = Config()
    _fill(cfg, data, "")
    cfg.validate()
    return cfg


def load(path: str | Path | None) -> Config:
    if path is None:
        return from_dict({})
    with open(path, "rb") as fh:
        return from_dict(tomllib.load(fh))
