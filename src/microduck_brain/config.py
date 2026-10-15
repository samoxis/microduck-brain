"""Brain configuration, loaded from TOML. Unknown keys are rejected, not silently ignored."""

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
    forward_speed: float = 0.24
    turn_speed: float = 1.4
    big_turn_speed: float = 1.4


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
    min_turn_degrees: float = 20.0
    max_turn_time: float = 6.0
    default_turn: str = "left"
    slow_distance: float = 0.70
    steer_yaw_max: float = 0.8
    steer_slew: float = 3.0
    low_obstacles: str = "avoid"
    drops: str = "avoid"
    corridor_half_width: float = 0.14
    sensor_lateral: float = 0.0224
    side_memory_seconds: float = 3.0


@dataclass
class Low:
    """Low objects (cubes, balls) seen in the bottom rows, where the floor is also visible."""

    nominal_height: float = 0.246
    nominal_pitch_deg: float = 0.0
    nominal_sensor_forward: float = 0.064
    tilt_tolerance_deg: float = 1.5
    tilt_tolerance_nopose_deg: float = 4.5
    range_margin: float = 0.06
    min_range: float = 0.10
    confirm_time: float = 0.12
    corridor_half_width: float = 0.15
    memory_seconds: float = 12.0
    extra_clearance: float = 0.15
    wall_margin: float = 0.25


@dataclass
class Drop:
    """Drops: ledges, stairs going down."""

    max_expected: float = 1.2
    range_margin: float = 0.10
    min_zones: int = 2
    confirm_time: float = 0.12
    clearance: float = 0.45
    corridor_half_width: float = 0.18
    memory_seconds: float = 120.0


@dataclass
class Corner:
    turns_before_big_turn: int = 3
    progress_time: float = 3.0
    big_turn_degrees: float = 160.0
    big_turn_max_time: float = 15.0
    fallback_turn_efficiency: float = 0.5


@dataclass
class Heading:
    enabled: bool = True
    kp: float = 2.0
    ki: float = 0.3
    max_correction: float = 0.5
    yaw_sign: float = 1.0


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
    low: Low = field(default_factory=Low)
    drop: Drop = field(default_factory=Drop)
    corner: Corner = field(default_factory=Corner)
    heading: Heading = field(default_factory=Heading)
    safety: Safety = field(default_factory=Safety)
    loop: Loop = field(default_factory=Loop)

    def validate(self) -> None:
        o = self.obstacle
        if not 0 <= o.first_row <= o.last_row <= 7:
            raise ValueError("obstacle: first_row/last_row must satisfy 0 <= first <= last <= 7")
        if not 0 <= o.front_first_col <= o.front_last_col <= 7:
            raise ValueError("obstacle: front_first_col/front_last_col must be within 0..7")
        if o.clear_distance < o.stop_distance:
            raise ValueError("obstacle: clear_distance must be >= stop_distance")
        if o.slow_distance < o.stop_distance:
            raise ValueError("obstacle: slow_distance must be >= stop_distance")
        if o.steer_yaw_max < 0 or o.steer_slew <= 0 or o.min_turn_degrees < 0:
            raise ValueError("obstacle: steer_yaw_max >= 0, steer_slew > 0, min_turn_degrees >= 0")
        if o.corridor_half_width <= 0 or o.side_memory_seconds < 0:
            raise ValueError("obstacle: corridor_half_width > 0, side_memory_seconds >= 0")
        if o.low_obstacles not in ("avoid", "ignore"):
            raise ValueError("obstacle: low_obstacles must be 'avoid' or 'ignore'")
        if o.drops not in ("avoid", "ignore"):
            raise ValueError("obstacle: drops must be 'avoid' or 'ignore'")
        dr = self.drop
        if dr.max_expected <= 0 or not 0 <= dr.range_margin < 1 or dr.min_zones < 1 or dr.confirm_time < 0:
            raise ValueError("drop: max_expected > 0, range_margin in [0, 1), min_zones >= 1, confirm_time >= 0")
        if dr.clearance < 0 or dr.corridor_half_width <= 0 or dr.memory_seconds < 0:
            raise ValueError("drop: clearance >= 0, corridor_half_width > 0, memory_seconds >= 0")
        lo = self.low
        if lo.nominal_height <= 0 or lo.tilt_tolerance_deg < 0 or lo.tilt_tolerance_nopose_deg < lo.tilt_tolerance_deg:
            raise ValueError("low: nominal_height > 0 and 0 <= tilt_tolerance_deg <= tilt_tolerance_nopose_deg")
        if not 0 <= lo.range_margin < 0.5 or lo.confirm_time < 0 or lo.corridor_half_width <= 0 or lo.memory_seconds < 0:
            raise ValueError("low: range_margin in [0, 0.5), confirm_time >= 0, corridor_half_width > 0, memory_seconds >= 0")
        if not 0 < self.corner.fallback_turn_efficiency <= 1 or self.corner.big_turn_max_time <= 0:
            raise ValueError("corner: fallback_turn_efficiency in (0, 1], big_turn_max_time > 0")
        if self.heading.yaw_sign not in (1, -1, 1.0, -1.0):
            raise ValueError("heading: yaw_sign must be 1 or -1")
        if self.heading.kp < 0 or self.heading.ki < 0 or self.heading.max_correction < 0:
            raise ValueError("heading: kp, ki, max_correction must be >= 0")
        if o.default_turn not in ("left", "right"):
            raise ValueError("obstacle: default_turn must be 'left' or 'right'")
        if self.motion.forward_speed < 0 or self.motion.turn_speed <= 0 or self.motion.big_turn_speed <= 0:
            raise ValueError("motion: speeds must be positive")
        if self.loop.rate_hz <= 0:
            raise ValueError("loop: rate_hz must be > 0")
        if self.corner.turns_before_big_turn < 1:
            raise ValueError("corner: turns_before_big_turn must be >= 1")


# Minimum commanded speeds, measured in the full simulator with the official walking policies
# (2026-10-03): below them the policy does not step at all and the robot stands still.
MIN_FORWARD_SPEED = 0.20   # m/s, commanded
MIN_TURN_SPEED = 1.1       # rad/s, commanded, turning in place


def speed_warnings(cfg: "Config") -> list[str]:
    """Warnings for configured speeds that are below the threshold at which the robot steps."""
    out = []
    m = cfg.motion
    if m.forward_speed < MIN_FORWARD_SPEED:
        out.append(
            f"motion.forward_speed = {m.forward_speed} m/s is below the walking threshold ({MIN_FORWARD_SPEED}): "
            "the robot will NOT step, it will stand still"
        )
    for key in ("turn_speed", "big_turn_speed"):
        v = getattr(m, key)
        if v < MIN_TURN_SPEED:
            out.append(
                f"motion.{key} = {v} rad/s is below the turning threshold ({MIN_TURN_SPEED}): "
                "the robot will NOT step while turning, it will stand still"
            )
    return out


def _fill(obj, data: dict, path: str) -> None:
    known = {f.name: f for f in fields(obj)}
    for key, value in data.items():
        if key not in known:
            raise ValueError(f"unknown configuration key: {path}{key}")
        current = getattr(obj, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"{path}{key} must be a section")
            _fill(current, value, f"{path}{key}.")
        else:
            if isinstance(current, bool):
                ok = isinstance(value, bool)
            elif isinstance(current, float):
                ok = isinstance(value, (int, float)) and not isinstance(value, bool)
                value = float(value) if ok else value
            else:
                ok = isinstance(value, type(current)) and not isinstance(value, bool)
            if not ok:
                raise ValueError(f"{path}{key}: expected {type(current).__name__}, got {value!r}")
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
