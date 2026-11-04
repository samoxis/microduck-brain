"""Brain configuration, loaded from TOML. Unknown keys are rejected, not silently ignored."""

from __future__ import annotations

import math

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
    stuck_seconds: float = 0.0
    wedge_seconds: float = 0.0
    wedge_ratio: float = 0.08


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
class Explore:
    """Prefer directions not visited yet (see explore.py). Off by default until measured in the full simulator."""

    enabled: bool = False
    cell: float = 0.25
    turn_margin: float = 1.5
    steer_margin: float = 0.0
    bend_every: float = 4.0
    bend_min_clear: float = 0.7
    max_bend_deg: float = 60.0
    novelty_margin: float = 0.3


@dataclass
class Head:
    """Scan with the head (the ToF sensor is on it) while walking: see head.py. Off by default."""

    scan: bool = False
    amplitude_deg: float = 25.0
    period: float = 2.5
    side_free_seconds: float = 3.0
    pitch: bool = False
    pitch_down_deg: float = 20.0
    pitch_after_turn: float = 1.5
    pitch_while_walking: bool = True


@dataclass
class Home:
    """Go back to where it started (see home.py). Off by default."""

    enabled: bool = False
    after_seconds: float = 240.0
    arrive_radius: float = 0.35
    lookahead_cells: int = 3
    face_threshold_deg: float = 60.0
    replan_every: float = 1.0
    yield_seconds: float = 4.0


@dataclass
class Localize:
    """Correct odometry drift against the walls seen before (see localize.py). Off by default."""

    enabled: bool = False
    every: int = 3
    min_points: int = 12
    min_gain_m: float = 0.02
    min_gain_rel: float = 0.5
    min_age: float = 20.0
    search_step: float = 0.04
    search_turn_deg: float = 1.5
    max_turn_rate: float = 0.05
    method: str = "grid"
    icp_reach: float = 0.15
    icp_gain: float = 0.3


@dataclass
class Recover:
    """Help a robot that stays down (see recover.py). Off by default."""

    enabled: bool = False
    after_seconds: float = 6.0
    init_seconds: float = 2.5
    verify_seconds: float = 8.0
    max_attempts: int = 3


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
    explore: Explore = field(default_factory=Explore)
    head: Head = field(default_factory=Head)
    recover: Recover = field(default_factory=Recover)
    home: Home = field(default_factory=Home)
    localize: Localize = field(default_factory=Localize)
    safety: Safety = field(default_factory=Safety)
    loop: Loop = field(default_factory=Loop)

    def validate(self) -> None:
        # TOML accepts nan and inf, and every "<" check below is False for nan: reject them first.
        def finite(obj, path):
            for f in fields(obj):
                v = getattr(obj, f.name)
                if is_dataclass(v):
                    finite(v, f"{path}{f.name}.")
                elif isinstance(v, float) and not math.isfinite(v):
                    raise ValueError(f"{path}{f.name} must be a finite number, got {v!r}")
        finite(self, "")
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
        if o.stuck_seconds < 0:
            raise ValueError("obstacle: stuck_seconds >= 0")
        if o.wedge_seconds < 0 or not 0 < o.wedge_ratio < 1:
            raise ValueError("obstacle: wedge_seconds >= 0, 0 < wedge_ratio < 1")
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
        e = self.explore
        if e.cell <= 0 or e.turn_margin < 0 or e.steer_margin < 0 or e.bend_every <= 0 or e.bend_min_clear < 0:
            raise ValueError("explore: cell > 0, turn_margin, steer_margin >= 0, bend_every > 0, bend_min_clear >= 0")
        if not 0 <= e.max_bend_deg <= 90 or not 0 <= e.novelty_margin <= 1:
            raise ValueError("explore: max_bend_deg in [0, 90], novelty_margin in [0, 1]")
        hd = self.head
        if not 0 <= hd.pitch_down_deg <= 40 or hd.pitch_after_turn < 0:
            raise ValueError("head: pitch_down_deg in [0, 40], pitch_after_turn >= 0")
        if not 0 <= hd.amplitude_deg <= 60 or hd.period <= 0 or hd.side_free_seconds < 0:
            raise ValueError("head: amplitude_deg in [0, 60], period > 0, side_free_seconds >= 0")
        hm = self.home
        if hm.after_seconds < 0 or hm.arrive_radius <= 0 or hm.lookahead_cells < 1 or hm.replan_every <= 0 \
                or hm.yield_seconds < 0:
            raise ValueError("home: after_seconds >= 0, arrive_radius > 0, lookahead_cells >= 1, replan_every > 0, yield_seconds >= 0")
        if not 10 <= hm.face_threshold_deg <= 180:
            raise ValueError("home: face_threshold_deg in [10, 180]")
        lz = self.localize
        if lz.every < 1 or lz.min_points < 1 or lz.min_gain_m <= 0 or lz.min_age < 0 or lz.search_step <= 0 \
                or lz.search_turn_deg <= 0:
            raise ValueError("localize: every, min_points >= 1; min_gain_m > 0; min_age >= 0; search_step, search_turn_deg > 0")
        if lz.method not in ("grid", "icp"):
            raise ValueError("localize: method must be 'grid' or 'icp'")
        rc = self.recover
        if rc.after_seconds < 0 or rc.init_seconds <= 0 or rc.verify_seconds <= 0 or rc.max_attempts < 1:
            raise ValueError("recover: after_seconds >= 0, init_seconds > 0, verify_seconds > 0, max_attempts >= 1")
        if not 5 <= self.loop.rate_hz <= 200:
            # robotd stops the robot ~0.5 s after the last robot.move: under 5 Hz it would stutter
            raise ValueError("loop: rate_hz must be in [5, 200]")
        sf = self.safety
        if not 0 < sf.tof_timeout <= 2.0 or not 0 < sf.state_timeout <= 2.0:
            # beyond ~2 s it would keep walking on a frame or a pose that no longer describes the room
            raise ValueError("safety: tof_timeout and state_timeout must be in (0, 2] s")
        if sf.health_poll <= 0 or sf.health_poll > 5.0 or sf.resume_delay < 0:
            raise ValueError("safety: health_poll in (0, 5] s, resume_delay >= 0")
        if not 0 < o.blind_fraction <= 1 or o.min_zones < 1 or o.max_range <= 0:
            raise ValueError("obstacle: blind_fraction in (0, 1], min_zones >= 1, max_range > 0")
        if o.min_turn_time < 0 or o.max_turn_time <= 0 or o.max_turn_time < o.min_turn_time or o.stop_pause < 0:
            raise ValueError("obstacle: 0 <= min_turn_time <= max_turn_time, max_turn_time > 0, stop_pause >= 0")
        if o.stop_distance <= 0:
            raise ValueError("obstacle: stop_distance > 0")
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
