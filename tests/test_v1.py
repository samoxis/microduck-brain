"""v1 tests: steering while walking, turns by measured angle, heading hold, speed thresholds.

``PlantSim`` is a simple robot: real turn rate = ``eff`` x the commanded one (+ ``bias`` rad/s while
walking forward), and it reports its yaw as ``odom.yaw`` (wrapping at ±pi, like a real angle).
"""

from __future__ import annotations

import logging
import math

import pytest

from microduck_brain import config as config_mod
from microduck_brain.brain import Brain, Inputs
from microduck_brain.cli import log_config_warnings
from microduck_brain.config import Config, from_dict

from .test_brain import FAR, clear, frame, wall_ahead


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class PlantSim:
    def __init__(self, cfg=None, eff=0.5, bias=0.0, yaw0=0.0, report_yaw=True):
        self.cfg = cfg or Config()
        self.brain = Brain(self.cfg)
        self.t = 0.0
        self.true_yaw = yaw0           # not wrapped
        self.eff, self.bias, self.report_yaw = eff, bias, report_yaw
        self.frame = clear()
        self.safe = True
        self.log: list[tuple[float, str, object, float]] = []

    def run(self, seconds, frame_=None, dt=0.05, safe=None):
        if frame_ is not None:
            self.frame = frame_
        if safe is not None:
            self.safe = safe
        end = self.t + seconds
        cmd = None
        while self.t < end - 1e-9:
            yaw = wrap(self.true_yaw) if self.report_yaw else None
            cmd = self.brain.tick(Inputs(self.t, self.frame, self.safe, "" if self.safe else "fallen", yaw))
            self.log.append((self.t, self.brain.state_name, cmd, self.true_yaw))
            if cmd.kind == "move":
                self.true_yaw += (self.eff * cmd.vyaw + (self.bias if cmd.vx > 0 else 0.0)) * dt
            self.t += dt
        return cmd

    def up(self):
        self.run(self.cfg.safety.resume_delay + 0.3)
        assert self.brain.state_name == "forward"


def obstacle_at(d, side):
    """Obstacle at distance d, on the 'left' or 'right' half of the frame (the rest clear)."""
    cols = range(0, 4) if side == "left" else range(4, 8)
    return frame(lambda r, c: d if (c in cols and r <= 4) else FAR)


# --- 1. viraj lin ------------------------------------------------------------
def test_obstacle_approaching_makes_it_steer_while_still_walking():
    s = PlantSim()
    s.up()
    cmd = s.run(1.0, obstacle_at(0.5, "left"))
    assert s.brain.state_name == "forward"       # does not stop
    assert cmd.vx == pytest.approx(s.cfg.motion.forward_speed)   # keeps walking
    assert cmd.vyaw < 0                            # to the right (the free side)
    assert s.brain.stats.brakes == 0 and s.brain.stats.avoids == 1


def test_steering_grows_as_the_obstacle_gets_closer():
    yaws = []
    for d in (0.65, 0.5, 0.35):
        s = PlantSim()
        s.up()
        yaws.append(abs(s.run(3.0, obstacle_at(d, "right")).vyaw))
    assert yaws[0] < yaws[1] < yaws[2]
    assert yaws[2] <= Config().obstacle.steer_yaw_max + 1e-9
    assert yaws[0] > 0


def test_steers_toward_the_free_side_both_ways():
    s = PlantSim(); s.up()
    assert s.run(2.0, obstacle_at(0.4, "right")).vyaw > 0   # obstacle on the right -> left
    s = PlantSim(); s.up()
    assert s.run(2.0, obstacle_at(0.4, "left")).vyaw < 0


def test_steering_is_smooth_no_jump_in_yaw_command():
    s = PlantSim()
    s.up()
    s.log.clear()
    s.run(0.3, clear())
    s.run(4.0, obstacle_at(0.35, "left"))
    yaws = [c.vyaw for _, _, c, _ in s.log if c.kind == "move"]
    steps = [abs(b - a) for a, b in zip(yaws, yaws[1:])]
    assert max(steps) <= s.cfg.obstacle.steer_slew * 0.05 + 1e-9
    # and never stopped walking
    assert all(c.vx > 0 for _, _, c, _ in s.log if c.kind == "move")


def test_stops_completely_only_when_there_is_no_room():
    s = PlantSim()
    s.up()
    s.log.clear()
    s.run(0.2, wall_ahead(0.2))              # sub stop_distance
    assert s.brain.state_name in ("brake", "turn")
    s.run(0.3)
    moves = [c for _, n, c, _ in s.log if n in ("brake", "turn") and c.kind == "move"]
    assert moves and all(c.vx == 0.0 for c in moves)
    assert s.brain.stats.brakes == 1


def test_steering_ends_and_walking_straight_resumes_when_clear():
    s = PlantSim()
    s.up()
    s.run(2.0, obstacle_at(0.4, "left"))
    cmd = s.run(6.0, clear())
    assert s.brain.state_name == "forward"
    assert abs(cmd.vyaw) < 0.05
    assert s.brain.stats.avoids == 1 and s.brain.stats.brakes == 0


def test_turn_then_continue_is_one_avoid_not_two():
    s = PlantSim()
    s.up()
    s.run(0.3, wall_ahead(0.2))
    s.run(2.0)                                  # stop + turn
    s.run(0.3, frame(lambda r, c: 0.5))         # liber peste clear_distance dar sub slow_distance
    assert s.brain.state_name == "forward"
    s.run(0.5)
    assert s.brain.stats.avoids == 1


# --- 2. turns by measured angle ---------------------------------------------------
def enter_bigturn(s):
    """Drive the robot into bigturn and return (true yaw, time) at entry; afterwards the way is clear."""
    s.cfg.corner.turns_before_big_turn = 1
    s.up()
    s.frame = wall_ahead(0.2)
    for _ in range(400):
        y, t = s.true_yaw, s.t
        s.run(0.05)
        if s.brain.state_name == "bigturn":
            s.frame = clear()      # after the turn the way is clear, so it doesn't stop again
            return y, t
    raise AssertionError("never reached bigturn")


def finish_bigturn(s, limit=40.0):
    t0 = s.t
    while s.brain.state_name == "bigturn" and s.t - t0 < limit:
        s.run(0.05)


@pytest.mark.parametrize("eff", [1.0, 0.5, 0.3])
def test_big_turn_reaches_the_measured_angle_whatever_the_real_rate(eff):
    s = PlantSim(eff=eff)
    y0, t0 = enter_bigturn(s)
    finish_bigturn(s)
    turned = abs(s.true_yaw - y0)
    target = math.radians(s.cfg.corner.big_turn_degrees)
    assert target <= turned <= target + 0.25, f"rotit {math.degrees(turned):.0f}°"
    # with low efficiency it takes proportionally longer
    expected = target / (s.cfg.motion.big_turn_speed * eff)
    assert s.t - t0 == pytest.approx(expected, abs=0.4)


def test_big_turn_works_across_the_pi_wraparound():
    s = PlantSim(eff=0.6, yaw0=math.radians(170))
    y0, _ = enter_bigturn(s)
    finish_bigturn(s)
    assert abs(s.true_yaw - y0) == pytest.approx(math.radians(160), abs=0.25)


def test_big_turn_gives_up_if_the_angle_never_grows():
    s = PlantSim(eff=0.0)                        # frozen odometry
    _, t0 = enter_bigturn(s)
    finish_bigturn(s)
    assert s.brain.stats.turn_timeouts == 1
    assert s.t - t0 == pytest.approx(s.cfg.corner.big_turn_max_time, abs=0.2)
    assert s.brain.state_name == "forward"


def test_big_turn_without_odometry_falls_back_to_timed():
    s = PlantSim(report_yaw=False)
    _, t0 = enter_bigturn(s)
    finish_bigturn(s)
    c = s.cfg
    expected = math.radians(c.corner.big_turn_degrees) / (c.motion.big_turn_speed * c.corner.fallback_turn_efficiency)
    assert s.t - t0 == pytest.approx(expected, abs=0.2)


def test_big_turn_degrees_comes_from_config():
    cfg = from_dict({"corner": {"big_turn_degrees": 90.0}})
    s = PlantSim(cfg, eff=1.0)
    y0, _ = enter_bigturn(s)
    finish_bigturn(s)
    assert abs(s.true_yaw - y0) == pytest.approx(math.radians(90), abs=0.25)


def test_turn_does_not_end_before_min_measured_angle_even_if_clear():
    s = PlantSim(eff=0.2)                        # turns very little per second
    s.up()
    s.run(0.5, wall_ahead(0.2))
    assert s.brain.state_name in ("brake", "turn")
    s.run(0.5, clear())                          # clear, but has not turned 20° yet
    assert s.brain.state_name == "turn"
    s.run(10.0)
    assert s.brain.state_name == "forward"


def test_yaw_sign_minus_one_measures_a_robot_whose_odom_grows_to_the_right():
    cfg = from_dict({"heading": {"yaw_sign": -1.0}})
    brain = Brain(cfg)
    brain.cfg.corner.turns_before_big_turn = 1
    reported, t = 0.0, 0.0
    def tick(fr):
        nonlocal reported, t
        cmd = brain.tick(Inputs(t, fr, True, "", reported))
        if cmd.kind == "move":
            reported -= 0.5 * cmd.vyaw * 0.05      # vyaw>0 (left) => odom.yaw DECREASES
        t += 0.05
    for _ in range(50):
        tick(clear())
    while brain.state_name != "bigturn" and t < 20:
        tick(wall_ahead(0.2))
    start = reported
    while brain.state_name == "bigturn" and t < 60:
        tick(clear())
    assert brain.stats.turn_timeouts == 0
    assert abs(reported - start) == pytest.approx(math.radians(160), abs=0.25)


# --- 4. heading hold -----------------------------------------------------------------
def drift_error(enabled, bias=0.10, eff=0.5, seconds=40.0):
    cfg = from_dict({"heading": {"enabled": enabled}})
    s = PlantSim(cfg, eff=eff, bias=bias)
    s.up()
    y_start = s.true_yaw
    s.run(seconds, clear())
    return abs(s.true_yaw - y_start)


def test_heading_hold_cancels_constant_drift():
    off = drift_error(False)
    on = drift_error(True)
    assert off > 3.0                 # without correction it drifts a lot (0.1 rad/s x 40 s)
    assert on < 0.12                 # with correction it stays within ~7°


def test_heading_hold_converges_for_other_rates_and_signs():
    for bias in (0.12, -0.12, 0.05):
        for eff in (0.4, 0.9):
            assert drift_error(True, bias=bias, eff=eff) < 0.15, (bias, eff)


def test_heading_correction_is_bounded():
    cfg = from_dict({"heading": {"max_correction": 0.2}})
    s = PlantSim(cfg, eff=0.5, bias=0.3)       # the drift is larger than it can correct
    s.up()
    s.log.clear()
    s.run(15.0, clear())
    assert max(abs(c.vyaw) for _, _, c, _ in s.log if c.kind == "move") <= 0.2 + 1e-9


def test_straight_walk_has_zero_yaw_command_without_drift():
    s = PlantSim(eff=0.5, bias=0.0)
    s.up()
    s.log.clear()
    s.run(5.0, clear())
    assert all(abs(c.vyaw) < 1e-9 for _, _, c, _ in s.log if c.kind == "move")


def test_new_heading_is_adopted_after_an_avoidance():
    s = PlantSim(eff=0.8)
    s.up()
    s.run(0.3, clear())
    start = s.true_yaw
    s.run(4.0, obstacle_at(0.35, "left"))      # a virat spre dreapta
    s.run(15.0, clear())
    assert s.true_yaw < start - 0.1             # did not go back to the old heading
    after = s.true_yaw
    s.run(10.0)
    assert abs(s.true_yaw - after) < 0.1        # and holds the new one


def test_heading_hold_off_without_odometry():
    s = PlantSim(report_yaw=False, bias=0.1)
    s.up()
    cmd = s.run(3.0, clear())
    assert cmd.vyaw == 0.0


def test_heading_integral_is_reset_after_a_fall():
    s = PlantSim(eff=0.5, bias=0.1)
    s.up()
    s.run(10.0, clear())
    assert s.brain.heading_integral != 0.0
    s.run(1.0, safe=False)
    assert s.brain.heading_integral == 0.0


# --- 3. speed thresholds ---------------------------------------------------------------
def test_default_speeds_are_above_the_walking_thresholds():
    assert config_mod.speed_warnings(Config()) == []


@pytest.mark.parametrize(
    "section,key,value",
    [("motion", "forward_speed", 0.10), ("motion", "turn_speed", 0.6), ("motion", "big_turn_speed", 1.0)],
)
def test_speeds_below_threshold_warn_naming_the_key(section, key, value):
    cfg = from_dict({section: {key: value}})
    w = config_mod.speed_warnings(cfg)
    assert len(w) == 1 and f"{section}.{key}" in w[0] and "will NOT step" in w[0]


def test_speeds_exactly_at_threshold_do_not_warn():
    cfg = from_dict({"motion": {"forward_speed": config_mod.MIN_FORWARD_SPEED,
                                "turn_speed": config_mod.MIN_TURN_SPEED,
                                "big_turn_speed": config_mod.MIN_TURN_SPEED}})
    assert config_mod.speed_warnings(cfg) == []


def test_startup_logs_a_clear_warning(caplog):
    cfg = from_dict({"motion": {"forward_speed": 0.1, "turn_speed": 0.6}})
    with caplog.at_level(logging.WARNING):
        out = log_config_warnings(cfg)
    assert len(out) == 2
    text = caplog.text
    assert "WARNING" in text and "forward_speed" in text and "turn_speed" in text
    assert "brain.toml" in text


def test_no_warning_logged_for_good_config(caplog):
    with caplog.at_level(logging.WARNING):
        assert log_config_warnings(Config()) == []
    assert caplog.text == ""


def test_new_config_validation():
    with pytest.raises(ValueError):
        from_dict({"obstacle": {"slow_distance": 0.1}})
    with pytest.raises(ValueError):
        from_dict({"heading": {"yaw_sign": 0.5}})
    with pytest.raises(ValueError):
        from_dict({"corner": {"fallback_turn_efficiency": 0.0}})
