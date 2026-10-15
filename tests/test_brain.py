"""Tests for the decision logic, with synthetic depth frames.

Conventions: row 0 at the top, column 0 = the robot's left (as in the simulator and ``tofd``).
"""

from __future__ import annotations

import math

import pytest

from microduck_brain.brain import Brain, Inputs
from microduck_brain.config import Config, from_dict
from microduck_brain.depth import STATUS_NO_TARGET, DepthFrame, analyse, floor_model

FAR = 3.0


_FLOOR = floor_model(None, Config().low)


def frame(fn=None, fill=FAR) -> DepthFrame:
    """8x8 frame; ``fn(row, col)`` returns metres, ``None`` = failed measurement,
    ``"free"`` = nothing in range. "Far" (FAR) in the bottom rows means the floor: a standing robot
    always sees the floor there (rows 5-7: 1.68 / 1.01 / 0.73 m)."""
    dist, status = [], []
    for r in range(8):
        for c in range(8):
            v = fill if fn is None else fn(r, c)
            if r >= 5 and v == FAR:
                v = _FLOOR.expected(r * 8 + c)
            if v is None:
                dist.append(0), status.append(12)
            elif v == "free":
                dist.append(0), status.append(STATUS_NO_TARGET)
            else:
                dist.append(round(v * 1000)), status.append(5)
    return DepthFrame(8, 8, tuple(dist), tuple(status))


# --- cadrele sintetice cerute ---------------------------------------------
def clear():
    return frame()


def wall_ahead(d=0.20):
    return frame(lambda r, c: d if r <= 4 else FAR)


def obstacle_left(d=0.20):
    # obstacle on columns 0-3 (left); right side clear
    return frame(lambda r, c: d if c <= 3 else FAR)


def obstacle_right(d=0.20):
    return frame(lambda r, c: d if c >= 4 else FAR)


def corner():
    # wall ahead, both sides equally close
    return frame(lambda r, c: 0.20)


class Sim:
    """Runs the Brain on a fake clock, 20 Hz."""

    def __init__(self, cfg: Config | None = None, safe=True):
        self.cfg = cfg or Config()
        self.brain = Brain(self.cfg)
        self.t = 0.0
        self.safe = safe
        self.frame = clear()
        self.log = []

    def run(self, seconds: float, frame_=None, safe=None, dt=0.05):
        if frame_ is not None:
            self.frame = frame_
        if safe is not None:
            self.safe = safe
        out = None
        end = self.t + seconds
        while self.t < end - 1e-9:
            out = self.brain.tick(Inputs(self.t, self.frame, self.safe, "" if self.safe else "fallen"))
            self.log.append((self.t, self.brain.state_name, out))
            self.t += dt
        return out

    def up(self):
        """Start and wait until it leaves hold."""
        self.run(self.cfg.safety.resume_delay + 0.3)
        assert self.brain.state_name == "forward"


# --- analiza cadrului ------------------------------------------------------
def test_clear_is_not_blocked():
    v = analyse(clear(), Config().obstacle)
    assert not v.blocked and not v.blind


def test_wall_is_blocked():
    assert analyse(wall_ahead(), Config().obstacle).blocked


def test_single_noisy_zone_is_ignored():
    f = frame(lambda r, c: 0.1 if (r, c) == (2, 3) else FAR)
    assert not analyse(f, Config().obstacle).blocked


def test_floor_rows_are_ignored():
    f = frame(lambda r, c: 0.1 if r >= 5 else FAR)
    assert not analyse(f, Config().obstacle).blocked


def test_no_target_counts_as_free_space():
    assert not analyse(frame(lambda r, c: "free"), Config().obstacle).blocked


def test_all_failed_measurements_is_blind_and_blocked():
    v = analyse(frame(lambda r, c: None), Config().obstacle)
    assert v.blind and v.blocked


def test_failed_zones_do_not_count_as_obstacles():
    f = frame(lambda r, c: None if c % 2 else FAR)
    v = analyse(f, Config().obstacle)
    assert not v.blocked


def test_freer_side():
    assert analyse(obstacle_left(), Config().obstacle).freer_side == "right"
    assert analyse(obstacle_right(), Config().obstacle).freer_side == "left"
    assert analyse(corner(), Config().obstacle).freer_side is None


# --- comportament ------------------------------------------------------------
def test_starts_held_then_walks_forward_slowly():
    s = Sim()
    first = s.run(0.05)
    assert s.brain.state_name == "hold"
    assert first.kind in ("stop", "none")
    s.up()
    cmd = s.run(0.2)
    assert cmd.kind == "move"
    assert cmd.vx == pytest.approx(s.cfg.motion.forward_speed)
    assert cmd.vyaw == 0.0


def test_free_path_keeps_going():
    s = Sim()
    s.up()
    s.run(10.0, clear())
    assert s.brain.state_name == "forward"
    assert s.brain.stats.avoids == 0


def test_wall_stops_then_turns_then_continues():
    s = Sim()
    s.up()
    s.run(0.1, wall_ahead())
    assert s.brain.state_name in ("brake", "turn")
    cmd = s.run(0.1)
    assert cmd.kind == "move" and cmd.vx == 0.0  # standing still, not walking
    s.run(1.0)  # the stop pause ends, the turn starts
    assert s.brain.state_name == "turn"
    cmd = s.run(0.1)
    assert cmd.vx == 0.0 and abs(cmd.vyaw) == pytest.approx(s.cfg.motion.turn_speed)
    # peretele dispare => se reia mersul
    s.run(1.0, clear())
    assert s.brain.state_name == "forward"
    assert s.brain.stats.avoids == 1
    assert s.run(0.1).vx > 0


def test_never_moves_forward_while_blocked():
    s = Sim()
    s.up()
    s.log.clear()
    s.run(3.0, wall_ahead())
    assert all(c.vx == 0.0 for _, _, c in s.log if c.kind == "move")


def test_obstacle_on_left_turns_right():
    s = Sim()
    s.up()
    s.run(2.0, obstacle_left())
    assert s.brain.state_name == "turn"
    assert s.run(0.1).vyaw < 0  # vyaw negativ = dreapta


def test_obstacle_on_right_turns_left():
    s = Sim()
    s.up()
    s.run(2.0, obstacle_right())
    assert s.brain.state_name == "turn"
    assert s.run(0.1).vyaw > 0  # positive vyaw = left


def test_symmetric_wall_uses_default_turn():
    cfg = from_dict({"obstacle": {"default_turn": "right"}})
    s = Sim(cfg)
    s.up()
    s.run(2.0, corner())
    assert s.run(0.1).vyaw < 0


def test_turn_waits_for_clear_distance_not_just_stop_distance():
    s = Sim()
    s.up()
    s.run(2.0, wall_ahead(0.2))
    assert s.brain.state_name == "turn"
    # 0.33 m: above stop_distance (0.28) but below clear_distance (0.40) => keeps turning
    s.run(1.0, frame(lambda r, c: 0.33))
    assert s.brain.state_name == "turn"
    s.run(0.5, clear())
    assert s.brain.state_name == "forward"


def test_corner_triggers_big_turn_after_repeated_avoids():
    s = Sim()
    s.up()
    # always a wall ahead: never makes progress between avoidances
    # (after each turn it sees clear for a moment, then a wall again)
    for _ in range(s.cfg.corner.turns_before_big_turn):
        s.run(0.2, wall_ahead())            # blocked => stop
        s.run(1.0)                          # stop + start of the turn
        s.run(0.7, clear())                 # the turn finds a clear way, walks briefly
        s.run(0.1, wall_ahead())            # dar iar perete imediat
        if s.brain.state_name == "bigturn":
            break
    s.run(1.0, wall_ahead())
    assert s.brain.stats.big_turns == 1
    assert s.brain.state_name == "bigturn"
    expected = math.radians(s.cfg.corner.big_turn_degrees) / s.cfg.motion.big_turn_speed
    t0 = s.t
    s.run(expected + 0.3, wall_ahead())
    assert s.brain.state_name != "bigturn"
    assert s.t - t0 > expected
    # while it lasted, it never walked forward
    assert all(c.vx == 0.0 for _, n, c in s.log if n == "bigturn" and c.kind == "move")


def test_progress_resets_the_corner_counter():
    s = Sim()
    s.up()
    for _ in range(5):  # many more than turns_before_big_turn
        s.run(0.2, wall_ahead())
        s.run(1.0)
        s.run(0.6, clear())
        s.run(s.cfg.corner.progress_time + 0.5, clear())  # walked far enough => not a corner
    assert s.brain.stats.big_turns == 0
    assert s.brain.stats.avoids == 5


def test_turn_that_never_clears_becomes_big_turn():
    s = Sim()
    s.up()
    s.run(s.cfg.obstacle.max_turn_time + 3.0, corner())
    assert s.brain.stats.turn_timeouts >= 1
    assert s.brain.stats.big_turns >= 1


# --- safety -------------------------------------------------------------------
def test_fall_stops_and_sends_no_walk_commands_until_upright():
    s = Sim()
    s.up()
    s.log.clear()
    cmd = s.run(0.1, safe=False)
    assert s.brain.state_name == "hold"
    s.run(5.0)
    kinds = [c.kind for _, _, c in s.log]
    assert "move" not in kinds, "no motion command may be sent while fallen"
    assert kinds.count("stop") == 1, "stop once, not in a loop"
    assert s.brain.stats.holds_unsafe == 1


def test_resumes_only_after_resume_delay():
    s = Sim()
    s.up()
    s.run(1.0, safe=False)
    s.log.clear()
    s.run(s.cfg.safety.resume_delay - 0.3, safe=True)
    assert all(c.kind != "move" for _, _, c in s.log)
    s.run(0.6)
    assert s.brain.state_name == "forward"


def test_flapping_health_restarts_the_delay():
    s = Sim()
    s.up()
    s.run(0.5, safe=False)
    s.run(1.0, safe=True)
    s.run(0.1, safe=False)       # unsafe again before resume_delay
    s.run(1.0, safe=True)
    assert s.brain.state_name == "hold"


def test_fall_during_turn_also_holds():
    s = Sim()
    s.up()
    s.run(1.5, wall_ahead())
    assert s.brain.state_name == "turn"
    s.log.clear()
    s.run(2.0, safe=False)
    assert s.brain.state_name == "hold"
    assert all(c.kind != "move" for _, _, c in s.log)


def test_missing_depth_frames_stop_the_robot():
    s = Sim()
    s.up()
    s.log.clear()
    s.frame = None
    s.run(1.0)
    assert s.brain.state_name == "hold"
    assert all(c.kind != "move" for _, _, c in s.log)
    assert s.brain.stats.holds_no_depth == 1


# --- configuration --------------------------------------------------------------
def test_unknown_config_key_is_rejected():
    with pytest.raises(ValueError, match="unknown"):
        from_dict({"motion": {"forward_sped": 0.1}})


def test_bad_config_values_are_rejected():
    with pytest.raises(ValueError):
        from_dict({"obstacle": {"stop_distance": 0.5, "clear_distance": 0.3}})
    with pytest.raises(ValueError):
        from_dict({"motion": {"forward_speed": "repede"}})


def test_shipped_brain_toml_equals_defaults():
    import tomllib
    from pathlib import Path

    data = tomllib.loads((Path(__file__).parent.parent / "brain.toml").read_text())
    assert from_dict(data) == Config()


def test_speed_comes_from_config():
    s = Sim(from_dict({"motion": {"forward_speed": 0.2}}))
    s.up()
    assert s.run(0.2).vx == pytest.approx(0.2)


# --- the body's corridor and side memory (found by the kinematic MuJoCo runs) -----
def right_edge(d):
    """A post seen only in the rightmost column (7), d metres away."""
    return frame(lambda r, c: d if (c == 7 and r <= 4) else FAR)


def test_a_post_seen_only_in_the_rightmost_column_still_blocks():
    # col 7 at 0.26 m: 6.5 cm right of the axis, inside the body's corridor; the old column-based
    # "ahead" (columns 1-6) ignored it and the right leg brushed past such posts
    s = Sim()
    s.up()
    s.run(0.2, right_edge(0.26))
    assert s.brain.state_name in ("brake", "turn")


def test_a_post_well_to_the_side_does_not_block():
    s = Sim()
    s.up()
    s.run(1.0, frame(lambda r, c: 0.5 if (c == 7 and r <= 4) else FAR))   # 15 cm right at 0.5 m: outside
    assert s.brain.state_name == "forward" and s.brain.stats.brakes == 0


@pytest.mark.parametrize("side_memory,blocked", [(3.0, True), (0.0, False)])
def test_side_memory_keeps_a_post_that_left_the_field_of_view(side_memory, blocked):
    from microduck_brain.brain import Inputs

    b = Brain(from_dict({"obstacle": {"side_memory_seconds": side_memory}}))
    t = 0.0
    while t < 2.0:                                                # hold -> forward
        b.tick(Inputs(t, clear(), True, "", 0.0, None, (0.0, 0.0)))
        t += 0.05
    b.tick(Inputs(t, right_edge(0.42), True, "", 0.0, None, (0.0, 0.0)))   # seen: 0.40 ahead, 12 cm right
    t += 0.05
    gone = clear()                                                # now beside the body: out of view
    b.tick(Inputs(t, gone, True, "", 0.0, None, (0.22, 0.0)))
    view = b.look(Inputs(t, gone, True, "", 0.0, None, (0.22, 0.0)))
    assert view.blocked is blocked
