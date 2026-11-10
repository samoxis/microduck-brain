"""The wedge detector (``[obstacle] wedge_seconds``): grinding slowly against something while walking."""

from __future__ import annotations

import pytest

from microduck_brain.brain import Brain, Inputs
from microduck_brain.config import Config, from_dict

from .test_brain import clear

ON = {"obstacle": {"wedge_seconds": 2.5}, "safety": {"resume_delay": 0.2}}


def walk(cfg, speed, seconds=8.0):
    """Walk in a clear view while the odometry advances at ``speed`` m/s; the states it went through."""
    b = Brain(from_dict(cfg))
    t, x, states = 0.0, 0.0, []
    while t < seconds:
        cmd = b.tick(Inputs(t, clear(), True, "", 0.0, None, (x, 0.0)))
        states.append(b.state_name)
        if b.state_name == "forward" and cmd.kind == "move":
            x += speed * 0.05
        t += 0.05
    return b, states


def test_creeping_against_a_leg_stops_and_turns():
    """Full simulator: 5 cm in 3.6 s pressed against a desk leg, then a fall. Each 5 cm of creep reset
    stuck_seconds, so only this caught it (replayed: 4.9 s before the fall)."""
    b, states = walk({**ON, "obstacle": {"wedge_seconds": 2.5, "stuck_seconds": 4.0}}, 0.012)
    assert b.stats.wedged >= 1 and "brake" in states


def test_the_normal_slow_gait_is_not_a_wedge():
    """The simulated robot walks at ~35% of the command (median of 2.5 s windows; 1% under 11%)."""
    b, _ = walk(ON, 0.24 * 0.12, seconds=20.0)
    assert b.stats.wedged == 0


def test_off_by_default():
    assert Config().obstacle.wedge_seconds == 0.0
    b, _ = walk({"safety": {"resume_delay": 0.2}}, 0.0)
    assert b.stats.wedged == 0


def test_wedge_config_validation():
    for bad in ({"wedge_seconds": -1.0}, {"wedge_ratio": 0.0}, {"wedge_ratio": 1.0}):
        with pytest.raises(ValueError):
            from_dict({"obstacle": bad})
