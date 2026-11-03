"""Going home (home.py): the planner, the trigger, arrival, and a closed loop in a 2-D room."""

from __future__ import annotations

import math

from microduck_brain.brain import Brain, Inputs
from microduck_brain.config import from_dict
from microduck_brain.home import plan

from . import world2d
from .test_brain import clear


def test_plan_goes_only_through_walked_cells_and_never_cuts_a_corner():
    # An L of walked cells: (0,0)..(4,0) then (4,1)..(4,4).
    visited = {(x, 0) for x in range(5)} | {(4, y) for y in range(5)}
    path = plan(visited, set(), (0, 0), (4, 4))
    assert path[0] == (0, 0) and path[-1] == (4, 4)
    assert all(c in visited for c in path)
    for a, b in zip(path, path[1:]):
        if a[0] != b[0] and a[1] != b[1]:                      # a diagonal step: both sides walked
            assert (b[0], a[1]) in visited and (a[0], b[1]) in visited


def test_plan_avoids_walls_and_says_when_there_is_no_way():
    visited = {(x, y) for x in range(5) for y in range(3)}
    blocked = {(2, 0), (2, 1)}
    path = plan(visited, blocked, (0, 0), (4, 0))
    assert path and not set(path) & blocked and (2, 2) in path
    assert plan(visited, blocked | {(2, 2)}, (0, 0), (4, 0)) is None


def test_homing_starts_after_after_seconds_of_walking_and_arrives():
    cfg = from_dict({"home": {"enabled": True, "after_seconds": 2.0}, "safety": {"resume_delay": 0.2}})
    b = Brain(cfg)
    t, x = 0.0, 0.0
    states = []
    while t < 40:
        cmd = b.tick(Inputs(t, clear(), True, "", 0.0 if b.stats.home_started == 0 else math.pi, None, (x, 0.0)))
        states.append(b.state_name)
        if cmd.kind == "move" and cmd.vx > 0:
            x += (0.05 if b.stats.home_started == 0 else -0.05)  # out along +x, back along -x
        t += 0.05
        if b.state_name == "home":
            break
    assert b.stats.home_started == 1 and b.stats.home_arrived == 1 and b.state_name == "home"
    assert abs(x) <= cfg.home.arrive_radius + 0.05


def test_a_fall_abandons_the_way_home():
    from microduck_brain.recover import FALLEN

    cfg = from_dict({"home": {"enabled": True, "after_seconds": 1.0}, "safety": {"resume_delay": 0.2}})
    b = Brain(cfg)
    for i in range(60):
        b.tick(Inputs(i * 0.05, clear(), True, "", 0.0, None, (i * 0.02, 0.0)))
    assert b.homing.active
    b.tick(Inputs(3.05, clear(), False, FALLEN, 0.0, None, (1.2, 0.0)))
    assert not b.homing.active and b.stats.home_aborted == 1


def test_goes_home_in_the_living_room_without_touching_anything():
    cfg = from_dict({"home": {"enabled": True, "after_seconds": 120.0}, "explore": {"enabled": True}})
    w = world2d.worlds()[3]
    r = world2d.run(w, cfg, 360, 0)
    assert r.stats.home_started == 1 and r.stats.home_arrived == 1 and r.contacts == 0


def test_plan_keeps_clear_of_walls_when_it_can():
    visited = {(x, y) for x in range(6) for y in range(3)}
    blocked = {(3, 0)}                                          # a chair leg in the bottom row
    path = plan(visited, blocked, (0, 0), (5, 0))
    near = {(3 + dx, 0 + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
    assert not set(path[1:-1]) & near                           # goes round, one cell clear
    # No clear way: it still finds the tight one.
    tight = {(x, 0) for x in range(6)} | {(x, 1) for x in range(6)}
    assert plan(tight, {(3, 1)}, (0, 0), (5, 0)) is not None


def test_stuck_detector_stops_and_turns_away_from_what_is_beside():
    from microduck_brain.hazards import SIDE, Hazard

    cfg = from_dict({"obstacle": {"stuck_seconds": 2.0}, "safety": {"resume_delay": 0.2}})
    b = Brain(cfg)
    t = 0.0
    while t < 1.0:
        b.tick(Inputs(t, clear(), True, "", 0.0, None, (0.0, 0.0)))
        t += 0.05
    assert b.state_name == "forward"
    orig = b.hazards.update

    def update(*args, **kwargs):                                # something beside the right leg
        orig(*args, **kwargs)
        b.hazards.items = [Hazard(0.1, -0.2, SIDE)]   # outside the corridor: not "ahead"
        return b.hazards.items

    b.hazards.update = update
    states = []
    while t < 4.0:                                              # walking, but the odometry is frozen
        b.tick(Inputs(t, clear(), True, "", 0.0, None, (0.0, 0.0)))
        states.append(b.state_name)
        t += 0.05
    assert b.stats.stuck == 1 and "bigturn" in states and b.turn_sign == 1.0   # away: to the left


def test_stuck_detector_is_off_by_default():
    b = Brain(from_dict({"safety": {"resume_delay": 0.2}}))
    for i in range(200):
        b.tick(Inputs(i * 0.05, clear(), True, "", 0.0, None, (0.0, 0.0)))
    assert b.stats.stuck == 0 and b.state_name == "forward"


def test_homing_in_the_dining_room_from_a_wedged_start():
    """The case that once alternated for ever between avoiding a table leg and turning back to it:
    seed 28 of the 2-D dining room (65 contacts before the fixes)."""
    cfg = from_dict({"home": {"enabled": True, "after_seconds": 240.0}, "obstacle": {"stuck_seconds": 4.0}})
    r = world2d.run(world2d.chair_legs(), cfg, 600, 28)
    assert r.stats.home_arrived == 1 and r.contacts <= 1
