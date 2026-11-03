"""Exploration (explore.py) and observe mode."""

from __future__ import annotations

import json
import math

import pytest

from microduck_brain.brain import Brain, Inputs
from microduck_brain.config import Config, from_dict
from microduck_brain.depth import View
from microduck_brain.explore import Explorer
from microduck_brain.hazards import SIDE, Hazard

from . import world2d
from .test_runner import robot, stop, wait_for  # noqa: F401  (fixture)

ON = {"explore": {"enabled": True}}


def explorer_with_trail(points):
    e = Explorer(from_dict(ON))
    for i, p in enumerate(points):
        e.update(p, float(i))
    return e


# --- the map ---------------------------------------------------------------------------
def test_unvisited_direction_is_more_novel_than_the_visited_one():
    # Walked along +x from the origin to 2 m: looking along the trail, every cell is visited.
    trail = [(x / 10, 0.0) for x in range(0, 21)]
    e = explorer_with_trail(trail)
    here = (0.0, 0.0)
    assert e.novelty(here, 0.0, 0.0) == 0.0                 # along the trail
    assert e.novelty(here, 0.0, math.pi / 2) == 1.0         # never been there


def wall_frame(d=0.6):
    """A frame whose wall rows all see a wall at ``d`` m."""
    from .test_brain import frame
    return frame(lambda r, c: d if r <= 4 else 4.0)


def test_cells_behind_a_wall_are_not_novel():
    from microduck_brain.config import Obstacle

    e = Explorer(from_dict(ON))
    assert e.novelty((0.0, 0.0), 0.0, 0.0) == 1.0          # nothing known: all novel
    e.update_walls(wall_frame(0.6), (0.0, 0.0), 0.0, 0.064, Obstacle())
    assert e.blocked_cells > 0
    # The wall is ~0.66 m ahead: only the 0.5 m probe is in front of it.
    assert e.novelty((0.0, 0.0), 0.0, 0.0) <= 1 / 7 + 1e-9
    assert e.novelty((0.0, 0.0), 0.0, math.pi) == 1.0        # behind the robot: still open


def test_a_wall_frame_is_mapped_once():
    from microduck_brain.config import Obstacle

    e = Explorer(from_dict(ON))
    f = wall_frame(0.6)
    e.update_walls(f, (0.0, 0.0), 0.0, 0.064, Obstacle())
    n = e.blocked_cells
    e.update_walls(f, (5.0, 5.0), 1.0, 0.064, Obstacle())     # same frame object: ignored
    assert e.blocked_cells == n
    e.forget()
    assert e.blocked_cells == 0


def test_wall_beside_and_on_course():
    e = Explorer(from_dict(ON))
    e._blocked[e._cell(0.1, -0.3)] = 0.0                      # beside the right leg
    assert e.wall_beside((0.0, 0.0), 0.0, -1.0)
    assert not e.wall_beside((0.0, 0.0), 0.0, 1.0)
    e2 = Explorer(from_dict(ON))
    e2._blocked[e2._cell(0.0, -0.5)] = 0.0                    # 0.5 m to the right
    assert e2.wall_on_course((0.0, 0.0), -math.pi / 2, 0.14)  # turning right walks into it
    assert not e2.wall_on_course((0.0, 0.0), 0.0, 0.14)       # straight ahead is clear


def test_side_novelty_prefers_the_side_not_walked():
    # A trail to the left of the robot (y > 0), none to the right.
    trail = [(x / 10, y / 10) for x in range(-20, 21) for y in range(3, 20, 2)]
    e = explorer_with_trail(trail)
    left, right = e.side_novelty((0.0, 0.0), 0.0)
    assert right > left


def test_memory_is_bounded(monkeypatch):
    import microduck_brain.explore as ex
    monkeypatch.setattr(ex, "MAX_CELLS", 50)
    e = explorer_with_trail([(i * 0.3, 0.0) for i in range(200)])
    assert e.cells == 50


# --- decisions -------------------------------------------------------------------------
def brain_at(pos, yaw, trail=(), cfg=None):
    b = Brain(cfg or from_dict(ON))
    for i, p in enumerate(trail):
        b.explorer.update(p, float(i))
    b._pos, b.yaw = pos, yaw
    return b


def equal_view(left=2.0, right=2.0):
    return View(blocked=True, blind=False, nearest_front=0.2, front=0.2, left_free=left, right_free=right)


def test_turn_in_place_goes_to_the_less_visited_side_when_both_are_free():
    trail = [(x / 10, y / 10) for x in range(-20, 21) for y in range(3, 20, 2)]   # visited: the left
    b = brain_at((0.0, 0.0), 0.0, trail)
    # Left a little freer, but within turn_margin: exploration chooses the right.
    assert b.pick_turn(equal_view(2.3, 2.0), b.cfg.explore.turn_margin) == -1.0
    assert b.stats.novel_turns == 1


def test_a_side_choice_is_logged_with_what_it_saw():
    trail = [(x / 10, y / 10) for x in range(-20, 21) for y in range(3, 20, 2)]   # visited: the left
    b = brain_at((0.0, 0.0), 0.0, trail)
    b.pick_turn(equal_view(2.3, 2.0), b.cfg.explore.turn_margin)
    t, ev, sign, why = b.explore_log[-1]
    assert ev == "side" and sign == -1.0
    assert why.startswith("overrode the freer side (left)") and "novelty L" in why and "free L 2.30 R 2.00" in why


def test_a_clearly_freer_side_still_wins():
    trail = [(x / 10, y / 10) for x in range(-20, 21) for y in range(-19, -2, 2)]  # visited: the right
    b = brain_at((0.0, 0.0), 0.0, trail)
    # Novelty says left, but the right is freer by more than turn_margin: free space wins.
    assert b.pick_turn(equal_view(0.3, 3.5), b.cfg.explore.turn_margin) == -1.0


def test_disabled_exploration_changes_nothing():
    trail = [(x / 10, y / 10) for x in range(-20, 21) for y in range(3, 20, 2)]
    b = brain_at((0.0, 0.0), 0.0, trail, Config())
    assert b.pick_turn(equal_view(2.3, 2.0), 1.5) == 1.0          # plain freer side
    assert b.explorer.cells == len({(math.floor(x / 0.25), math.floor(y / 0.25)) for x, y in trail})


def test_a_bend_is_aborted_when_something_is_beside_the_body_on_that_side():
    from microduck_brain.brain import Forward

    b = brain_at((0.0, 0.0), 0.0)
    b.heading_target, b.bend = -0.8, (4.0, -0.8)              # bending right
    b.side_busy_until[-1.0] = 10.0                            # a table corner beside the right leg
    Forward._check_bend(b, 5.0)
    assert b.heading_target == b.yaw and b.bend is None and b.stats.bends_aborted == 1
    assert b.explore_log[-1][1] == "aborted" and "right" in b.explore_log[-1][3]


def test_the_ordinary_heading_hold_is_never_aborted():
    """Regression (full simulator, 600 s: 27 'aborts' for 3 bends): with no bend in progress, a
    heading error from the gait's sway near a wall must not reset the heading target."""
    from microduck_brain.brain import Forward

    b = brain_at((0.0, 0.0), 0.0)
    b.heading_target = -0.15                                  # drifted 8.6° off the held heading
    b.side_busy_until[-1.0] = 10.0
    Forward._check_bend(b, 5.0)
    assert b.heading_target == -0.15 and b.stats.bends_aborted == 0


def test_a_bend_completes_when_the_heading_is_reached():
    from microduck_brain.brain import Forward

    b = brain_at((0.0, 0.0), -0.75)
    b.heading_target, b.bend = -0.8, (4.0, -0.8)
    Forward._check_bend(b, 6.0)
    assert b.bend is None and b.stats.bends_completed == 1


def test_side_hazard_marks_its_side_busy():
    b = Brain(from_dict(ON))
    b.hazards.items = [Hazard(0.2, -0.16, SIDE)]            # something beside the right leg
    b.mark_busy_sides(3.0)
    assert b.side_busy_until[-1.0] == 3.0 + b.cfg.explore.bend_every
    assert b.side_busy_until[1.0] < 0


def test_explore_config_validation():
    for bad in ({"cell": 0.0}, {"max_bend_deg": 120.0}, {"novelty_margin": 1.5}, {"bend_every": 0.0}):
        with pytest.raises(ValueError):
            from_dict({"explore": bad})


# --- closed loop in a flat world --------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1])
def test_exploration_covers_more_of_the_living_room_without_touching_anything(seed):
    world = world2d.worlds()[3]                               # furniture: sofa, table, two stools
    base = world2d.run(world, Config(), 300, seed)
    expl = world2d.run(world, from_dict(ON), 300, seed)
    assert base.contacts == 0 and expl.contacts == 0
    assert expl.area > base.area


def test_flat_world_has_no_false_low_objects_or_drops():
    r = world2d.run(world2d.worlds()[0], Config(), 60, 0)
    assert r.stats.low_seen == 0 and r.stats.drops_seen == 0 and r.contacts == 0


def test_exploration_replays_exactly(tmp_path):
    from microduck_brain.record import Recorder, replay

    cfg = from_dict(ON)
    path = tmp_path / "run.jsonl.gz"
    rec = Recorder(path)
    r = world2d.run(world2d.worlds()[3], cfg, 600, 2, recorder=rec)
    rec.close()
    st = r.stats
    assert st.bends > 0 and st.bends_refused > 0
    assert st.bends == st.bends_completed + st.bends_interrupted + st.bends_aborted + (1 if r.bend_open else 0)
    rep = replay(path, cfg)
    assert rep.ticks == 12000 and rep.differing == 0

    # The report replays the same recording exactly and accounts for every bend.
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("explore_report", Path(__file__).parents[1] / "tools" / "explore_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = mod.report(str(path), cfg, 6.0)
    assert out["replay_differing_commands"] == 0
    assert out["bends_started"] == st.bends and sum(out["refusals"].values()) == st.bends_refused
    assert out["after_side_choices"]["events"] == len(out["side_choice_list"])
    assert out["after_every_turn_in_place"]["events"] > 0
    assert sum(out["refusals_by_bend_side"].values()) == st.bends_refused


def test_ab_summary_groups_by_config_and_flags_overlap(tmp_path):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("ab_summary", Path(__file__).parents[1] / "tools" / "ab_summary.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def run_dir(name, enabled, area, avoids):
        d = tmp_path / name
        d.mkdir()
        (d / "brain.toml").write_text(f"[explore]\nenabled = {'true' if enabled else 'false'}\n")
        (d / "summary.json").write_text(json.dumps({
            "commands_differing_from_recording": 0,
            "coverage": {"area_covered_m2": area, "cells_visited": int(area / 0.0625), "distance_m": 80.0,
                         "bounding_box_m": [4.0, 3.0]},
            "replayed": {"avoidances": avoids, "avoidances_with_stop": 30, "big_turns": 4,
                         "seconds_per_state": {"turn": 60.0}, "holds_fall_or_unhealthy": 0},
        }))
        return d

    dirs = [run_dir("a", False, 7.0, 190), run_dir("b", False, 7.4, 200), run_dir("c", True, 8.0, 195), run_dir("d", True, 8.5, 205)]
    res = mod.summarise(dirs)
    assert res["metrics"]["area_m2"]["off"]["mean"] == 7.2 and res["metrics"]["area_m2"]["on"]["n"] == 2
    assert res["metrics"]["area_m2"]["ranges_overlap"] is False          # 7.0–7.4 vs 8.0–8.5
    assert res["metrics"]["avoidances"]["ranges_overlap"] is True        # 190–200 vs 195–205


# --- observe mode -----------------------------------------------------------------------
def test_observe_mode_sends_the_robot_nothing(robot):  # noqa: F811
    import threading

    from microduck_brain.runner import Runner

    cfg = from_dict({"safety": {"resume_delay": 0.2, "health_poll": 0.1}, "loop": {"log_every": 0}})
    runner = Runner(cfg, robot.robot_path, robot.tof_path, observe=True)
    t = threading.Thread(target=runner.run, daemon=True)
    t.start()
    assert wait_for(lambda: runner.brain.state_name == "forward", 3)   # it decides to walk...
    stop(runner, t)
    assert robot.moves == [] and robot.stops == 0                      # ...but sends nothing
