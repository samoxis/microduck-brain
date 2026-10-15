"""Drops (ledges, stairs going down): real frames from the MuJoCo apartment scene, which has a
4-step staircase going down in the corridor. From the north the first step is at -10 cm, from the
south the last one is at -40 cm."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from microduck_brain.brain import Brain, Inputs
from microduck_brain.config import Config, from_dict
from microduck_brain.depth import DepthFrame, Pose, drop_zones
from microduck_brain.hazards import DROP

FX = json.loads((Path(__file__).parent / "fixtures" / "mujoco_drop.json").read_text())
LOW_FX = json.loads((Path(__file__).parent / "fixtures" / "mujoco_low.json").read_text())
CFG = Config()


def F(f):
    return DepthFrame(8, 8, tuple(f["distance_mm"]), tuple(f["status"]))


def P(f):
    p = f["pose"]
    return Pose(tuple(p["sensor_pos"]), tuple(p["sensor_quat"]), tuple(p["gravity"]), p["trunk_z"])


def labelled(prefix, pitch=0.0):
    return [f for f in FX if f["label"].startswith(prefix) and f["pitch_deg"] == pitch]


def test_no_drop_alarm_in_the_apartment_rooms():
    rooms = [f for f in FX if f["label"].startswith("room")]
    assert len(rooms) == 64
    for f in rooms:
        assert drop_zones(F(f), P(f), CFG) == [], f["label"]
        assert drop_zones(F(f), None, CFG) == [], f["label"]


def test_no_drop_alarm_on_flat_floor_cubes_balls_and_tilted_body():
    for f in LOW_FX:
        assert drop_zones(F(f), P(f), CFG) == []


@pytest.mark.parametrize("label", ["north d=1.0", "north d=0.75", "north d=0.6", "north d=0.45", "north d=0.3", "north d=0.2"])
def test_stair_seen_continuously_when_approaching_from_the_north(label):
    f = [f for f in labelled(label) if f["label"] == label][0]
    zones = drop_zones(F(f), P(f), CFG)
    assert zones, label
    assert all(z.row >= 6 for z in zones)


def test_stair_from_the_south_reads_longer_distances_like_a_real_sensor_would():
    """From the south the beams hit the steps: ranges longer than the floor, not just 'nothing in range'."""
    for d in ("0.9", "0.6", "0.4"):
        f = [f for f in FX if f["label"] == f"south d={d}"][0]
        zones = drop_zones(F(f), P(f), CFG)
        assert zones and any(not z.no_target for z in zones), d


def test_narrow_hole_is_only_visible_in_a_window_hence_the_memory():
    """From the west the hole (0.4 m wide) is visible at 0.4-0.6 m, then the beams land on the floor beyond it."""
    seen = {f["label"]: bool(drop_zones(F(f), P(f), CFG)) for f in labelled("west")}
    assert seen["west d=0.5"] and seen["west d=0.4"]
    assert not seen["west d=0.3"] and not seen["west d=0.2"]


def test_drop_blocks_the_robot_and_is_remembered_after_it_disappears():
    b = Brain(Config())
    t = 0.0
    floor = [f for f in FX if f["label"].startswith("room")][0]
    while t < 2.0:                                      # hold -> forward, on the floor
        b.tick(Inputs(t, F(floor), True, "", 0.0, P(floor), (0.0, 0.0)))
        t += 0.05
    assert b.state_name == "forward"
    hole = [f for f in labelled("west") if f["label"] == "west d=0.5"][0]
    for _ in range(6):
        b.tick(Inputs(t, F(hole), True, "", 0.0, P(hole), (0.0, 0.0)))
        t += 0.05
    assert b.state_name in ("brake", "turn"), "a drop 0.65 m ahead must stop the robot"
    assert b.stats.drops_seen == 1 and b.hazards.remembered(DROP) > 0
    gone = [f for f in labelled("west") if f["label"] == "west d=0.3"][0]   # the hole is no longer visible
    view = None
    for _ in range(10):
        b.tick(Inputs(t, F(gone), True, "", 0.0, P(gone), (0.0, 0.0)))
        view = b.look(Inputs(t, F(gone), True, "", 0.0, P(gone), (0.0, 0.0)))
        t += 0.05
    assert view.blocked, "the remembered drop must keep blocking"


def test_drops_ignore_switch():
    b = Brain(from_dict({"obstacle": {"drops": "ignore"}}))
    hole = [f for f in labelled("west") if f["label"] == "west d=0.5"][0]
    t = 0.0
    while t < 3.0:
        b.tick(Inputs(t, F(hole), True, "", 0.0, P(hole), (0.0, 0.0)))
        t += 0.05
    assert b.hazards.remembered(DROP) == 0 and b.state_name == "forward"


def test_drop_config_validation():
    with pytest.raises(ValueError, match="drops"):
        from_dict({"obstacle": {"drops": "jump"}})
    with pytest.raises(ValueError):
        from_dict({"drop": {"min_zones": 0}})
