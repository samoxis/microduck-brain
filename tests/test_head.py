"""Head scanning (``[head] scan``): the command pattern, the geometry under a turned head, recording."""

from __future__ import annotations

import math

import pytest

from microduck_brain.brain import Brain, Inputs
from microduck_brain.config import Config, from_dict
from microduck_brain.depth import analyse, column_azimuth, head_yaw, rotate

from . import world2d
from .test_brain import FAR, clear, frame
from .test_runner import robot, stop, wait_for  # noqa: F401  (fixture)

SCAN = {"head": {"scan": True}}


def test_rotate_by_zero_is_exact():
    assert rotate(0.123456789, -0.98765, 0.0) == (0.123456789, -0.98765)


def test_head_yaw_from_the_sensor_quaternion():
    for deg in (-40, -10, 0, 25):
        a = math.radians(deg)
        assert head_yaw(world2d._pose(a)) == pytest.approx(a)
    assert head_yaw(None) == 0.0


def test_a_post_seen_with_the_head_turned_is_placed_where_it_is():
    """A post 0.5 m ahead, seen in the left columns while the head looks 30° right, is ahead of the
    trunk; with the head's turn ignored it would look 30° off to the left and outside the corridor."""
    cfg = Config().obstacle
    yaw = math.radians(-30)
    # The two columns whose directions, after the head's turn, point closest to straight ahead.
    cols = sorted(range(8), key=lambda c: abs(column_azimuth(c) + yaw))[:2]
    f = frame(lambda r, c: 0.5 if (c in cols and r <= 4) else FAR)
    assert analyse(f, cfg, yaw=yaw).front == pytest.approx(0.5, abs=0.06)   # ahead of the trunk
    assert analyse(f, cfg, yaw=0.0).front > 1.0                              # turn ignored: off to the side


def test_sides_follow_the_head():
    cfg = Config().obstacle
    near_left_cols = frame(lambda r, c: 0.4 if (c < 4 and r <= 4) else FAR)
    v0 = analyse(near_left_cols, cfg, yaw=0.0)
    assert v0.left_free < v0.right_free
    # Head turned 45° right: the left columns now look ahead-right, so the right side is the near one.
    v = analyse(near_left_cols, cfg, yaw=math.radians(-45))
    assert v.right_free < v.left_free or v.left_free == 0.0


def test_the_head_scans_only_while_walking_and_is_centred_otherwise():
    cfg = from_dict({**SCAN, "safety": {"resume_delay": 0.2}})
    b = Brain(cfg)
    pose = world2d._pose(0.0)
    heads, t = [], 0.0
    while t < 6.0:
        cmd = b.tick(Inputs(t, clear(), True, "", 0.0, pose, (0.0, 0.0)))
        heads.append((b.state_name, cmd.head_yaw))
        t += 0.05
    assert all(h == 0.0 for s, h in heads if s == "hold")
    walking = [h for s, h in heads if s == "forward"]
    amp = math.radians(cfg.head.amplitude_deg)
    assert max(walking) == pytest.approx(amp, abs=0.02) and min(walking) == pytest.approx(-amp, abs=0.02)


def test_no_pose_no_scan_and_scan_off_sends_no_head_command():
    b = Brain(from_dict({**SCAN, "safety": {"resume_delay": 0.2}}))
    for i in range(40):
        cmd = b.tick(Inputs(i * 0.05, clear(), True, "", 0.0, None, (0.0, 0.0)))
    assert b.state_name == "forward" and cmd.head_yaw == 0.0      # can't know where it looks: centred
    b = Brain(Config())
    cmd = b.tick(Inputs(0.0, clear(), True, "", 0.0, None, (0.0, 0.0)))
    assert cmd.head_yaw is None


def test_head_config_validation():
    for bad in ({"amplitude_deg": 90.0}, {"period": 0.0}, {"side_free_seconds": -1.0}):
        with pytest.raises(ValueError):
            from_dict({"head": bad})


def test_scan_avoids_chair_legs_in_the_dining_room():
    """The case the scan is for: 28 chair and table legs, a body 11 cm wide. Measured over 30 seeds:
    2 contacts without the scan, 0 with it; here two of the seeds, for speed."""
    w = world2d.chair_legs()
    for seed in (0, 1):
        r = world2d.run(w, from_dict(SCAN), 300, seed, with_pose=True, body_radius=0.11)
        assert r.contacts == 0
        assert r.stats.low_seen == 0 and r.stats.drops_seen == 0      # the turned head sees no ghosts


def test_scan_replays_exactly(tmp_path):
    from microduck_brain.record import Recorder, read, replay

    cfg = from_dict(SCAN)
    path = tmp_path / "run.jsonl.gz"
    rec = Recorder(path)
    world2d.run(world2d.chair_legs(), cfg, 120, 3, recorder=rec, with_pose=True)
    rec.close()
    heads = [t.cmd.head_yaw for t in read(path)]
    assert all(h is not None for h in heads) and max(heads) > 0.3
    rep = replay(path, cfg)
    assert rep.differing == 0
    assert replay(path, Config()).differing > 0                    # without the scan: no head commands


def test_runner_sends_robot_head_while_scanning_and_centres_it_on_exit(robot):  # noqa: F811
    import threading

    from microduck_brain.runner import Runner

    heads: list[float] = []
    orig = Runner._notify

    def spy(self, method, params):
        if method == "robot.head":
            heads.append(params["head_yaw"])
        return orig(self, method, params)

    cfg = from_dict({**SCAN, "safety": {"resume_delay": 0.2, "health_poll": 0.1}, "loop": {"log_every": 0}})
    Runner._notify = spy
    try:
        runner = Runner(cfg, robot.robot_path, robot.tof_path)
        t = threading.Thread(target=runner.run, daemon=True)
        t.start()
        assert wait_for(lambda: any(abs(h) > 0.05 for h in heads), 4)
        stop(runner, t)
    finally:
        Runner._notify = orig
    assert robot.moves                                               # it walked


# --- head pitch ---------------------------------------------------------------------------
PITCH = {"head": {"pitch": True}}


def _fixture_frames():
    import json
    from pathlib import Path

    from microduck_brain.depth import DepthFrame, Pose

    for f in json.loads((Path(__file__).parent / "fixtures" / "mujoco_head_pitch.json").read_text()):
        p = f["pose"]
        yield f, DepthFrame(8, 8, tuple(f["distance_mm"]), tuple(f["status"])), \
            Pose(tuple(p["sensor_pos"]), tuple(p["sensor_quat"]), tuple(p["gravity"]), p["trunk_z"])


def test_pitched_head_sees_no_ghosts_on_real_mujoco_floor_frames():
    from microduck_brain.depth import drop_zones, geometry, low_obstacles

    cfg = from_dict(PITCH)
    n = 0
    for f, frame_, pose in _fixture_frames():
        if f["shape"] != "none":
            continue
        g = geometry(pose, cfg)
        assert not low_obstacles(frame_, pose, cfg, g) and not drop_zones(frame_, pose, cfg, g)
        assert analyse(frame_, cfg.obstacle, geom=g).front >= 1.0          # the floor is not a wall
        n += 1
    assert n == 48


def test_without_pitch_awareness_a_pitched_head_reads_the_floor_as_a_wall():
    """Why the geometry is needed: the same frames through the level-head arithmetic."""
    cfg = Config()
    alarms = sum(1 for f, frame_, pose in _fixture_frames()
                 if f["shape"] == "none" and f["head"].get("head_pitch", 0) >= 0.35
                 and analyse(frame_, cfg.obstacle).front < 1.0)
    assert alarms == 24


@pytest.mark.parametrize("hp,closest", [(0.0, 0.50), (0.35, 0.20), (0.5, 0.16)])
def test_looking_down_brings_close_cubes_into_view(hp, closest):
    """Real MuJoCo frames: a 7 cm cube is invisible under 45 cm with the head level (finding 3);
    pitched 20 deg down it is seen from 20 cm, with no gap up to 60 cm."""
    from microduck_brain.depth import geometry, low_obstacles

    cfg = from_dict(PITCH)
    seen = {}
    for f, frame_, pose in _fixture_frames():
        if f["shape"] == "cube" and abs(f["head"].get("head_pitch", 0) - hp) < 1e-9:
            hit = bool(low_obstacles(frame_, pose, cfg, geometry(pose, cfg)))
            seen.setdefault(f["x"], []).append(hit)
    assert all(all(v) for x, v in seen.items() if x >= closest)
    assert not any(any(v) for x, v in seen.items() if x < closest)


def test_pitch_goes_down_while_turning_and_stays_level_walking_clear():
    from .test_brain import wall_ahead

    cfg = from_dict({**PITCH, "safety": {"resume_delay": 0.2}})
    b = Brain(cfg)
    pose = world2d._pose(0.0)
    t, yaw, log = 0.0, 0.0, []
    for frame_, secs in ((clear(), 5.0), (wall_ahead(), 2.0), (clear(), 6.0)):
        end = t + secs
        while t < end:
            cmd = b.tick(Inputs(t, frame_, True, "", yaw, pose, (0.0, 0.0)))
            if cmd.kind == "move":
                yaw += 0.5 * cmd.vyaw * 0.05                 # turns at half the commanded rate
            log.append((t, b.state_name, cmd.head_pitch))
            t += 0.05
    down = math.radians(cfg.head.pitch_down_deg)
    assert all(p == 0.0 for tt, s, p in log if s == "forward" and tt < 4.9)          # clear: level
    assert any(p == down for tt, s, p in log if s in ("brake", "turn"))
    assert log[-1][2] == 0.0                                                          # level again


def test_pitch_replays_exactly_and_old_recordings_still_do(tmp_path):
    from microduck_brain.record import Recorder, replay

    cfg = from_dict({**PITCH, "head": {"pitch": True, "scan": True}})
    path = tmp_path / "run.jsonl.gz"
    rec = Recorder(path)
    r = world2d.run(world2d.worlds()[3], cfg, 120, 1, recorder=rec, with_pose=True)
    rec.close()
    assert r.contacts == 0
    assert replay(path, cfg).differing == 0
