"""tools/first_robot_check.py: an observe recording is checked before the robot may walk."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from microduck_brain.config import Config
from microduck_brain.record import Recorder

from . import world2d

spec = importlib.util.spec_from_file_location(
    "first_robot_check", Path(__file__).resolve().parents[1] / "tools" / "first_robot_check.py")
frc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frc)


def verdicts(path):
    return {name: v for name, v, _ in frc.check(str(path), None)}


def test_a_healthy_recording_passes(tmp_path):
    path = tmp_path / "ok.jsonl.gz"
    rec = Recorder(path)
    world2d.run(world2d.worlds()[0], Config(), 30, 0, recorder=rec, with_pose=True)
    rec.close()
    v = verdicts(path)
    assert "FAIL" not in v.values(), v
    assert v["replay"] == "PASS" and v["frames: rate"] == "PASS"


def test_no_depth_frames_and_no_pose_fail(tmp_path):
    path = tmp_path / "dead.jsonl.gz"
    rec = Recorder(path)
    world2d.run(world2d.worlds()[0], Config(), 30, 0, recorder=rec, frame_every=10 ** 9)
    rec.close()
    v = verdicts(path)
    assert v["pose: present"] == "FAIL"
    assert v.get("frames: any") == "FAIL" or v["frames: rate"] == "FAIL"
