"""Regression tests for the bugs found in the audit of 2026-11-04 (each one reproduced first)."""

from __future__ import annotations

import gzip
import math
import os
import socket
import tempfile
import threading

import pytest

from microduck_brain.brain import Brain, Inputs
from microduck_brain.config import Config, from_dict
from microduck_brain.recover import FALLEN

from . import world2d
from .test_brain import clear

FAST = {"safety": {"resume_delay": 0.2}}


# --- 2: configuration ----------------------------------------------------------------------
@pytest.mark.parametrize("bad", [
    {"obstacle": {"stop_distance": float("nan")}},      # walked into a wall 15 cm ahead
    {"safety": {"tof_timeout": float("inf")}},          # kept walking on the last frame of a dead tofd
    {"motion": {"forward_speed": float("nan")}},
    {"obstacle": {"blind_fraction": 0.0}},              # the "blind sensor" check switched off
    {"loop": {"rate_hz": 0.2}},                         # slower than robotd's ~0.5 s move timeout
    {"obstacle": {"max_turn_time": 0.0}},               # face/forward alternating every tick
    {"safety": {"state_timeout": 10.0}},
])
def test_unsafe_configurations_are_rejected(bad):
    with pytest.raises(ValueError):
        from_dict(bad)


def test_the_shipped_brain_toml_is_still_valid():
    import tomllib
    from pathlib import Path

    from_dict(tomllib.loads((Path(__file__).resolve().parents[1] / "brain.toml").read_text()))


# --- 4: NaN odometry -------------------------------------------------------------------------
def test_one_nan_yaw_does_not_poison_the_heading():
    b = Brain(from_dict(FAST))
    t = 0.0
    for i in range(200):
        yaw = math.nan if i == 50 else 0.0
        b.tick(Inputs(t, clear(), True, "", yaw, None, (0.01 * i, 0.0)))
        t += 0.05
    assert b.yaw == 0.0 and b.state_name == "forward"


def test_a_nan_position_does_not_crash_the_map():
    b = Brain(from_dict({**FAST, "explore": {"enabled": True}, "home": {"enabled": True}}))
    for i in range(60):
        pos = (math.nan, 0.0) if i == 30 else (0.01 * i, 0.0)
        b.tick(Inputs(i * 0.05, clear(), True, "", 0.0, None, pos))


# --- 5 and 6: a fall during a hold, and a hold at home ------------------------------------------
def _walked(cfg, seconds=3.0):
    b = Brain(cfg)
    t = 0.0
    while t < seconds:
        b.tick(Inputs(t, clear(), True, "", 0.0, None, (t * 0.1, 0.0)))
        t += 0.05
    return b, t


def test_a_fall_that_comes_during_a_hold_still_resets_home():
    b, t = _walked(from_dict({**FAST, "home": {"enabled": True, "after_seconds": 100.0}}))
    assert b.homing.home is not None
    b.tick(Inputs(t, clear(), False, "robot.health unavailable", 0.0, None, (0.3, 0.0)))
    assert b.state_name == "hold" and b.homing.home is not None      # not a fall: kept
    b.tick(Inputs(t + 0.05, clear(), False, FALLEN, 0.0, None, (0.3, 0.0)))
    assert b.homing.home is None and b.homing.abandoned


def test_a_hold_at_home_goes_back_to_home_not_exploring():
    b = Brain(from_dict({**FAST, "home": {"enabled": True, "after_seconds": 1.0}}))
    t, x = 0.0, 0.0
    while t < 30 and b.state_name != "home":
        cmd = b.tick(Inputs(t, clear(), True, "", 0.0 if b.stats.home_started == 0 else math.pi, None, (x, 0.0)))
        if cmd.kind == "move" and cmd.vx > 0:
            x += 0.05 if b.stats.home_started == 0 else -0.05
        t += 0.05
    assert b.state_name == "home"
    b.tick(Inputs(t, clear(), False, "control loop at 40 Hz", 0.0, None, (x, 0.0)))
    assert b.state_name == "hold"
    for _ in range(20):
        t += 0.05
        b.tick(Inputs(t, clear(), True, "", math.pi, None, (x, 0.0)))
    assert b.state_name == "home"


# --- 3 and 7: malformed input from robotd / tofd -------------------------------------------------
def _runner():
    from microduck_brain.runner import Runner

    return Runner(Config(), "/nonexistent/robot.sock", "/nonexistent/tof.sock")


@pytest.mark.parametrize("odom", [
    {"yaw": None, "position": [None, 0.0, 0.1]},        # serde_json writes NaN as null
    {"yaw": "x", "position": "nope"},
    {"yaw": float("nan"), "position": [float("inf"), 0.0]},
    "not an object",
])
def test_malformed_odometry_reads_as_missing(odom):
    r = _runner()
    r._on_state("robot.state", {"odom": odom, "safety": {}})
    assert r._yaw is None and r._pos is None


def test_a_4x4_frame_is_dropped_not_fatal():
    r = _runner()
    r._on_tof("tof.frame", {"rows": 4, "cols": 4, "distance_mm": [500] * 16, "status": [5] * 16})
    assert r._frame is None


def test_a_bad_stream_message_does_not_end_the_stream():
    from microduck_brain.ipc import Stream

    d = tempfile.mkdtemp()
    path = os.path.join(d, "s.sock")
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    srv.listen(1)
    got, stop = [], threading.Event()

    def on_message(method, params):
        if params.get("boom"):
            raise TypeError("float() argument must be a real number, not 'NoneType'")
        got.append(params["n"])

    st = Stream(path, "robot.subscribe", None, on_message, stop)
    st.start()
    conn, _ = srv.accept()
    conn.recv(1024)
    conn.sendall(b'{"jsonrpc":"2.0","method":"robot.state","params":{"boom":true}}\n'
                 b'{"jsonrpc":"2.0","method":"robot.state","params":{"n":1}}\n')
    for _ in range(100):
        if got:
            break
        stop.wait(0.02)
    stop.set()
    conn.close()
    srv.close()
    st.join(2)
    assert got == [1]


def test_a_garbled_reply_is_an_oserror():
    from microduck_brain.ipc import Link

    d = tempfile.mkdtemp()
    path = os.path.join(d, "r.sock")
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    srv.listen(1)

    def serve():
        c, _ = srv.accept()
        c.recv(1024)
        c.sendall(b"{not json\n")
        c.close()

    th = threading.Thread(target=serve, daemon=True)
    th.start()
    link = Link(path, timeout=1.0)
    link.connect()
    with pytest.raises(OSError):
        link.call("robot.health")
    assert not link.connected
    th.join(2)
    srv.close()


# --- 8: a recording cut off by a power cut --------------------------------------------------------
def test_a_recording_cut_off_mid_write_still_replays(tmp_path):
    from microduck_brain.record import Recorder, read

    path = tmp_path / "run.jsonl.gz"
    rec = Recorder(path)
    world2d.run(world2d.worlds()[0], Config(), 20, 0, recorder=rec)
    rec._fh.flush()
    raw = open(path, "rb").read()          # the file as it is on disk before close(): no gzip trailer
    rec.close()
    cut = tmp_path / "cut.jsonl.gz"
    cut.write_bytes(raw[: len(raw) - 37])  # and a few bytes less, mid-line
    ticks = list(read(cut))
    assert len(ticks) > 300                # everything up to the last flush is there
    full = list(read(path))
    assert [t.inputs.now for t in ticks] == [t.inputs.now for t in full[: len(ticks)]]
    assert gzip.open(path).read()          # the closed file is a normal gzip
