"""Commands from outside ([commands]) and the hand gesture ([gestures])."""

from __future__ import annotations

import os
import stat

import pytest

from microduck_brain.brain import Brain, Inputs
from microduck_brain.config import Config, from_dict
from microduck_brain.gestures import HandGesture

from .test_brain import FAR, clear, frame
from .test_runner import robot, start, stop, wait_for  # noqa: F401  (fixture)

CMD = {"commands": {"enabled": True}, "safety": {"resume_delay": 0.2}}
GEST = {"gestures": {"enabled": True}, "safety": {"resume_delay": 0.2}}


def hand(d=0.10):
    """A palm d metres in front of the head: every wall-row zone, the floor rows unchanged."""
    return frame(lambda r, c: d if r <= 4 else FAR)


class Run:
    def __init__(self, cfg):
        self.b = Brain(from_dict(cfg))
        self.t = 0.0
        self.states: list[str] = []
        self.cmds = []

    def tick(self, f=None, command=None, safe=True, why=""):
        c = self.b.tick(Inputs(self.t, f if f is not None else clear(), safe, why, 0.0, None,
                               (self.t * 0.01, 0.0), command=command))
        self.states.append(self.b.state_name)
        self.cmds.append(c)
        self.t += 0.05
        return c

    def for_(self, seconds, f=None, **kw):
        end = self.t + seconds
        while self.t < end - 1e-9:
            self.tick(f, **kw)


# --- the brain -------------------------------------------------------------------------------
def test_stop_and_go():
    r = Run(CMD)
    r.for_(1.0)
    assert r.b.state_name == "forward"
    c = r.tick(command="stop")
    assert r.b.state_name == "paused" and c.kind == "stop"
    r.for_(2.0)
    assert r.b.state_name == "paused" and all(c.kind == "none" for c in r.cmds[-30:])
    r.tick(command="go")
    assert r.b.state_name == "forward"
    assert r.b.stats.commands == 2


def test_paused_stays_paused_through_a_hold():
    r = Run(CMD)
    r.for_(1.0)
    r.tick(command="stop")
    r.for_(0.3, safe=False, why="control loop at 40 Hz")
    assert r.b.state_name == "hold"
    r.for_(1.0)
    assert r.b.state_name == "paused"


def test_go_never_overrides_the_safety_hold():
    r = Run(CMD)
    r.for_(1.0)
    r.tick(command="stop")
    r.for_(0.3, safe=False, why="the robot has fallen")
    r.tick(command="go", safe=False, why="the robot has fallen")
    assert r.b.state_name == "hold"
    assert all(c.kind != "move" for c in r.cmds[-8:])


def test_start_paused_waits_for_go():
    r = Run({**CMD, "commands": {"enabled": True, "start_paused": True}})
    r.for_(3.0)
    assert r.b.state_name == "paused" and not any(c.kind == "move" for c in r.cmds)
    r.tick(command="go")
    r.for_(0.5)
    assert r.b.state_name == "forward"


def test_toggle_and_home():
    r = Run({**CMD, "home": {"enabled": True, "after_seconds": 1000.0}})
    r.for_(2.0)
    r.tick(command="toggle")
    assert r.b.state_name == "paused"
    r.tick(command="home")
    assert r.b.state_name != "paused" and r.b.stats.home_started == 1 and not r.b.paused


def test_home_without_home_enabled_is_ignored():
    r = Run(CMD)
    r.for_(1.0)
    r.tick(command="home")
    assert r.b.state_name == "forward" and r.b.stats.commands == 0


def test_commands_are_recorded_and_replay_exactly(tmp_path):
    from microduck_brain.record import Recorder, read, replay

    cfg = from_dict(CMD)
    b = Brain(cfg)
    path = tmp_path / "run.jsonl.gz"
    rec = Recorder(path)
    t = 0.0
    for i in range(200):
        cmd = {50: "stop", 120: "go"}.get(i)
        inp = Inputs(t, clear(), True, "", 0.0, None, (t * 0.01, 0.0), command=cmd)
        rec.write(inp, b.tick(inp), b.state_name)
        t += 0.05
    rec.close()
    assert [x.inputs.command for x in read(path) if x.inputs.command] == ["stop", "go"]
    assert replay(path, cfg).differing == 0


def test_commands_off_by_default():
    assert not Config().commands.enabled and not Config().gestures.enabled
    b = Brain(Config())
    assert b.gesture is None and not b.paused


# --- the gesture ---------------------------------------------------------------------------------
def test_hand_held_then_taken_away_stops_a_walking_robot_and_starts_it_again():
    r = Run(GEST)
    r.for_(1.5)
    assert r.b.state_name == "forward"
    r.for_(1.0, hand())                       # it brakes for the hand, and waits instead of turning
    assert r.b.state_name == "brake" and "turn" not in r.states[-20:]
    r.for_(0.6)                                # taken away
    assert r.b.state_name == "paused" and r.b.stats.gestures == 1
    r.for_(2.0)
    assert r.b.state_name == "paused"
    r.for_(1.0, hand())
    r.for_(0.6)
    assert r.b.stats.gestures == 2 and r.b.state_name == "forward"


@pytest.mark.parametrize("held,counts", [(0.3, False), (1.0, True), (5.0, False)])
def test_only_a_hold_of_the_right_length_counts(held, counts):
    g = HandGesture(from_dict(GEST))
    t, fired = 0.0, False
    for f, secs in ((clear(), 0.5), (hand(), held), (clear(), 1.0)):
        end = t + secs
        while t < end:
            fired = g.update(f, t) or fired
            t += 0.05
    assert fired == counts


def test_a_hand_seen_while_turning_in_place_does_not_count():
    """Full-simulator recordings: the only two false gestures in 8.6 h were furniture swept past the
    head during a turn in place."""
    g = HandGesture(from_dict(GEST))
    t, fired = 0.0, False
    for f, secs, turning in ((hand(0.09), 1.0, True), (clear(), 1.0, True)):
        end = t + secs
        while t < end:
            fired = g.update(f, t, turning) or fired
            t += 0.05
    assert not fired


def test_a_far_wall_or_the_floor_is_not_a_hand():
    g = HandGesture(from_dict(GEST))
    assert not g.near(frame(lambda r, c: 0.30 if r <= 4 else FAR))
    assert not g.near(clear())
    assert g.near(hand(0.10))


def test_gesture_replays_exactly(tmp_path):
    from microduck_brain.record import Recorder, replay

    cfg = from_dict(GEST)
    b = Brain(cfg)
    path = tmp_path / "run.jsonl.gz"
    rec = Recorder(path)
    t = 0.0
    for f, secs in ((clear(), 1.5), (hand(), 1.0), (clear(), 2.0)):
        end = t + secs
        while t < end:
            inp = Inputs(t, f, True, "", 0.0, None, (t * 0.01, 0.0))
            rec.write(inp, b.tick(inp), b.state_name)
            t += 0.05
    rec.close()
    assert b.stats.gestures == 1
    assert replay(path, cfg).differing == 0
    assert replay(path, Config()).differing > 0                    # without it: no pause


def test_gesture_config_validation():
    for bad in ({"near": 0.4}, {"hold_seconds": 5.0}, {"min_fraction": 0.0}):
        with pytest.raises(ValueError):
            from_dict({"gestures": bad})


# --- the socket ----------------------------------------------------------------------------------
def test_stop_over_the_socket_stops_the_robot(robot, tmp_path):  # noqa: F811
    from microduck_brain.commands import send

    sock = str(tmp_path / "cmd.sock")
    runner, t = start(robot, commands={"enabled": True, "socket": sock})
    try:
        assert wait_for(lambda: robot.moves, 4)
        assert stat.S_IMODE(os.stat(sock).st_mode) == 0o600          # this user only
        assert send(sock, "brain.command", {"name": "stop"}) == {"queued": "stop"}
        assert wait_for(lambda: runner.brain.state_name == "paused", 2)
        stops = robot.stops
        assert stops >= 1
        assert wait_for(lambda: send(sock, "brain.status").get("paused") is True, 2)
        with pytest.raises(RuntimeError):
            send(sock, "brain.command", {"name": "dance"})
        n = len(robot.moves)
        assert send(sock, "brain.command", {"name": "go"}) == {"queued": "go"}
        assert wait_for(lambda: len(robot.moves) > n + 3, 2)
    finally:
        stop(runner, t)
    assert not os.path.exists(sock)                                 # removed on exit


def test_cli_client_without_a_brain_says_so(tmp_path, capsys):
    from microduck_brain.cli import cmd_main

    assert cmd_main(["stop", "--socket", str(tmp_path / "none.sock")]) == 1
    assert "no answer" in capsys.readouterr().err
