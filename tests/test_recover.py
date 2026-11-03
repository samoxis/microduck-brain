"""Recovery of a robot that stays down (recover.py): the sequence, its timing, its limits."""

from __future__ import annotations

from microduck_brain.brain import Brain, Inputs
from microduck_brain.config import from_dict
from microduck_brain.recover import FALLEN

from .test_brain import clear

ON = {"recover": {"enabled": True}, "safety": {"resume_delay": 0.2}}


def run(brain, start, seconds, safe, reason=FALLEN, dt=0.05):
    calls, t = [], start
    while t < start + seconds - 1e-9:
        cmd = brain.tick(Inputs(t, clear(), safe, "" if safe else reason, 0.0, None, (0.0, 0.0)))
        if cmd.kind == "call":
            calls.append((round(t, 2), cmd.method, cmd.params))
        t += dt
    return calls, t


def up(brain):
    _, t = run(brain, 0.0, 1.0, True)
    assert brain.state_name == "forward"
    return t


def test_the_sequence_after_after_seconds_down():
    b = Brain(from_dict(ON))
    t = up(b)
    calls, _ = run(b, t, 12.0, False)
    methods = [(m, p) for _, m, p in calls]
    assert methods == [("robot.enable", {"on": False}), ("robot.init", {}), ("robot.enable", {"on": True})]
    first = calls[0][0] - t
    assert 6.0 <= first < 6.2                                   # after_seconds
    assert calls[1][0] - calls[0][0] >= 0.5 and calls[2][0] - calls[1][0] >= 2.5
    assert b.stats.recovery_attempts == 1


def test_it_retries_at_most_max_attempts_times_per_fall():
    b = Brain(from_dict(ON))
    t = up(b)
    calls, _ = run(b, t, 120.0, False)
    assert b.stats.recovery_attempts == 3 and len(calls) == 9 and b.stats.recoveries_ok == 0


def test_getting_up_counts_as_a_recovery_and_resets():
    b = Brain(from_dict(ON))
    t = up(b)
    _, t = run(b, t, 10.0, False)                               # the sequence has run
    _, t = run(b, t, 2.0, True)                                 # up again
    assert b.stats.recoveries_ok == 1 and b.state_name == "forward"
    calls, _ = run(b, t, 7.0, False)                            # a new fall: a fresh count
    assert calls and b.stats.recovery_attempts == 2


def test_nothing_for_other_unsafe_reasons_or_when_off():
    b = Brain(from_dict(ON))
    t = up(b)
    calls, _ = run(b, t, 20.0, False, reason="control loop at 40.0 Hz, below the 45.0 Hz floor")
    assert calls == []
    b = Brain(from_dict({"safety": {"resume_delay": 0.2}}))
    t = up(b)
    calls, _ = run(b, t, 20.0, False)
    assert calls == []


def test_recovery_calls_replay_exactly(tmp_path):
    from microduck_brain.record import Recorder, read, replay

    cfg = from_dict(ON)
    path = tmp_path / "run.jsonl"
    rec = Recorder(path)
    b, t = Brain(cfg), 0.0
    while t < 20.0:
        safe = t < 1.0 or t > 15.0
        inp = Inputs(t, clear(), safe, "" if safe else FALLEN, 0.0, None, (0.0, 0.0))
        cmd = b.tick(inp)
        rec.write(inp, cmd, b.state_name)
        t += 0.05
    rec.close()
    assert [t.cmd.method for t in read(path) if t.cmd.kind == "call"] == ["robot.enable", "robot.init", "robot.enable"]
    assert replay(path, cfg).differing == 0
    assert replay(path, from_dict({"safety": {"resume_delay": 0.2}})).differing == 3   # off: no calls


def test_recover_config_validation():
    import pytest

    for bad in ({"init_seconds": 0.0}, {"max_attempts": 0}, {"after_seconds": -1.0}):
        with pytest.raises(ValueError):
            from_dict({"recover": bad})
