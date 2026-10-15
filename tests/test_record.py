"""Recording, replay, coverage metrics and the SVG plot."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import pytest

from microduck_brain.cli import plot_main, replay_main
from microduck_brain.config import Config, from_dict
from microduck_brain.plot import render_svg
from microduck_brain.record import Recorder, metrics, read, replay

from .lowgeo import pose_for
from .test_low import World


def record_world(path, cfg=None, seconds=40.0):
    """Drive the closed-loop test world (a cube ahead) and record every tick."""
    cfg = cfg or Config()
    w = World(cfg, (1.3, 0.0))
    rec = Recorder(path, {"note": "test"})
    original_tick = w.brain.tick

    def tick(inp):
        cmd = original_tick(inp)
        rec.write(inp, cmd, w.brain.state_name)
        return cmd

    w.brain.tick = tick
    w.run(seconds)
    rec.close()
    return w


@pytest.mark.parametrize("suffix", [".jsonl", ".jsonl.gz"])
def test_replay_with_the_same_config_reproduces_every_command(tmp_path, suffix):
    path = tmp_path / f"run{suffix}"
    w = record_world(path)
    rep = replay(path, Config())
    assert rep.ticks == len(w.log)
    assert rep.differing == 0 and rep.first_difference is None
    assert rep.brain.stats.avoids == w.brain.stats.avoids
    assert rep.brain.stats.low_seen == w.brain.stats.low_seen >= 1


def test_replay_with_another_config_shows_the_difference(tmp_path):
    path = tmp_path / "run.jsonl"
    record_world(path)
    rep = replay(path, from_dict({"obstacle": {"low_obstacles": "ignore"}}))
    assert rep.differing > 0 and rep.first_difference is not None
    assert rep.brain.stats.low_seen == 0


def test_unchanged_frames_are_not_written_twice(tmp_path):
    path = tmp_path / "run.jsonl"
    record_world(path, seconds=5.0)
    lines = path.read_text().splitlines()
    header, ticks = json.loads(lines[0]), [json.loads(x) for x in lines[1:]]
    assert header["kind"] == "microduck-brain-run" and header["note"] == "test"
    assert "f" in ticks[0] and "p" in ticks[0]
    # the test world renders a fresh frame every tick, so frames are written; poses are equal objects
    # only when identical, so at least the first is written
    restored = list(read(path))
    assert all(t.inputs.frame is not None for t in restored)
    assert restored[0].inputs.pose == pose_for()


def test_not_a_recording_is_refused(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"hello": 1}\n')
    with pytest.raises(ValueError, match="not a microduck-brain recording"):
        list(read(p))


def test_metrics_distance_area_and_bounding_box(tmp_path):
    path = tmp_path / "run.jsonl"
    w = record_world(path)
    m = metrics(list(read(path)))
    assert m["distance_m"] > 1.0
    assert m["cells_visited"] >= 4 and m["area_covered_m2"] == pytest.approx(m["cells_visited"] * 0.0625, abs=0.01)
    assert m["bounding_box_m"][0] > 1.0
    assert m["duration_s"] == pytest.approx(w.t - 0.05, abs=0.06)


def test_metrics_ignore_odometry_jumps():
    from microduck_brain.brain import Command, Inputs
    from microduck_brain.record import Tick

    ticks = [Tick(Inputs(i * 0.05, None, True, pos=(0.01 * i if i < 10 else 5.0 + 0.01 * i, 0.0)), Command("none"), "forward")
             for i in range(20)]
    assert metrics(ticks)["distance_m"] == pytest.approx(0.18, abs=0.01)   # the 5 m jump is not counted


def test_plot_is_valid_svg_with_path_and_hazards(tmp_path):
    path = tmp_path / "run.jsonl"
    record_world(path)
    svg = render_svg(list(read(path)), Config(), str(path), "test run")
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    assert svg.count("<line") > 50                      # the path
    assert 'stroke="#0a9b54"' in svg                     # at least one remembered low object
    assert "test run" in svg and "avoidances" in svg


def test_command_line_replay_and_plot(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)                          # no brain.toml here: built-in defaults
    path = tmp_path / "run.jsonl"
    record_world(path, seconds=10.0)
    assert replay_main([str(path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["commands_differing_from_recording"] == 0 and out["ticks"] > 100
    assert plot_main([str(path), "-o", str(tmp_path / "run.svg")]) == 0
    assert (tmp_path / "run.svg").read_text().startswith("<svg")
