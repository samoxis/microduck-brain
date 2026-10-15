"""Entry points.

* ``microduck-brain``         — run the brain against a robot (or the simulator)
* ``microduck-brain-replay``  — re-run a recorded run through the brain, possibly with another config
* ``microduck-brain-plot``    — draw a recorded run as an SVG
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading
import time
from pathlib import Path

from . import config as config_mod
from .runner import Runner, default_sockets


def log_config_warnings(cfg) -> list[str]:
    """Say clearly, at startup, which configured speeds are below the threshold at which the robot steps."""
    warnings = config_mod.speed_warnings(cfg)
    for w in warnings:
        logging.warning("WARNING: %s", w)
    if warnings:
        logging.warning(
            "WARNING: the minimum speeds (%.2f m/s forward, %.1f rad/s turning) were measured in the "
            "simulator; fix brain.toml or the robot will not move.",
            config_mod.MIN_FORWARD_SPEED, config_mod.MIN_TURN_SPEED,
        )
    return warnings


def _load_config(path: str | None):
    if path is None:
        path = "brain.toml" if Path("brain.toml").exists() else None
    return config_mod.load(path), path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="microduck-brain", description="Autonomous behaviour layer for Microduck")
    ap.add_argument("--config", "-c", help="TOML file (default: ./brain.toml if present, else built-in defaults)")
    ap.add_argument("--robot-socket", help="robotd's socket (overrides the config)")
    ap.add_argument("--tof-socket", help="tofd's socket (overrides the config)")
    ap.add_argument("--duration", type=float, default=0.0, help="stop by itself after this many seconds (0 = until Ctrl+C)")
    ap.add_argument("--record", help="write every tick to this file (.jsonl, or .jsonl.gz) for replay and plotting")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        cfg, path = _load_config(args.config)
    except (OSError, ValueError) as e:
        print(f"invalid configuration: {e}", file=sys.stderr)
        return 2

    log_config_warnings(cfg)
    robot, tof = default_sockets(cfg)
    robot = args.robot_socket or robot
    tof = args.tof_socket or tof
    logging.info("robot: %s | depth: %s | config: %s", robot, tof, path or "(defaults)")

    recorder = None
    if args.record:
        from .record import Recorder

        recorder = Recorder(args.record, {"config": path, "started": time.strftime("%Y-%m-%dT%H:%M:%S")})
        logging.info("recording to %s", args.record)

    runner = Runner(cfg, robot, tof, recorder=recorder)

    def _stop(signum, frame):  # Ctrl+C and SIGTERM => clean stop
        runner.stop_event.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    started = time.monotonic()
    if args.duration > 0:
        timer = threading.Timer(args.duration, runner.stop_event.set)
        timer.daemon = True
        timer.start()
    runner.run()

    print(json.dumps(summary(runner.brain.stats, time.monotonic() - started), indent=2))
    return 0


def summary(st, duration: float) -> dict:
    return {
        "duration_s": round(duration, 1),
        "avoidances": st.avoids,
        "avoidances_with_stop": st.brakes,
        "low_objects_seen": st.low_seen,
        "drops_seen": st.drops_seen,
        "big_turns": st.big_turns,
        "turn_timeouts": st.turn_timeouts,
        "holds_fall_or_unhealthy": st.holds_unsafe,
        "holds_no_depth": st.holds_no_depth,
        "walking_forward_s": round(st.forward_seconds, 1),
        "seconds_per_state": {k: round(v, 1) for k, v in st.state_seconds.items()},
    }


def replay_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="microduck-brain-replay",
        description="Re-run a recording through the brain. With the recording's own config the commands "
        "must match exactly; with another config, see what it would have done differently.",
    )
    ap.add_argument("run", help="recording made with microduck-brain --record")
    ap.add_argument("--config", "-c", help="TOML file to replay with (default: ./brain.toml or defaults)")
    args = ap.parse_args(argv)
    from .record import metrics, read, replay

    try:
        cfg, _ = _load_config(args.config)
    except (OSError, ValueError) as e:
        print(f"invalid configuration: {e}", file=sys.stderr)
        return 2
    rep = replay(args.run, cfg)
    ticks = list(read(args.run))
    out = {
        "ticks": rep.ticks,
        "commands_differing_from_recording": rep.differing,
        "first_difference_at_s": rep.first_difference,
        "replayed": summary(rep.brain.stats, ticks[-1].inputs.now - ticks[0].inputs.now if ticks else 0.0),
        "coverage": metrics(ticks),
    }
    print(json.dumps(out, indent=2))
    return 0


def plot_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="microduck-brain-plot", description="Draw a recorded run as an SVG.")
    ap.add_argument("run", help="recording made with microduck-brain --record")
    ap.add_argument("--output", "-o", default="run.svg")
    ap.add_argument("--title", default="")
    ap.add_argument("--config", "-c", help="TOML file (default: ./brain.toml or defaults)")
    args = ap.parse_args(argv)
    from .plot import render_svg
    from .record import read

    try:
        cfg, _ = _load_config(args.config)
    except (OSError, ValueError) as e:
        print(f"invalid configuration: {e}", file=sys.stderr)
        return 2
    ticks = list(read(args.run))
    Path(args.output).write_text(render_svg(ticks, cfg, args.run, args.title), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
