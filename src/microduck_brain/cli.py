"""Punctul de intrare: ``uv run microduck-brain --config brain.toml``."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time

from . import config as config_mod
from .runner import Runner, default_sockets


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="microduck-brain", description="Creier autonom v0 pentru Microduck")
    ap.add_argument("--config", "-c", help="fișier TOML (implicit: valorile din brain.toml, dacă există în directorul curent)")
    ap.add_argument("--robot-socket", help="socket-ul lui robotd (suprascrie configurația)")
    ap.add_argument("--tof-socket", help="socket-ul lui tofd (suprascrie configurația)")
    ap.add_argument("--duration", type=float, default=0.0, help="se oprește singur după atâtea secunde (0 = până la Ctrl+C)")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    path = args.config
    if path is None:
        from pathlib import Path

        path = "brain.toml" if Path("brain.toml").exists() else None
    try:
        cfg = config_mod.load(path)
    except (OSError, ValueError) as e:
        print(f"configurație invalidă: {e}", file=sys.stderr)
        return 2

    robot, tof = default_sockets(cfg)
    robot = args.robot_socket or robot
    tof = args.tof_socket or tof
    logging.info("robot: %s | adâncime: %s | config: %s", robot, tof, path or "(implicită)")

    runner = Runner(cfg, robot, tof)

    def _stop(signum, frame):  # Ctrl+C și SIGTERM => oprire curată
        runner.stop_event.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    started = time.monotonic()
    if args.duration > 0:
        import threading

        timer = threading.Timer(args.duration, runner.stop_event.set)
        timer.daemon = True
        timer.start()
    runner.run()

    st = runner.brain.stats
    summary = {
        "durata_s": round(time.monotonic() - started, 1),
        "ocoliri": st.avoids,
        "intoarceri_mari": st.big_turns,
        "intoarceri_expirate": st.turn_timeouts,
        "opriri_cadere_sau_nesanatos": st.holds_unsafe,
        "opriri_fara_adancime": st.holds_no_depth,
        "mers_inainte_s": round(st.forward_seconds, 1),
        "timp_pe_stari_s": {k: round(v, 1) for k, v in st.state_seconds.items()},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
