"""Test de integrare: Runner + un robot fals pe socket-uri unix reale (același NDJSON JSON-RPC)."""

from __future__ import annotations

import json
import socketserver
import threading
import time

import pytest

from microduck_brain.config import from_dict
from microduck_brain.runner import Runner

from .test_brain import clear, wall_ahead  # noqa: F401  (cadre sintetice)


class FakeRobot:
    def __init__(self, tmp_path):
        self.robot_path = str(tmp_path / "robot.sock")
        self.tof_path = str(tmp_path / "tof.sock")
        self.fallen = False
        self.healthy = True
        self.frame = clear()
        self.moves: list[tuple[float, dict]] = []
        self.stops = 0
        outer = self

        class RobotH(socketserver.StreamRequestHandler):
            def handle(self):
                for line in self.rfile:
                    msg = json.loads(line)
                    m, id_ = msg["method"], msg.get("id")
                    if m == "robot.move":
                        outer.moves.append((time.monotonic(), msg["params"]))
                    elif m == "robot.stop":
                        outer.stops += 1
                        self._reply(id_, {"accepted": True})
                    elif m == "robot.health":
                        self._reply(id_, {"healthy": outer.healthy, "volts": 7.4, "percent": 50.0})
                    elif m == "robot.subscribe":
                        self._reply(id_, {"accepted": True})
                        while True:
                            st = {"safety": {"fallen": outer.fallen, "limp": False, "gravity": [0, 0, -1]}}
                            self.wfile.write((json.dumps({"jsonrpc": "2.0", "method": "robot.state", "params": st}) + "\n").encode())
                            time.sleep(0.05)

            def _reply(self, id_, result):
                self.wfile.write((json.dumps({"jsonrpc": "2.0", "id": id_, "result": result}) + "\n").encode())

        class TofH(socketserver.StreamRequestHandler):
            def handle(self):
                for line in self.rfile:
                    msg = json.loads(line)
                    self.wfile.write((json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": {"accepted": True, "rows": 8, "cols": 8, "hz": 15}}) + "\n").encode())
                    while True:
                        f = outer.frame
                        p = {"seq": 1, "at_us": 0, "rows": 8, "cols": 8, "distance_mm": list(f.distance_mm), "status": list(f.status)}
                        self.wfile.write((json.dumps({"jsonrpc": "2.0", "method": "tof.frame", "params": p}) + "\n").encode())
                        time.sleep(0.05)

        class Srv(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
            daemon_threads = True

        self.servers = [Srv(self.robot_path, RobotH), Srv(self.tof_path, TofH)]
        for s in self.servers:
            threading.Thread(target=s.serve_forever, daemon=True).start()

    def close(self):
        for s in self.servers:
            s.shutdown()
            s.server_close()


@pytest.fixture
def robot(tmp_path):
    r = FakeRobot(tmp_path)
    yield r
    r.close()


def start(robot, **overrides):
    cfg = from_dict({"safety": {"resume_delay": 0.2, "health_poll": 0.1}, "loop": {"log_every": 0}, **overrides})
    runner = Runner(cfg, robot.robot_path, robot.tof_path)
    t = threading.Thread(target=runner.run, daemon=True)
    t.start()
    return runner, t


def stop(runner, t):
    runner.stop_event.set()
    t.join(3)
    assert not t.is_alive()


def wait_for(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_walks_forward_and_stops_cleanly_on_exit(robot):
    runner, t = start(robot)
    assert wait_for(lambda: any(m["vx"] > 0 for _, m in robot.moves))
    stop(runner, t)
    assert robot.stops >= 1
    assert robot.moves[-1][1]["vx"] == 0.0 and robot.moves[-1][1]["vyaw"] == 0.0


def test_wall_makes_it_turn_in_place(robot):
    robot.frame = wall_ahead()
    runner, t = start(robot)
    assert wait_for(lambda: any(m["vyaw"] != 0 and m["vx"] == 0 for _, m in robot.moves), 4)
    stop(runner, t)
    assert not any(m["vx"] > 0 for _, m in robot.moves)


def test_fall_stops_walk_commands_until_upright(robot):
    runner, t = start(robot)
    assert wait_for(lambda: any(m["vx"] > 0 for _, m in robot.moves))
    robot.fallen = True
    assert wait_for(lambda: robot.stops >= 1)
    time.sleep(0.3)  # lasă să se așeze
    n = len(robot.moves)
    time.sleep(1.0)
    assert len(robot.moves) == n, "nicio comandă de mers cât robotul e căzut"
    robot.fallen = False
    assert wait_for(lambda: len(robot.moves) > n and robot.moves[-1][1]["vx"] > 0, 4)
    stop(runner, t)


def test_unhealthy_robot_gets_no_walk_commands(robot):
    robot.healthy = False
    runner, t = start(robot)
    time.sleep(1.0)
    stop(runner, t)
    assert not any(m["vx"] > 0 for _, m in robot.moves)
