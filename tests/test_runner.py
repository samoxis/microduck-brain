"""Test de integrare: Runner + un robot fals pe socket-uri unix reale (același NDJSON JSON-RPC)."""

from __future__ import annotations

import json
import math
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
        # „corpul": rotirea reală = eff x cerută (+ deriva cât merge); raportat ca odom.yaw
        self.eff = 0.5
        self.bias = 0.0
        self.yaw = 0.0                      # nedesfășurat
        self.yaw_log: list[float] = []      # yaw-ul real în momentul fiecărei comenzi
        self._closed = False
        outer = self

        def body():
            last = time.monotonic()
            while not outer._closed:
                time.sleep(0.01)
                now = time.monotonic()
                dt, last = now - last, now
                if outer.moves and now - outer.moves[-1][0] < 0.5:     # deadman ca la robotd
                    m = outer.moves[-1][1]
                    outer.yaw += (outer.eff * m["vyaw"] + (outer.bias if m["vx"] > 0 else 0.0)) * dt

        threading.Thread(target=body, daemon=True).start()

        class RobotH(socketserver.StreamRequestHandler):
            def handle(self):
                for line in self.rfile:
                    msg = json.loads(line)
                    m, id_ = msg["method"], msg.get("id")
                    if m == "robot.move":
                        outer.moves.append((time.monotonic(), msg["params"]))
                        outer.yaw_log.append(outer.yaw)
                    elif m == "robot.stop":
                        outer.stops += 1
                        self._reply(id_, {"accepted": True})
                    elif m == "robot.health":
                        self._reply(id_, {"healthy": outer.healthy, "volts": 7.4, "percent": 50.0})
                    elif m == "robot.subscribe":
                        self._reply(id_, {"accepted": True})
                        while True:
                            wrapped = (outer.yaw + math.pi) % (2 * math.pi) - math.pi
                            st = {
                                "safety": {"fallen": outer.fallen, "limp": False, "gravity": [0, 0, -1]},
                                "odom": {"position": [0.0, 0.0, 0.12], "yaw": wrapped},
                            }
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
        self._closed = True
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


def test_big_turn_closed_loop_over_sockets_uses_the_measured_angle(robot):
    """Robotul fals se rotește doar cu 50% din rata cerută: un cronometru l-ar opri la jumătate."""
    robot.frame = wall_ahead()
    runner, t = start(
        robot,
        corner={"turns_before_big_turn": 1, "big_turn_degrees": 90.0},
        motion={"big_turn_speed": 1.4},
    )
    # prima rotire în loc, continuă (vx == 0, vyaw != 0) până se oprește
    def spinning():
        return [i for i, (_, m) in enumerate(robot.moves) if m["vx"] == 0 and m["vyaw"] != 0]
    assert wait_for(lambda: len(spinning()) > 5, 4)
    first = spinning()[0]
    robot.frame = clear()          # după întoarcere drumul e liber
    assert wait_for(lambda: robot.moves[-1][1]["vx"] > 0, 8)
    last_spin = max(i for i in spinning() if i < len(robot.moves) - 1)
    stop(runner, t)
    turned = abs(robot.yaw_log[last_spin] - robot.yaw_log[first])
    assert math.radians(85) <= turned <= math.radians(110), f"{math.degrees(turned):.0f}°"


def test_heading_hold_over_sockets_cancels_drift(robot):
    robot.bias = 0.12
    robot.frame = clear()
    runner, t = start(robot)
    assert wait_for(lambda: len(robot.moves) > 10)
    y0 = robot.yaw
    time.sleep(6.0)
    stop(runner, t)
    assert abs(robot.yaw - y0) < 0.35, "fără corecție ar fi ~0.7 rad în 6 s"
    assert any(abs(m["vyaw"]) > 0.01 for _, m in robot.moves), "corecția trebuie să apară în comenzi"
