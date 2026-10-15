"""The link between the brain and the robot: read the sensors, run ``Brain.tick``, send commands."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from pathlib import Path

from .brain import Brain, Inputs
from .config import Config
from .depth import DepthFrame, Pose
from .ipc import Link, RpcError, Stream

log = logging.getLogger("microduck_brain")


def default_sockets(cfg: Config) -> tuple[str, str]:
    """Socket paths: from the config, else the simulator's if present, else the real robot's."""
    robot, tof = cfg.sockets.robot, cfg.sockets.tof
    sim = Path(os.environ.get("DUCK_SIM_STATE", Path.home() / ".cache" / "duck-sim"))
    duck = os.environ.get("DUCK_SIM_DUCK", "duck-a")
    if not robot:
        robot = str(sim / f"{duck}.sock") if (sim / f"{duck}.sock").exists() else "/run/robotd.sock"
    if not tof:
        tof = str(sim / f"{duck}-tof.sock") if (sim / f"{duck}-tof.sock").exists() else "/run/tofd/tof.sock"
    return robot, tof


class Runner:
    def __init__(self, cfg: Config, robot_sock: str, tof_sock: str, clock=time.monotonic,
                 recorder=None) -> None:
        self.cfg = cfg
        self.brain = Brain(cfg)
        self.clock = clock
        self.recorder = recorder          # optional: record.Recorder, one line per tick
        self.stop_event = threading.Event()
        self.link = Link(robot_sock)
        self._lock = threading.Lock()
        self._frame: DepthFrame | None = None
        self._frame_at = -1e9
        self._fallen = False
        self._limp = False
        self._state_at = -1e9
        self._yaw: float | None = None
        self._pos: tuple[float, float] | None = None
        self._poses: deque[tuple[int, Pose | None]] = deque(maxlen=64)   # (t_ns, pose) from robot.state
        self._frame_t_ns = 0
        self._healthy = False
        self._health_reason = "not asked yet"
        self._health_at = -1e9
        self.tof = Stream(tof_sock, "tof.stream", None, self._on_tof, self.stop_event)
        self.state = Stream(robot_sock, "robot.subscribe", {"hz": 25}, self._on_state, self.stop_event)

    # --- streams ---------------------------------------------------------------
    def _on_tof(self, method: str, params: dict) -> None:
        if method != "tof.frame":
            return
        try:
            frame = DepthFrame.from_params(params)
        except (KeyError, ValueError, TypeError):
            return
        with self._lock:
            self._frame, self._frame_at = frame, self.clock()
            t_ns = params.get("t_ns")
            self._frame_t_ns = int(t_ns) if isinstance(t_ns, int) else 0

    def _on_state(self, method: str, params: dict) -> None:
        if method != "robot.state":
            return
        safety = params.get("safety") or {}
        with self._lock:
            self._fallen = bool(safety.get("fallen", False))
            self._limp = bool(safety.get("limp", False))
            yaw = (params.get("odom") or {}).get("yaw")
            self._yaw = float(yaw) if isinstance(yaw, (int, float)) else None
            position = (params.get("odom") or {}).get("position")
            self._pos = (float(position[0]), float(position[1])) if isinstance(position, list) and len(position) >= 2 else None
            t_ns = params.get("t_ns")
            self._poses.append((int(t_ns) if isinstance(t_ns, int) else 0, Pose.from_state(params)))
            self._state_at = self.clock()

    def _pose_for_frame(self) -> Pose | None:
        """The pose from the state sample closest in time to the depth frame (shared clock, t_ns);
        without t_ns, the latest pose. More than 100 ms apart is not trusted: better no pose (and
        the wider tolerance) than a wrong one."""
        if not self._poses:
            return None
        if self._frame_t_ns > 0:
            t, pose = min(self._poses, key=lambda e: abs(e[0] - self._frame_t_ns) if e[0] else 1 << 62)
            if t and abs(t - self._frame_t_ns) <= 100_000_000:
                return pose
            return None
        return self._poses[-1][1]

    # --- inputs ----------------------------------------------------------------
    def _poll_health(self, now: float) -> None:
        if now - self._health_at < self.cfg.safety.health_poll:
            return
        self._health_at = now
        try:
            if not self.link.connected:
                self.link.connect()
            res = self.link.call("robot.health")
            self._healthy = bool(res.get("healthy", False))
            self._health_reason = "" if self._healthy else str(res.get("reason") or "unhealthy")
        except (OSError, RpcError) as e:
            self._healthy = False
            self._health_reason = f"robot.health unavailable ({e})"

    def inputs(self, now: float) -> Inputs:
        self._poll_health(now)
        s = self.cfg.safety
        with self._lock:
            frame = self._frame if now - self._frame_at <= s.tof_timeout else None
            fallen, limp = self._fallen, self._limp
            state_age = now - self._state_at
            fresh = state_age <= s.state_timeout
            yaw = self._yaw if fresh else None
            pos = self._pos if fresh else None
            pose = self._pose_for_frame() if fresh else None
        reason = ""
        if state_age > s.state_timeout:
            reason = "no robot.state"
        elif fallen:
            reason = "the robot has fallen"
        elif limp:
            reason = "the robot is limp"
        elif not self._healthy:
            reason = self._health_reason
        return Inputs(now=now, frame=frame, safe=not reason, safe_reason=reason, yaw=yaw, pose=pose, pos=pos)

    # --- output ----------------------------------------------------------------
    def _send(self, cmd) -> None:
        if cmd.kind == "none":
            return
        try:
            if not self.link.connected:
                self.link.connect()
            if cmd.kind == "move":
                self.link.notify("robot.move", {"vx": cmd.vx, "vy": 0.0, "vyaw": cmd.vyaw})
            else:
                self.link.call("robot.stop")
        except (OSError, RpcError) as e:
            log.warning("could not send %s: %s", cmd.kind, e)

    def send_stop(self) -> None:
        """Safe stop on exit: zero velocity + robot.stop, reconnecting once if needed."""
        for _ in range(2):
            try:
                if not self.link.connected:
                    self.link.connect()
                self.link.notify("robot.move", {"vx": 0.0, "vy": 0.0, "vyaw": 0.0})
                self.link.call("robot.stop")
                return
            except (OSError, RpcError):
                self.link.close()

    # --- loop ------------------------------------------------------------------
    def run(self) -> None:
        self.tof.start()
        self.state.start()
        period = 1.0 / self.cfg.loop.rate_hz
        last_log = self.clock()
        last_name = None
        try:
            while not self.stop_event.is_set():
                t0 = self.clock()
                inp = self.inputs(t0)
                cmd = self.brain.tick(inp)
                self._send(cmd)
                name = self.brain.state_name
                if self.recorder is not None:
                    self.recorder.write(inp, cmd, name)
                if name != last_name:
                    extra = f" ({self.brain.last_hold_reason})" if name == "hold" and self.brain.last_hold_reason else ""
                    log.info("state: %s%s", name, extra)
                    last_name = name
                if self.cfg.loop.log_every and t0 - last_log >= self.cfg.loop.log_every:
                    last_log = t0
                    st = self.brain.stats
                    log.info(
                        "state=%s avoids=%d (stops=%d) low_objects=%d drops=%d big_turns=%d safety_holds=%d walked=%.0fs",
                        name, st.avoids, st.brakes, st.low_seen, st.drops_seen, st.big_turns,
                        st.holds_unsafe, st.forward_seconds,
                    )
                self.stop_event.wait(max(0.0, period - (self.clock() - t0)))
        finally:
            self.send_stop()
            self.stop_event.set()
            self.link.close()
            if self.recorder is not None:
                self.recorder.close()
