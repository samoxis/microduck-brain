"""The brain's state machine.

The decision logic is pure: ``Brain.tick(Inputs) -> Command``. It touches no sockets and reads no
clock (time arrives in ``Inputs.now``), so it is tested with synthetic and recorded frames.

States::

    hold ──(safe + depth frames, after resume_delay)──> forward
    forward ──(obstacle approaching)──> steers away while walking (still ``forward``)
    forward ──(no room left: under stop_distance)──> brake ──> turn ──(clear)──> forward
                                                       └─(too many stops without progress)──> bigturn ──> forward
    any state ──(fallen / unhealthy / no depth)──> hold

``forward`` does two things: while the way is clear it holds its heading with ``odom.yaw``
(cancelling the gait's drift), and when an obstacle approaches (between ``slow_distance`` and
``stop_distance``) it steers away while walking, harder the closer it gets. Only when there is no
room left (under ``stop_distance``) does it stop and turn in place. ``turn`` and ``bigturn``
measure the angle from ``odom.yaw`` rather than timing it.

**Adding a state** (dance, rest, ...): write a class with ``name`` and
``step(brain, inputs) -> (Command, next_name | None)``, optionally ``enter``, and register it with
``brain.register(MyState())``. The safety guard in ``Brain.tick`` sits above every state: if the
robot falls, whatever state it was in is replaced by ``hold``. A state that does *not* need depth
(dance, rest) sets ``needs_depth = False``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .config import Config
from .depth import DepthFrame, Pose, View, analyse
from .hazards import LOW, HazardMap


@dataclass(frozen=True)
class Inputs:
    now: float                    # seconds, monotonic clock
    frame: DepthFrame | None      # None = no fresh frame
    safe: bool                    # the robot is upright and healthy
    safe_reason: str = ""         # why it is not safe (for the log)
    yaw: float | None = None      # odom.yaw (rad, + = left), or None if the robot doesn't report it
    pose: Pose | None = None      # sensor pose from robot.state, for the floor geometry
    pos: tuple[float, float] | None = None   # odom.position[0:2] (m), for the hazard memory


@dataclass(frozen=True)
class Command:
    kind: str                     # "move" | "stop" | "none"
    vx: float = 0.0
    vyaw: float = 0.0


NONE = Command("none")
STOP = Command("stop")


def move(vx: float, vyaw: float) -> Command:
    return Command("move", vx, vyaw)


@dataclass
class Stats:
    avoids: int = 0               # avoidance episodes started (steering while walking and/or stop + turn)
    brakes: int = 0               # of those, how many needed a full stop and a turn in place
    big_turns: int = 0
    low_seen: int = 0             # new low objects (cubes, balls) seen in the bottom rows
    drops_seen: int = 0           # new drops (ledges, stairs going down)
    turn_timeouts: int = 0        # turns that did not clear within max_turn_time
    holds_unsafe: int = 0         # holds for a fall / unhealthy robot
    holds_no_depth: int = 0       # holds for missing depth frames
    forward_seconds: float = 0.0
    state_seconds: dict[str, float] = field(default_factory=dict)


class State:
    name = "?"
    needs_depth = True

    def enter(self, brain: "Brain", inputs: Inputs) -> None:
        pass

    def step(self, brain: "Brain", inputs: Inputs) -> tuple[Command, str | None]:
        raise NotImplementedError


class Hold(State):
    """Sends no walking commands. Leaves only after everything has been fine for a while."""

    name = "hold"
    needs_depth = False

    def __init__(self) -> None:
        self._ok_since: float | None = None
        self._stop_sent = False

    def enter(self, brain, inputs):
        self._ok_since = None
        self._stop_sent = False
        brain.heading_integral = 0.0
        brain.hazards.forget()   # after a fall the odometry's position can no longer be trusted

    def step(self, brain, inputs):
        cmd = NONE
        if not self._stop_sent:
            self._stop_sent = True
            cmd = STOP
        if inputs.safe and inputs.frame is not None:
            if self._ok_since is None:
                self._ok_since = inputs.now
            if inputs.now - self._ok_since >= brain.cfg.safety.resume_delay:
                return cmd, "forward"
        else:
            self._ok_since = None
        return cmd, None


def _closeness(front: float | None, cfg) -> float:
    """0 = clear (beyond slow_distance), 1 = at stop_distance."""
    o = cfg.obstacle
    if front is None or o.slow_distance <= o.stop_distance:
        return 0.0
    return min(1.0, max(0.0, (o.slow_distance - front) / (o.slow_distance - o.stop_distance)))


class Forward(State):
    """Walk forward. Clear way: hold the heading. Obstacle approaching: steer away while walking."""

    name = "forward"

    def enter(self, brain, inputs):
        self._last = inputs.now
        self._yaw_cmd = 0.0
        self._straight = False

    def step(self, brain, inputs):
        cfg = brain.cfg
        dt = min(0.2, max(0.0, inputs.now - self._last))
        self._last = inputs.now
        view = brain.look(inputs)

        if view.blocked:
            # No room left: stop completely.
            self._start_episode(brain, view)
            brain.stats.brakes += 1
            brain.consecutive_avoids += 1
            return STOP, "brake"

        c = _closeness(view.front, cfg)
        if c > 0.0:
            brain.clear_time = 0.0
            self._start_episode(brain, view)
            self._straight = False
            target = brain.steer_sign * cfg.obstacle.steer_yaw_max * c
        else:
            brain.episode_open = False
            brain.clear_time += dt
            if brain.clear_time >= cfg.corner.progress_time:
                brain.consecutive_avoids = 0   # walked far enough: not a corner
            if not self._straight:
                # Clear again: the current direction becomes the heading to hold.
                self._straight = True
                brain.heading_target = brain.yaw
            target = self._heading_correction(brain, dt)

        # Slew-limit the turn command: no sudden jerks.
        step = cfg.obstacle.steer_slew * dt
        self._yaw_cmd += max(-step, min(step, target - self._yaw_cmd))
        return move(cfg.motion.forward_speed, self._yaw_cmd), None

    @staticmethod
    def _start_episode(brain, view) -> None:
        if not brain.episode_open:
            brain.episode_open = True
            brain.stats.avoids += 1
            brain.steer_sign = brain.pick_turn(view)

    @staticmethod
    def _heading_correction(brain, dt: float) -> float:
        h = brain.cfg.heading
        if not h.enabled or brain.yaw is None or brain.heading_target is None:
            return 0.0
        err = brain.heading_target - brain.yaw
        limit = h.max_correction
        if h.ki > 0:
            brain.heading_integral = max(-limit / h.ki, min(limit / h.ki, brain.heading_integral + err * dt))
        out = h.kp * err + h.ki * brain.heading_integral
        return max(-limit, min(limit, out))


class Brake(State):
    """Stand still briefly so the gait settles before turning."""

    name = "brake"

    def enter(self, brain, inputs):
        self.since = inputs.now

    def step(self, brain, inputs):
        if inputs.now - self.since < brain.cfg.obstacle.stop_pause:
            return move(0.0, 0.0), None
        view = brain.look(inputs)
        brain.turn_sign = brain.pick_turn(view)
        brain.steer_sign = brain.turn_sign
        if brain.consecutive_avoids >= brain.cfg.corner.turns_before_big_turn:
            brain.consecutive_avoids = 0
            brain.stats.big_turns += 1
            return move(0.0, 0.0), "bigturn"
        return move(0.0, 0.0), "turn"


class _Rotation(State):
    """Base for turns in place: measures the angle turned from odom.yaw."""

    def enter(self, brain, inputs):
        self.since = inputs.now
        self.yaw0 = brain.yaw

    def turned(self, brain) -> float | None:
        """Angle turned in the turning direction (rad, >= 0 when all is well), or None without odometry."""
        if self.yaw0 is None or brain.yaw is None:
            return None
        return brain.turn_sign * (brain.yaw - self.yaw0)


class Turn(_Rotation):
    """Turn towards the freer side until the way ahead is clear (and the minimum angle is covered)."""

    name = "turn"

    def step(self, brain, inputs):
        cfg = brain.cfg
        elapsed = inputs.now - self.since
        angle = self.turned(brain)
        enough = elapsed >= cfg.obstacle.min_turn_time and (
            angle is None or angle >= math.radians(cfg.obstacle.min_turn_degrees)
        )
        if enough:
            view = brain.look(inputs, cfg.obstacle.clear_distance)
            if not view.blocked:
                return move(0.0, 0.0), "forward"
        if elapsed >= cfg.obstacle.max_turn_time:
            # It won't clear: probably a corner. Go straight to the big turn.
            brain.stats.turn_timeouts += 1
            brain.consecutive_avoids = 0
            brain.stats.big_turns += 1
            return move(0.0, 0.0), "bigturn"
        return move(0.0, brain.turn_sign * cfg.motion.turn_speed), None


class BigTurn(_Rotation):
    """A big turn in place, to the MEASURED angle (odom.yaw)."""

    name = "bigturn"

    def enter(self, brain, inputs):
        super().enter(brain, inputs)
        c = brain.cfg.corner
        self.target = math.radians(c.big_turn_degrees)
        # Without odometry: timed, assuming the robot achieves only part of the commanded rate.
        self.fallback = self.target / (brain.cfg.motion.big_turn_speed * c.fallback_turn_efficiency)

    def step(self, brain, inputs):
        elapsed = inputs.now - self.since
        angle = self.turned(brain)
        if angle is not None:
            done = angle >= self.target
        else:
            done = elapsed >= self.fallback
        if not done and elapsed >= brain.cfg.corner.big_turn_max_time:
            brain.stats.turn_timeouts += 1   # the angle never grew: frozen odometry or a wrong sign
            done = True
        if done:
            return move(0.0, 0.0), "forward"
        return move(0.0, brain.turn_sign * brain.cfg.motion.big_turn_speed), None


class Brain:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.stats = Stats()
        self.consecutive_avoids = 0
        self.turn_sign = 1.0 if cfg.obstacle.default_turn == "left" else -1.0
        self.steer_sign = self.turn_sign
        self.episode_open = False       # the current avoidance has already been counted
        self.clear_time = 0.0           # how long it has walked with a clear way, continuously
        self.yaw: float | None = None   # unwrapped odom.yaw (no jump at ±pi), + = left
        self._raw_yaw: float | None = None
        self.heading_target: float | None = None
        self.heading_integral = 0.0
        self.hazards = HazardMap(cfg, self.stats)
        self.states: dict[str, State] = {}
        for s in (Hold(), Forward(), Brake(), Turn(), BigTurn()):
            self.register(s)
        self.state: State = self.states["hold"]
        self._last_now: float | None = None
        self._started = False
        self.last_hold_reason = ""

    def register(self, state: State) -> None:
        self.states[state.name] = state

    @property
    def state_name(self) -> str:
        return self.state.name

    def _track_yaw(self, raw: float | None) -> None:
        if raw is None:
            self.yaw = None
            self._raw_yaw = None
            return
        raw *= self.cfg.heading.yaw_sign
        if self._raw_yaw is None or self.yaw is None:
            self.yaw = raw
        else:
            d = (raw - self._raw_yaw + math.pi) % (2 * math.pi) - math.pi
            self.yaw += d
        self._raw_yaw = raw

    def look(self, inputs: Inputs, threshold: float | None = None) -> View:
        """The view ahead: walls from the upper rows plus low objects and drops (seen now or
        remembered), see ``hazards.py``."""
        o = self.cfg.obstacle
        view = analyse(inputs.frame, o, threshold)
        return self.hazards.merge(view, o.stop_distance if threshold is None else threshold)

    @property
    def low_items(self) -> list[tuple[float, float]]:
        """Current low objects (forward, left), seen or remembered."""
        return [(h.forward, h.lateral) for h in self.hazards.items if h.kind == LOW]

    def pick_turn(self, view: View) -> float:
        """+1 = left, -1 = right: the freer side; on a tie, the configured default."""
        side = view.freer_side
        if side is None:
            side = self.cfg.obstacle.default_turn
        return 1.0 if side == "left" else -1.0

    def _goto(self, name: str, inputs: Inputs) -> None:
        self.state = self.states[name]
        self.state.enter(self, inputs)

    def tick(self, inputs: Inputs) -> Command:
        # time accounting
        if self._last_now is not None:
            dt = max(0.0, inputs.now - self._last_now)
            n = self.state.name
            self.stats.state_seconds[n] = self.stats.state_seconds.get(n, 0.0) + dt
            if n == "forward":
                self.stats.forward_seconds += dt
        self._last_now = inputs.now
        self._track_yaw(inputs.yaw)
        self.hazards.update(inputs.frame, inputs.pose, inputs.pos, self.yaw, inputs.safe, inputs.now)

        if not self._started:
            self._started = True
            self.state.enter(self, inputs)

        # The guard: above every state.
        if self.state.name != "hold":
            reason = None
            if not inputs.safe:
                reason = "unsafe"
                self.stats.holds_unsafe += 1
            elif inputs.frame is None and self.state.needs_depth:
                reason = "no_depth"
                self.stats.holds_no_depth += 1
            if reason:
                self.last_hold_reason = inputs.safe_reason or reason
                self._goto("hold", inputs)

        cmd, nxt = self.state.step(self, inputs)
        if nxt is not None:
            self._goto(nxt, inputs)
        return cmd
