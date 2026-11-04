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
from .depth import DepthFrame, Pose, View, analyse, column_azimuth, geometry, head_yaw
from .explore import Explorer
from .hazards import LOW, SIDE, HazardMap
from .home import Homing
from .localize import Localizer
from .recover import FALLEN, LIMP, Recovery


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
    head_yaw: float | None = None  # robot.head yaw target (rad, + = left), None = send no head command
    head_pitch: float | None = None  # robot.head head_pitch delta (rad, + = down), with head_yaw
    method: str = ""               # kind "call": a discrete request to robotd (see recover.py)
    params: dict | None = None


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
    bends: int = 0                # exploration: heading bent towards a less visited direction
    bends_completed: int = 0      # ... and the new heading reached
    bends_interrupted: int = 0    # ... ended by an obstacle ahead (the avoidance takes over)
    bends_aborted: int = 0        # ... cancelled: something came beside the body on that side
    bends_refused: int = 0        # a more novel direction existed, but the bend was not started (see the log)
    stuck: int = 0                # walking without moving for obstacle.stuck_seconds: stopped and turned
    home_started: int = 0         # home.py: homing started (after home.after_seconds of walking)
    home_arrived: int = 0         # ... and reached home
    home_aborted: int = 0         # ... and given up (a fall: the odometry can't be trusted)
    home_seconds: float = 0.0     # how long the way home took
    home_waits: int = 0           # ticks it held its heading: something beside the body on the turning side
    recovery_attempts: int = 0    # recover.py: sequences started (stop policy, init, start policy)
    recoveries_ok: int = 0        # ... after which the robot was safe again
    novel_turns: int = 0          # exploration: turned towards the less visited side (both about equally free)
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
        if inputs.safe_reason in (FALLEN, LIMP) or not (brain.cfg.home.enabled or brain.cfg.explore.enabled):
            # The map and the way home go only after a fall; a hold for a slow control loop or a
            # missing frame doesn't move the robot.
            brain.explorer.forget()
            if brain.homing.active or brain.homing.home is not None:
                if brain.homing.active:
                    brain.stats.home_aborted += 1
                brain.homing.reset()
                # Abandoned for the rest of the run: a new "home" set where it got up would be
                # reached and reported as home (a full-simulator run did exactly that, 2.1 m off).
                brain.homing.abandoned = True

    def step(self, brain, inputs):
        cmd = NONE
        if not self._stop_sent:
            self._stop_sent = True
            cmd = STOP
        else:
            call = brain.recovery.tick(inputs.now, inputs.safe, inputs.safe_reason, brain.stats)
            if call is not None:
                cmd = Command("call", method=call[0], params=call[1])
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
        self._progress = (inputs.now, inputs.pos)      # stuck detection: where and when it last moved
        if brain.homing.yield_pending:
            brain.homing.yield_pending = False
            brain.homing.suspended_until = inputs.now + brain.cfg.home.yield_seconds

    def step(self, brain, inputs):
        cfg = brain.cfg
        dt = min(0.2, max(0.0, inputs.now - self._last))
        self._last = inputs.now
        view = brain.look(inputs)

        if self._stuck(brain, inputs):
            # Walking, but the odometry hasn't moved: pressed against something the sensor can't
            # see (a chair leg closer than its shortest range reads as nothing at all).
            self._start_episode(brain, view)
            self._yield_home(brain)
            brain.stuck_turn = brain.away_from_beside(inputs)
            brain.stats.stuck += 1
            brain.stats.brakes += 1
            brain.consecutive_avoids += 1
            return STOP, "brake"

        if view.blocked:
            # No room left: stop completely.
            self._start_episode(brain, view)
            self._yield_home(brain)
            brain.stats.brakes += 1
            brain.consecutive_avoids += 1
            return STOP, "brake"

        c = _closeness(view.front, cfg)
        if c > 0.0:
            brain.clear_time = 0.0
            self._start_episode(brain, view)
            self._straight = False
            brain.end_bend(inputs.now, "interrupted", "obstacle ahead")
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
            if brain.homing.active and inputs.pos is not None and brain.yaw is not None:
                done = self._home_heading(brain, inputs)
                if done is not None:
                    return done
            else:
                self._check_bend(brain, inputs.now)
                self._maybe_bend(brain, inputs, view)
            target = self._heading_correction(brain, dt)

        # Slew-limit the turn command: no sudden jerks.
        step = cfg.obstacle.steer_slew * dt
        self._yaw_cmd += max(-step, min(step, target - self._yaw_cmd))
        return move(cfg.motion.forward_speed, self._yaw_cmd), None

    def _stuck(self, brain, inputs) -> bool:
        limit = brain.cfg.obstacle.stuck_seconds
        if limit <= 0 or inputs.pos is None:
            return False
        t0, p0 = self._progress
        if p0 is None or math.hypot(inputs.pos[0] - p0[0], inputs.pos[1] - p0[1]) >= 0.05:
            self._progress = (inputs.now, inputs.pos)
            return False
        return inputs.now - t0 >= limit

    @staticmethod
    def _home_heading(brain, inputs):
        """Homing: arrived -> stop; the way home far off the heading -> turn to face it in place;
        otherwise hold the heading towards the next waypoint."""
        h = brain.homing
        if h.distance(inputs.pos) <= brain.cfg.home.arrive_radius:
            h.active, h.arrived = False, True
            brain.stats.home_arrived += 1
            brain.stats.home_seconds = inputs.now - h.since
            return STOP, "home"
        if inputs.now < h.suspended_until:
            return None             # just avoided something: let the avoidance carry it clear first
        b = h.bearing(brain.explorer, inputs.pos, inputs.now)
        err = (b - brain.yaw + math.pi) % (2 * math.pi) - math.pi
        if abs(err) > math.radians(brain.cfg.home.face_threshold_deg):
            brain.face_target = brain.yaw + err
            return move(0.0, 0.0), "face"
        side = 1.0 if err > 0 else -1.0
        if abs(err) > 0.1 and (inputs.now < brain.side_busy_until[side]
                               or brain.explorer.wall_beside(inputs.pos, brain.yaw, side)):
            # Something beside the body on the side the way home turns to (the 45 deg view doesn't
            # see beside the legs): keep going straight until it is behind.
            brain.heading_target = brain.yaw
            brain.stats.home_waits += 1
            return None
        if brain.heading_target is None or abs(brain.yaw + err - brain.heading_target) > 0.05:
            brain.heading_target = brain.yaw + err
        return None

    def _maybe_bend(self, brain, inputs, view) -> None:
        """Exploration: now and then, bend the heading towards a less visited direction."""
        e = brain.cfg.explore
        if not e.enabled or inputs.pos is None or brain.yaw is None or brain.heading_target is None:
            return
        if brain.bend is not None or inputs.now - brain.last_bend_check < e.bend_every:
            return
        brain.last_bend_check = inputs.now
        bend = brain.explorer.best_bend(inputs.pos, brain.yaw)
        if bend is None:
            return                            # straight ahead is as novel as anything else
        side = 1.0 if bend > 0 else -1.0
        reason = ""
        if view.front is not None and view.front < e.bend_min_clear:
            reason = f"ahead clear only {view.front:.2f} m"
        elif (view.left_free if side > 0 else view.right_free) < e.bend_min_clear:
            reason = f"{'left' if side > 0 else 'right'} half of the view clear only " \
                     f"{view.left_free if side > 0 else view.right_free:.2f} m"
        elif inputs.now < brain.side_busy_until[side]:
            reason = f"something beside the body on the {'left' if side > 0 else 'right'}"
        elif brain.explorer.wall_beside(inputs.pos, brain.yaw, side):
            reason = f"a mapped wall beside the body on the {'left' if side > 0 else 'right'}"
        elif brain.explorer.wall_on_course(inputs.pos, brain.yaw + bend, brain.cfg.obstacle.corridor_half_width):
            reason = "a mapped wall on the new course"
        if reason:
            brain.stats.bends_refused += 1
            brain.explore_event(inputs.now, "refused", bend, reason)
            return
        brain.heading_target = brain.yaw + bend
        brain.heading_integral = 0.0
        brain.bend = (inputs.now, bend)
        brain.stats.bends += 1
        brain.explore_event(inputs.now, "bend", bend, "")

    @staticmethod
    def _check_bend(brain, now: float) -> None:
        """A bend in progress: done once the heading is reached; aborted if something is now beside
        the body on that side (the 45° view doesn't see beside the legs, finding 6, and the side
        memory lasts less than the slow robot needs to walk past a table corner). Only a bend is
        ever aborted: the ordinary heading hold is left alone."""
        if brain.bend is None or brain.yaw is None or brain.heading_target is None:
            return
        err = brain.heading_target - brain.yaw
        if abs(err) < 0.1:
            brain.end_bend(now, "completed", "")
        elif now < brain.side_busy_until[1.0 if err > 0 else -1.0]:
            brain.heading_target = brain.yaw
            brain.heading_integral = 0.0
            brain.end_bend(now, "aborted", f"something beside the body on the {'left' if err > 0 else 'right'}")
        elif brain._pos is not None and brain.explorer.wall_beside(brain._pos, brain.yaw, 1.0 if err > 0 else -1.0):
            brain.heading_target = brain.yaw
            brain.heading_integral = 0.0
            brain.end_bend(now, "aborted", f"a mapped wall beside the body on the {'left' if err > 0 else 'right'}")

    @staticmethod
    def _yield_home(brain) -> None:
        """Homing gives way after a full stop: without this, the way home turns the robot straight back
        into what it stopped for, and the two alternate for ever (a robot wedged between a table leg
        and a chair leg, in the 2-D dining room). Only after a stop: steering past a wall happens all
        the time along a wall, and yielding to it kept a robot circling a room for ever."""
        if brain.homing.active:
            brain.homing.yield_pending = True   # counted from when it walks again (Forward.enter)

    @staticmethod
    def _start_episode(brain, view) -> None:
        if not brain.episode_open:
            brain.episode_open = True
            brain.stats.avoids += 1
            brain.steer_sign = brain.pick_turn(view, brain.cfg.explore.steer_margin)

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
        brain.turn_sign = brain.pick_turn(view, brain.cfg.explore.turn_margin)
        if brain.stuck_turn is not None:
            # Stuck against something beside the body: turn away from it, all the way round. A short
            # turn towards it (whatever the view ahead said) left the body touching it.
            brain.turn_sign, brain.stuck_turn = brain.stuck_turn, None
            brain.steer_sign = brain.turn_sign
            brain.stats.big_turns += 1
            return move(0.0, 0.0), "bigturn"
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


class Face(_Rotation):
    """Homing: turn in place to face the way home (measured with odom.yaw)."""

    name = "face"

    def step(self, brain, inputs):
        elapsed = inputs.now - self.since
        if brain.yaw is None or brain.face_target is None or elapsed >= brain.cfg.obstacle.max_turn_time:
            return move(0.0, 0.0), "forward"
        err = brain.face_target - brain.yaw
        if abs(err) < math.radians(15):
            brain.heading_target = brain.face_target
            return move(0.0, 0.0), "forward"
        return move(0.0, (1.0 if err > 0 else -1.0) * brain.cfg.motion.turn_speed), None


class AtHome(State):
    """Homing: arrived. Stand still (one robot.stop), and stay."""

    name = "home"
    needs_depth = False

    def enter(self, brain, inputs):
        self._sent = False

    def step(self, brain, inputs):
        if not self._sent:
            self._sent = True
            return STOP, None
        return NONE, None


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
        self.recovery = Recovery(cfg)
        self.homing = Homing(cfg)
        self.localizer = Localizer(cfg)
        self.face_target: float | None = None
        self.stuck_turn: float | None = None   # after getting stuck: turn this way (+1 left), all the way
        self.explorer = Explorer(cfg)
        self.last_bend_check = -1e9      # exploration: when a bend was last considered (any state)
        self.side_busy_until = {1.0: -1e9, -1.0: -1e9}   # exploration: no bends towards that side until then
        self.bend: tuple[float, float] | None = None     # exploration: (started at, angle) of the bend in progress
        self.explore_log: list[tuple[float, str, float, str]] = []   # (t, event, angle rad, reason), drained by the runner
        self.head_now = 0.0             # where the head looks, measured (rad, + = left); 0 unless scanning
        self._scan_t = 0.0              # time spent scanning in the current forward stretch
        self._side_free: dict[str, tuple[float, float]] = {}   # side -> (free, t), while scanning
        self._last_turn_at = -1e9       # head pitch: when the robot last stopped or turned in place
        self._close_ahead = False       # head pitch: something within slow_distance ahead, last look
        self._pos: tuple[float, float] | None = None
        self.states: dict[str, State] = {}
        for s in (Hold(), Forward(), Brake(), Turn(), BigTurn(), Face(), AtHome()):
            self.register(s)
        self.state: State = self.states["hold"]
        self._last_now: float | None = None
        self._started = False
        self.last_hold_reason = ""

    def explore_event(self, now: float, event: str, angle: float, reason: str) -> None:
        self.explore_log.append((now, event, angle, reason))
        if len(self.explore_log) > 1000:
            del self.explore_log[:500]

    def end_bend(self, now: float, how: str, reason: str) -> None:
        if self.bend is None:
            return
        _, angle = self.bend
        self.bend = None
        if how == "completed":
            self.stats.bends_completed += 1
        elif how == "interrupted":
            self.stats.bends_interrupted += 1
        else:
            self.stats.bends_aborted += 1
        self.explore_event(now, how, angle, reason)

    def away_from_beside(self, inputs: Inputs) -> float | None:
        """+1 (turn left) / -1 (right) away from what is nearest beside the body (side memory, then
        the wall map); None if nothing is known there."""
        best = None
        for h in self.hazards.items:
            if h.kind == SIDE and -0.15 < h.forward < 0.6 and abs(h.lateral) < 0.35:
                if best is None or abs(h.lateral) < abs(best):
                    best = h.lateral
        if best is not None:
            return -1.0 if best >= 0 else 1.0
        if inputs.pos is not None and self.yaw is not None:
            lat = self.explorer.nearest_beside(inputs.pos, self.yaw)
            if lat is not None:
                return -1.0 if lat >= 0 else 1.0
        return None

    def _localized(self, inputs: Inputs) -> Inputs:
        """Odometry corrected against the walls seen before (localize.py); a fall resets it, like
        the map, since the odometry's position can't be trusted after one."""
        from dataclasses import replace

        lz = self.localizer
        if not inputs.safe and inputs.safe_reason in (FALLEN, LIMP):
            lz.reset()
        elif inputs.safe and inputs.yaw is not None:
            sx = inputs.pose.sensor_pos[0] if inputs.pose else self.cfg.low.nominal_sensor_forward
            raw = inputs.yaw * self.cfg.heading.yaw_sign
            lz.update(inputs.frame, inputs.pos, raw, sx, inputs.now, head_yaw(inputs.pose) if self.cfg.head.scan else 0.0)
        pos, yaw = lz.correct(inputs.pos, None if inputs.yaw is None else inputs.yaw * self.cfg.heading.yaw_sign)
        if yaw is not None:
            yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
            yaw *= self.cfg.heading.yaw_sign
        return replace(inputs, pos=pos, yaw=yaw)

    def mark_busy_sides(self, now: float) -> None:
        """Exploration: a side with something near the body (side memory) gets no bends for a while."""
        for h in self.hazards.items:
            if h.kind == SIDE and -0.15 < h.forward < 0.6 and abs(h.lateral) < 0.35:
                self.side_busy_until[1.0 if h.lateral >= 0 else -1.0] = now + self.cfg.explore.bend_every

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
        geom = geometry(inputs.pose, self.cfg)
        view = analyse(inputs.frame, o, threshold, self.head_now if geom is None else 0.0, geom)
        if self.cfg.head.scan:
            view = self._remember_sides(view, inputs.now)
        view = self.hazards.merge(view, o.stop_distance if threshold is None else threshold)
        if threshold is None:
            self._close_ahead = view.front is not None and view.front < o.slow_distance
        return view

    def _remember_sides(self, view: View, now: float) -> View:
        """While the head looks to one side, the other half of the view has few or no columns: use
        what that side looked like the last time the head saw it (within side_free_seconds)."""
        from dataclasses import replace

        out = {}
        for side in ("left", "right"):
            cols = sum(1 for c in range(8) if (column_azimuth(c) + self.head_now > 0) == (side == "left"))
            value = view.left_free if side == "left" else view.right_free
            if cols >= 2:
                self._side_free[side] = (value, now)
                out[side] = value
            else:
                old = self._side_free.get(side)
                out[side] = old[0] if old and now - old[1] <= self.cfg.head.side_free_seconds else value
        return replace(view, left_free=out["left"], right_free=out["right"])

    def _head_target(self, inputs: Inputs, dt: float) -> float | None:
        """Scan only while walking forward with a pose (without frames.tof the brain can't know where
        the head looks); anywhere else, centre the head."""
        h = self.cfg.head
        if not h.scan:
            return None if not h.pitch else 0.0
        if self.state.name != "forward" or inputs.pose is None:
            self._scan_t = 0.0
            return 0.0
        self._scan_t += dt
        return math.radians(h.amplitude_deg) * math.sin(2 * math.pi * self._scan_t / h.period)

    def _pitch_target(self, inputs: Inputs) -> float | None:
        """Look down where the 7 cm objects hide under the level beams (finding 3): while stopping or
        turning in place, for ``pitch_after_turn`` s after, and while something is close ahead.
        Level while walking with a clear way, to see far. Needs a pose, like the scan."""
        h = self.cfg.head
        if not h.pitch:
            return None
        if inputs.pose is None or self.state.name in ("hold", "home"):
            return 0.0
        down = math.radians(h.pitch_down_deg)
        if self.state.name in ("brake", "turn", "bigturn", "face"):
            self._last_turn_at = inputs.now
            return down
        if not h.pitch_while_walking:
            # Looking down while walking hid what passes beside the legs (full simulator: a desk
            # leg touched 4-22 times a run): only look down while standing or turning in place.
            return 0.0
        if inputs.now - self._last_turn_at < h.pitch_after_turn or self._close_ahead:
            return down
        return 0.0

    @property
    def low_items(self) -> list[tuple[float, float]]:
        """Current low objects (forward, left), seen or remembered."""
        return [(h.forward, h.lateral) for h in self.hazards.items if h.kind == LOW]

    def pick_turn(self, view: View, margin: float = 0.0) -> float:
        """+1 = left, -1 = right: the freer side; on a tie, the configured default. With exploration,
        when the two sides' free space differs by less than ``margin`` (m), the less visited side."""
        side = view.freer_side
        e = self.cfg.explore
        if e.enabled and self._pos is not None and self.yaw is not None \
                and abs(view.left_free - view.right_free) < margin:
            nl, nr = self.explorer.side_novelty(self._pos, self.yaw)
            if abs(nl - nr) >= 0.1:
                novel = "left" if nl > nr else "right"
                if side is None:
                    how = "tie broken"
                elif novel != side:
                    how = f"overrode the freer side ({side})"
                else:
                    how = "agreed with the freer side"
                if novel != side:
                    self.stats.novel_turns += 1
                self.explore_event(
                    self._last_now or 0.0, "side", 1.0 if novel == "left" else -1.0,
                    f"{how}: margin {margin:.1f} m, novelty L {nl:.2f} R {nr:.2f}, "
                    f"free L {view.left_free:.2f} R {view.right_free:.2f} m",
                )
                side = novel
        if side is None:
            side = self.cfg.obstacle.default_turn
        return 1.0 if side == "left" else -1.0

    def _goto(self, name: str, inputs: Inputs) -> None:
        if name != "forward":
            self.end_bend(inputs.now, "interrupted", f"state {name}")
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
        tick_dt = 0.0 if self._last_now is None else max(0.0, min(0.2, inputs.now - self._last_now))
        self._last_now = inputs.now
        if self.cfg.localize.enabled:
            inputs = self._localized(inputs)
        self._track_yaw(inputs.yaw)
        self.head_now = head_yaw(inputs.pose) if self.cfg.head.scan else 0.0
        self.hazards.update(inputs.frame, inputs.pose, inputs.pos, self.yaw, inputs.safe, inputs.now)
        self._pos = inputs.pos
        hm = self.cfg.home
        if hm.enabled and inputs.safe and inputs.pos is not None and not self.homing.arrived \
                and not self.homing.abandoned:
            if self.homing.home is None:
                self.homing.home = inputs.pos
            if not self.homing.active and self.stats.forward_seconds >= hm.after_seconds:
                self.homing.active, self.homing.since = True, inputs.now
                self.stats.home_started += 1
        if (self.cfg.explore.enabled or hm.enabled) and inputs.safe:
            self.explorer.update(inputs.pos, inputs.now)
            sx = inputs.pose.sensor_pos[0] if inputs.pose else self.cfg.low.nominal_sensor_forward
            g = geometry(inputs.pose, self.cfg)
            self.explorer.update_walls(inputs.frame, inputs.pos, self.yaw, sx, self.cfg.obstacle,
                                       self.head_now if g is None else 0.0, g)
            self.mark_busy_sides(inputs.now)

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
        head = self._head_target(inputs, tick_dt)
        pitch = self._pitch_target(inputs)
        if head is not None:
            from dataclasses import replace
            cmd = replace(cmd, head_yaw=head, head_pitch=pitch)
        return cmd
