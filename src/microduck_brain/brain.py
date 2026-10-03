"""Mașina de stări a creierului.

Logica de decizie e pură: ``Brain.tick(Inputs) -> Command``. Nu atinge socket-uri și nu
citește ceasul (timpul vine în ``Inputs.now``), deci se testează cu cadre sintetice.

Stări::

    hold ──(sigur + cadre de adâncime, după resume_delay)──> forward
    forward ──(obstacol aproape)──> viră din mers spre partea liberă (tot în ``forward``)
    forward ──(nu mai are loc: sub stop_distance)──> brake ──> turn ──(liber)──> forward
                                                       └─(prea multe frâne fără progres)──> bigturn ──> forward
    orice stare ──(căzut / nesănătos / fără adâncime)──> hold

``forward`` face două lucruri: cât drumul e liber ține direcția cu ``odom.yaw`` (corectează deriva),
iar când un obstacol se apropie (între ``slow_distance`` și ``stop_distance``) virează din mers,
mai tare cu cât e mai aproape. Doar dacă nu mai are loc (sub ``stop_distance``) frânează și se
rotește pe loc. ``turn`` și ``bigturn`` măsoară unghiul din ``odom.yaw``, nu din timp.

**Cum adaugi o stare nouă** (dans, odihnă...): scrie o clasă cu ``name`` și
``step(brain, inputs) -> (Command, next_name | None)``, opțional ``enter``, și
înregistreaz-o cu ``brain.register(MyState())``. Gardianul de siguranță din
``Brain.tick`` rămâne deasupra tuturor stărilor: dacă robotul a căzut, orice stare
e înlocuită cu ``hold``. O stare care *nu* are nevoie de adâncime (dans, odihnă)
setează ``needs_depth = False``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from .config import Config
from .depth import DepthFrame, Pose, View, analyse, low_obstacles


@dataclass(frozen=True)
class Inputs:
    now: float                    # secunde, ceas monoton
    frame: DepthFrame | None      # None = nu există cadru proaspăt
    safe: bool                    # robotul e în picioare și sănătos
    safe_reason: str = ""         # de ce nu e sigur (pentru log)
    yaw: float | None = None      # odom.yaw (rad, + = stânga) sau None dacă robotul nu îl raportează
    pose: Pose | None = None      # poza senzorului din robot.state, pentru geometria podelei
    pos: tuple[float, float] | None = None   # odom.position[0:2] (m), pentru memoria obiectelor joase


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
    avoids: int = 0               # ocoliri începute (viraj din mers și/sau frână + întoarcere)
    brakes: int = 0               # din ele, câte au ajuns la frână + rotire pe loc
    big_turns: int = 0
    low_seen: int = 0             # obiecte joase (cuburi, mingi) noi, văzute în rândurile de jos
    turn_timeouts: int = 0        # întoarceri care nu s-au eliberat în max_turn_time
    holds_unsafe: int = 0         # opriri pentru cădere / nesănătos
    holds_no_depth: int = 0       # opriri pentru lipsa cadrelor de adâncime
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
    """Nu trimite comenzi de mers. Iese doar după ce totul e bine un timp continuu."""

    name = "hold"
    needs_depth = False

    def __init__(self) -> None:
        self._ok_since: float | None = None
        self._stop_sent = False

    def enter(self, brain, inputs):
        self._ok_since = None
        self._stop_sent = False
        brain.heading_integral = 0.0
        brain.forget_low()   # după o cădere poziția odometriei nu mai e de încredere

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
    """0 = drum liber (dincolo de slow_distance), 1 = la stop_distance."""
    o = cfg.obstacle
    if front is None or o.slow_distance <= o.stop_distance:
        return 0.0
    return min(1.0, max(0.0, (o.slow_distance - front) / (o.slow_distance - o.stop_distance)))


class Forward(State):
    """Merge înainte. Drum liber: ține direcția. Obstacol aproape: viră din mers."""

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
            # Nu mai are loc: oprește-te de tot.
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
                brain.consecutive_avoids = 0   # a mers destul: nu e colț
            if not self._straight:
                # Drum liber din nou: direcția de acum devine direcția de ținut.
                self._straight = True
                brain.heading_target = brain.yaw
            target = self._heading_correction(brain, dt)

        # Netezește comanda de rotire: nicio schimbare bruscă.
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
    """Stă pe loc scurt, ca mersul să se oprească înainte de întoarcere."""

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
    """Bază pentru rotirile în loc: măsoară unghiul parcurs din odom.yaw."""

    def enter(self, brain, inputs):
        self.since = inputs.now
        self.yaw0 = brain.yaw

    def turned(self, brain) -> float | None:
        """Unghiul parcurs în direcția rotirii (rad, >= 0 dacă merge bine), sau None fără odometrie."""
        if self.yaw0 is None or brain.yaw is None:
            return None
        return brain.turn_sign * (brain.yaw - self.yaw0)


class Turn(_Rotation):
    """Se întoarce spre partea mai liberă până vede liber în față (și a parcurs unghiul minim)."""

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
            # Nu se mai eliberează: e probabil un colț. Trecem direct la întoarcerea mare.
            brain.stats.turn_timeouts += 1
            brain.consecutive_avoids = 0
            brain.stats.big_turns += 1
            return move(0.0, 0.0), "bigturn"
        return move(0.0, brain.turn_sign * cfg.motion.turn_speed), None


class BigTurn(_Rotation):
    """Întoarcere mare în loc, până la unghiul MĂSURAT (odom.yaw)."""

    name = "bigturn"

    def enter(self, brain, inputs):
        super().enter(brain, inputs)
        c = brain.cfg.corner
        self.target = math.radians(c.big_turn_degrees)
        # Fără odometrie: cronometrat, presupunând că robotul face doar o parte din rata cerută.
        self.fallback = self.target / (brain.cfg.motion.big_turn_speed * c.fallback_turn_efficiency)

    def step(self, brain, inputs):
        elapsed = inputs.now - self.since
        angle = self.turned(brain)
        if angle is not None:
            done = angle >= self.target
        else:
            done = elapsed >= self.fallback
        if not done and elapsed >= brain.cfg.corner.big_turn_max_time:
            brain.stats.turn_timeouts += 1   # unghiul nu a crescut: odometrie blocată sau semn greșit
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
        self.episode_open = False       # ocolirea curentă a fost deja numărată
        self.clear_time = 0.0           # cât a mers fără obstacol, continuu
        self.yaw: float | None = None   # odom.yaw desfășurat (fără salt la ±pi), + = stânga
        self._raw_yaw: float | None = None
        self.heading_target: float | None = None
        self.heading_integral = 0.0
        self._low_first: float | None = None
        self._memory: list[list[float]] = []     # [x, y, t] în cadrul lumii odom
        self.low_items: list[tuple[float, float]] = []   # (înainte, stânga) față de senzor, acum
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

    # --- obiecte joase -----------------------------------------------------------
    def forget_low(self) -> None:
        self._memory.clear()
        self._low_first = None
        self.low_items = []

    def _sensor_forward(self, inputs: Inputs) -> float:
        return inputs.pose.sensor_pos[0] if inputs.pose else self.cfg.low.nominal_sensor_forward

    def _update_low(self, inputs: Inputs) -> None:
        """O dată pe tick: detectează obiecte joase în cadru, confirmă-le în timp, ține minte unde
        sunt (în coordonate odom) și calculează ce e acum în fața robotului. Un cub de 7 cm dispare
        sub cea mai de jos rază când robotul se apropie, de aceea memoria."""
        cfg = self.cfg
        self.low_items = []
        if cfg.obstacle.low_obstacles != "avoid":
            return
        if not inputs.safe:
            self.forget_low()        # căzut: poza și odometria nu mai sunt de încredere
            return
        lo = cfg.low
        now = inputs.now
        can_remember = inputs.pos is not None and self.yaw is not None and lo.memory_seconds > 0
        sx = self._sensor_forward(inputs)

        found = low_obstacles(inputs.frame, inputs.pose, cfg) if inputs.frame is not None else []
        if found:
            if self._low_first is None:
                self._low_first = now
            confirmed = now - self._low_first >= lo.confirm_time
        else:
            self._low_first = None
            confirmed = False

        if confirmed:
            for o in found:
                self.low_items.append((o.forward, o.lateral))
                if can_remember:
                    self._remember(inputs, o.forward + sx, o.lateral, now)

        if can_remember and self._memory:
            c, s_ = math.cos(self.yaw), math.sin(self.yaw)
            keep = []
            for m in self._memory:
                if now - m[2] > lo.memory_seconds:
                    continue
                dx, dy = m[0] - inputs.pos[0], m[1] - inputs.pos[1]
                rx, ry = c * dx + s_ * dy, -s_ * dx + c * dy       # cadrul trunchiului
                fwd = rx - sx
                if fwd < -0.15 or fwd > 2.0:
                    continue                                        # l-am depășit / prea departe
                keep.append(m)
                if confirmed and any(abs(o.forward - fwd) < 0.05 and abs(o.lateral - ry) < 0.05 for o in found):
                    continue                                        # deja în lista curentă
                self.low_items.append((fwd, ry))
            self._memory = keep

    def _remember(self, inputs: Inputs, rx: float, ry: float, now: float) -> None:
        c, s_ = math.cos(self.yaw), math.sin(self.yaw)
        wx = inputs.pos[0] + c * rx - s_ * ry
        wy = inputs.pos[1] + s_ * rx + c * ry
        for m in self._memory:
            if abs(m[0] - wx) < 0.07 and abs(m[1] - wy) < 0.07:
                m[0], m[1], m[2] = wx, wy, now
                return
        self._memory.append([wx, wy, now])
        self.stats.low_seen += 1

    def look(self, inputs: Inputs, threshold: float | None = None) -> View:
        """Vederea din față: pereții din rândurile de sus + obiectele joase (văzute sau ținute minte)."""
        o = self.cfg.obstacle
        view = analyse(inputs.frame, o, threshold)
        if not self.low_items:
            return view
        thr = o.stop_distance if threshold is None else threshold
        hw = self.cfg.low.corridor_half_width
        extra = self.cfg.low.extra_clearance
        front, blocked = view.front, view.blocked
        left_free, right_free = view.left_free, view.right_free
        for f, lat in self.low_items:
            if abs(lat) > hw:
                continue
            f -= extra    # tratat ca mai aproape: reacție mai fermă, fiindcă obiectul se vede doar scurt
            front = f if front is None else min(front, f)
            blocked = blocked or f < thr
            if lat >= 0:
                left_free = min(left_free, f)
            else:
                right_free = min(right_free, f)
        return replace(view, blocked=blocked, front=front, left_free=left_free, right_free=right_free)

    def pick_turn(self, view: View) -> float:
        """+1 = stânga, -1 = dreapta: partea mai liberă; la egalitate, implicit."""
        side = view.freer_side
        if side is None:
            side = self.cfg.obstacle.default_turn
        return 1.0 if side == "left" else -1.0

    def _goto(self, name: str, inputs: Inputs) -> None:
        self.state = self.states[name]
        self.state.enter(self, inputs)

    def tick(self, inputs: Inputs) -> Command:
        # contabilitate de timp
        if self._last_now is not None:
            dt = max(0.0, inputs.now - self._last_now)
            n = self.state.name
            self.stats.state_seconds[n] = self.stats.state_seconds.get(n, 0.0) + dt
            if n == "forward":
                self.stats.forward_seconds += dt
        self._last_now = inputs.now
        self._track_yaw(inputs.yaw)
        self._update_low(inputs)

        if not self._started:
            self._started = True
            self.state.enter(self, inputs)

        # Gardianul: deasupra oricărei stări.
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
