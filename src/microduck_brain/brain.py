"""Mașina de stări a creierului.

Logica de decizie e pură: ``Brain.tick(Inputs) -> Command``. Nu atinge socket-uri și nu
citește ceasul (timpul vine în ``Inputs.now``), deci se testează cu cadre sintetice.

Stări v0::

    hold ──(sigur + cadre de adâncime, după resume_delay)──> forward
    forward ──(obstacol)──> brake ──> turn ──(liber)──> forward
                              └─(prea multe ocoliri fără progres)──> bigturn ──> forward
    orice stare ──(căzut / nesănătos / fără adâncime)──> hold

**Cum adaugi o stare nouă** (dans, odihnă...): scrie o clasă cu ``name`` și
``step(brain, inputs) -> (Command, next_name | None)``, opțional ``enter``, și
înregistreaz-o cu ``brain.register(MyState())``. Gardianul de siguranță din
``Brain.tick`` rămâne deasupra tuturor stărilor: dacă robotul a căzut, orice stare
e înlocuită cu ``hold``. O stare care *nu* are nevoie de adâncime (dans, odihnă)
setează ``needs_depth = False``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .config import Config
from .depth import DepthFrame, View, analyse


@dataclass(frozen=True)
class Inputs:
    now: float                    # secunde, ceas monoton
    frame: DepthFrame | None      # None = nu există cadru proaspăt
    safe: bool                    # robotul e în picioare și sănătos
    safe_reason: str = ""         # de ce nu e sigur (pentru log)


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
    avoids: int = 0               # ocoliri începute (frână + întoarcere)
    big_turns: int = 0
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


class Forward(State):
    name = "forward"

    def enter(self, brain, inputs):
        self.since = inputs.now

    def step(self, brain, inputs):
        view = analyse(inputs.frame, brain.cfg.obstacle)
        if not view.blocked:
            return move(brain.cfg.motion.forward_speed, 0.0), None
        # A mers destul înainte de obstacolul ăsta? Atunci nu e colț, numărătoarea reîncepe.
        if inputs.now - self.since >= brain.cfg.corner.progress_time:
            brain.consecutive_avoids = 0
        brain.consecutive_avoids += 1
        brain.stats.avoids += 1
        return STOP, "brake"


class Brake(State):
    """Stă pe loc scurt, ca mersul să se oprească înainte de întoarcere."""

    name = "brake"

    def enter(self, brain, inputs):
        self.since = inputs.now

    def step(self, brain, inputs):
        if inputs.now - self.since < brain.cfg.obstacle.stop_pause:
            return move(0.0, 0.0), None
        view = analyse(inputs.frame, brain.cfg.obstacle)
        brain.turn_sign = brain.pick_turn(view)
        if brain.consecutive_avoids >= brain.cfg.corner.turns_before_big_turn:
            brain.consecutive_avoids = 0
            brain.stats.big_turns += 1
            return move(0.0, 0.0), "bigturn"
        return move(0.0, 0.0), "turn"


class Turn(State):
    """Se întoarce spre partea mai liberă până vede liber în față."""

    name = "turn"

    def enter(self, brain, inputs):
        self.since = inputs.now

    def step(self, brain, inputs):
        cfg = brain.cfg
        elapsed = inputs.now - self.since
        if elapsed >= cfg.obstacle.min_turn_time:
            view = analyse(inputs.frame, cfg.obstacle, cfg.obstacle.clear_distance)
            if not view.blocked:
                return move(0.0, 0.0), "forward"
        if elapsed >= cfg.obstacle.max_turn_time:
            # Nu se mai eliberează: e probabil un colț. Trecem direct la întoarcerea mare.
            brain.stats.turn_timeouts += 1
            brain.consecutive_avoids = 0
            brain.stats.big_turns += 1
            return move(0.0, 0.0), "bigturn"
        return move(0.0, brain.turn_sign * cfg.motion.turn_speed), None


class BigTurn(State):
    """Întoarcere mare, cronometrată, în loc."""

    name = "bigturn"

    def enter(self, brain, inputs):
        m = brain.cfg.motion
        self.since = inputs.now
        self.duration = math.radians(brain.cfg.corner.big_turn_degrees) / m.big_turn_speed

    def step(self, brain, inputs):
        if inputs.now - self.since >= self.duration:
            return move(0.0, 0.0), "forward"
        return move(0.0, brain.turn_sign * brain.cfg.motion.big_turn_speed), None


class Brain:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.stats = Stats()
        self.consecutive_avoids = 0
        self.turn_sign = 1.0 if cfg.obstacle.default_turn == "left" else -1.0
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
