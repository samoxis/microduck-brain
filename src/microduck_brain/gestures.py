"""A hand gesture read from the depth sensor alone: held close in front of the head, then taken away.

The sensor (on the head) is 8x8 zones over 45°. A palm 10 cm in front covers nearly all of its wall
rows; nothing in the room ever comes that close while the robot walks normally (it stops at
``stop_distance``, 0.28 m, and turns until 0.40 m is clear). So:

    near      = at least ``min_fraction`` of the wall-row zones read a valid distance under ``near``
    gesture   = near for ``hold_seconds`` .. ``max_seconds`` without a break, then not near for
                ``release_seconds``

It fires on the release, so the hand is gone by the time the robot acts on it (a "go" with the hand
still in front would only make it brake for the hand). Longer than ``max_seconds`` is not a gesture:
a wall it is pressed against, or the sensor covered. Nor is anything seen while the robot turns in
place: in 8.6 h of full-simulator recordings, the only two "gestures" were furniture swept past the
head 1-9 cm away during a turn. (While a hand is there the brain waits in "brake" instead of turning
away from it.) A function of the frames and the brain's state, so it replays.
"""

from __future__ import annotations

VALID = (5, 9)    # VL53L5CX target status: range valid


class HandGesture:
    def __init__(self, cfg) -> None:
        self.g = cfg.gestures
        self.rows = range(cfg.obstacle.first_row, cfg.obstacle.last_row + 1)
        self.reset()

    def reset(self) -> None:
        self._since: float | None = None     # near since
        self._last_near: float | None = None
        self._held = 0.0                     # how long the last presence lasted
        self._spoiled = False                # the robot turned in place during this presence

    def near(self, frame) -> bool:
        if frame is None:
            return False
        n = close = 0
        for r in self.rows:
            for c in range(frame.cols):
                i = r * frame.cols + c
                n += 1
                if frame.status[i] in VALID and frame.distance_mm[i] < self.g.near * 1000.0:
                    close += 1
        return n > 0 and close >= self.g.min_fraction * n

    @property
    def present(self) -> bool:
        """Something hand-like is in front now (or was, within release_seconds)."""
        return self._since is not None

    def update(self, frame, now: float, turning: bool = False) -> bool:
        """True once, at the release of a gesture. ``turning``: the robot is turning in place."""
        if frame is None:
            return False
        if self.near(frame):
            if self._since is None:
                self._since, self._spoiled = now, False
            self._spoiled = self._spoiled or turning
            self._last_near = now
            self._held = now - self._since
            return False
        if self._since is None:
            return False
        if now - self._last_near < self.g.release_seconds:
            return False                     # a dropped frame or a flicker: not taken away yet
        held, self._since = self._held, None
        return not self._spoiled and self.g.hold_seconds <= held <= self.g.max_seconds
