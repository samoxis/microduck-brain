"""Help a robot that stays down.

robotd already gets the robot up after most falls: it goes limp as the robot falls, lands, ramps the
joints back to the standing pose and hands over to the standing policy, which stands it up (in the
full simulator, 9 of 10 shoves; tools/push/push_trials.sh). What it does not do is try again: its
sequence only starts on a robot that is *falling*, never on one already on the floor ("an endless
limp-pose-retry loop" is what that guard prevents), and the standing policy alone does not get a
robot up from lying on its side.

So when the robot has been down for ``after_seconds``, this runs the same sequence by hand, through
the robot's public API: stop the policy (``robot.enable`` off: the robot holds where it is), ramp the
joints to the home pose (``robot.init``, which "works whatever the robot is lying on"), and start the
policy again (``robot.enable`` on), which, with nobody driving, hands the robot to the standing
policy from a still, home-posed robot -- the case robotd's own comment says it is good at. Then it
waits ``verify_seconds``; still down, it tries again, at most ``max_attempts`` times per fall.

Like the brain, this is a pure function of its inputs (time and the fall reason), so it replays.
"""

from __future__ import annotations

FALLEN = "the robot has fallen"
LIMP = "the robot is limp"


class Recovery:
    def __init__(self, cfg) -> None:
        self.cfg = cfg.recover
        self.reset()

    def reset(self) -> None:
        self.down_since: float | None = None
        self.step: int = -1             # -1 idle; 0..2 the three calls; 3 verifying
        self.step_at = 0.0
        self.attempts = 0

    @property
    def active(self) -> bool:
        return self.step >= 0

    def tick(self, now: float, safe: bool, reason: str, stats) -> tuple[str, dict] | None:
        """The call to make now, if any: (method, params)."""
        if not self.cfg.enabled:
            return None
        down = (not safe) and reason in (FALLEN, LIMP)
        if safe:
            if self.active:
                stats.recoveries_ok += 1
            self.reset()
            return None
        if not down and not self.active:
            self.down_since = None          # unhealthy for another reason: not ours to fix
            return None
        if self.down_since is None:
            self.down_since = now
        if self.step == -1:
            if self.attempts >= self.cfg.max_attempts or now - self.down_since < self.cfg.after_seconds:
                return None
            self.attempts += 1
            stats.recovery_attempts += 1
            self.step, self.step_at = 0, now
            return "robot.enable", {"on": False}
        if self.step == 0 and now - self.step_at >= 0.5:
            self.step, self.step_at = 1, now
            return "robot.init", {}
        if self.step == 1 and now - self.step_at >= self.cfg.init_seconds:
            self.step, self.step_at = 2, now
            return "robot.enable", {"on": True}
        if self.step == 2:
            self.step, self.step_at = 3, now
            return None
        if self.step == 3 and now - self.step_at >= self.cfg.verify_seconds:
            self.step = -1                   # still down: wait after_seconds again, then retry
            self.down_since = now
        return None
