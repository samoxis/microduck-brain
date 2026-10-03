"""Teste pentru logica de decizie, cu cadre de adâncime sintetice.

Convenții: rândul 0 sus, coloana 0 = stânga robotului (ca în simulator și în ``tofd``).
"""

from __future__ import annotations

import math

import pytest

from microduck_brain.brain import Brain, Inputs
from microduck_brain.config import Config, from_dict
from microduck_brain.depth import STATUS_NO_TARGET, DepthFrame, analyse

FAR = 3.0


def frame(fn=None, fill=FAR) -> DepthFrame:
    """Cadru 8x8; ``fn(row, col)`` întoarce metri, ``None`` = măsurătoare eșuată,
    ``"free"`` = nimic în rază."""
    dist, status = [], []
    for r in range(8):
        for c in range(8):
            v = fill if fn is None else fn(r, c)
            if v is None:
                dist.append(0), status.append(12)
            elif v == "free":
                dist.append(0), status.append(STATUS_NO_TARGET)
            else:
                dist.append(round(v * 1000)), status.append(5)
    return DepthFrame(8, 8, tuple(dist), tuple(status))


# --- cadrele sintetice cerute ---------------------------------------------
def clear():
    return frame()


def wall_ahead(d=0.20):
    return frame(lambda r, c: d if r <= 4 else FAR)


def obstacle_left(d=0.20):
    # obstacol pe coloanele 0-3 (stânga); dreapta liberă
    return frame(lambda r, c: d if c <= 3 else FAR)


def obstacle_right(d=0.20):
    return frame(lambda r, c: d if c >= 4 else FAR)


def corner():
    # perete în față, ambele părți la fel de apropiate
    return frame(lambda r, c: 0.20)


class Sim:
    """Rulează Brain-ul cu un ceas fals, 20 Hz."""

    def __init__(self, cfg: Config | None = None, safe=True):
        self.cfg = cfg or Config()
        self.brain = Brain(self.cfg)
        self.t = 0.0
        self.safe = safe
        self.frame = clear()
        self.log = []

    def run(self, seconds: float, frame_=None, safe=None, dt=0.05):
        if frame_ is not None:
            self.frame = frame_
        if safe is not None:
            self.safe = safe
        out = None
        end = self.t + seconds
        while self.t < end - 1e-9:
            out = self.brain.tick(Inputs(self.t, self.frame, self.safe, "" if self.safe else "căzut"))
            self.log.append((self.t, self.brain.state_name, out))
            self.t += dt
        return out

    def up(self):
        """Pornește și așteaptă să iasă din hold."""
        self.run(self.cfg.safety.resume_delay + 0.3)
        assert self.brain.state_name == "forward"


# --- analiza cadrului ------------------------------------------------------
def test_clear_is_not_blocked():
    v = analyse(clear(), Config().obstacle)
    assert not v.blocked and not v.blind


def test_wall_is_blocked():
    assert analyse(wall_ahead(), Config().obstacle).blocked


def test_single_noisy_zone_is_ignored():
    f = frame(lambda r, c: 0.1 if (r, c) == (2, 3) else FAR)
    assert not analyse(f, Config().obstacle).blocked


def test_floor_rows_are_ignored():
    f = frame(lambda r, c: 0.1 if r >= 5 else FAR)
    assert not analyse(f, Config().obstacle).blocked


def test_no_target_counts_as_free_space():
    assert not analyse(frame(lambda r, c: "free"), Config().obstacle).blocked


def test_all_failed_measurements_is_blind_and_blocked():
    v = analyse(frame(lambda r, c: None), Config().obstacle)
    assert v.blind and v.blocked


def test_failed_zones_do_not_count_as_obstacles():
    f = frame(lambda r, c: None if c % 2 else FAR)
    v = analyse(f, Config().obstacle)
    assert not v.blocked


def test_freer_side():
    assert analyse(obstacle_left(), Config().obstacle).freer_side == "right"
    assert analyse(obstacle_right(), Config().obstacle).freer_side == "left"
    assert analyse(corner(), Config().obstacle).freer_side is None


# --- comportament ------------------------------------------------------------
def test_starts_held_then_walks_forward_slowly():
    s = Sim()
    first = s.run(0.05)
    assert s.brain.state_name == "hold"
    assert first.kind in ("stop", "none")
    s.up()
    cmd = s.run(0.2)
    assert cmd.kind == "move"
    assert cmd.vx == pytest.approx(s.cfg.motion.forward_speed)
    assert cmd.vyaw == 0.0


def test_free_path_keeps_going():
    s = Sim()
    s.up()
    s.run(10.0, clear())
    assert s.brain.state_name == "forward"
    assert s.brain.stats.avoids == 0


def test_wall_stops_then_turns_then_continues():
    s = Sim()
    s.up()
    s.run(0.1, wall_ahead())
    assert s.brain.state_name in ("brake", "turn")
    cmd = s.run(0.1)
    assert cmd.kind == "move" and cmd.vx == 0.0  # stă pe loc, nu merge
    s.run(1.0)  # frâna trece, începe întoarcerea
    assert s.brain.state_name == "turn"
    cmd = s.run(0.1)
    assert cmd.vx == 0.0 and abs(cmd.vyaw) == pytest.approx(s.cfg.motion.turn_speed)
    # peretele dispare => se reia mersul
    s.run(1.0, clear())
    assert s.brain.state_name == "forward"
    assert s.brain.stats.avoids == 1
    assert s.run(0.1).vx > 0


def test_never_moves_forward_while_blocked():
    s = Sim()
    s.up()
    s.log.clear()
    s.run(3.0, wall_ahead())
    assert all(c.vx == 0.0 for _, _, c in s.log if c.kind == "move")


def test_obstacle_on_left_turns_right():
    s = Sim()
    s.up()
    s.run(2.0, obstacle_left())
    assert s.brain.state_name == "turn"
    assert s.run(0.1).vyaw < 0  # vyaw negativ = dreapta


def test_obstacle_on_right_turns_left():
    s = Sim()
    s.up()
    s.run(2.0, obstacle_right())
    assert s.brain.state_name == "turn"
    assert s.run(0.1).vyaw > 0  # vyaw pozitiv = stânga


def test_symmetric_wall_uses_default_turn():
    cfg = from_dict({"obstacle": {"default_turn": "right"}})
    s = Sim(cfg)
    s.up()
    s.run(2.0, corner())
    assert s.run(0.1).vyaw < 0


def test_turn_waits_for_clear_distance_not_just_stop_distance():
    s = Sim()
    s.up()
    s.run(2.0, wall_ahead(0.2))
    assert s.brain.state_name == "turn"
    # 0.33 m: peste stop_distance (0.28) dar sub clear_distance (0.40) => continuă să se întoarcă
    s.run(1.0, frame(lambda r, c: 0.33))
    assert s.brain.state_name == "turn"
    s.run(0.5, clear())
    assert s.brain.state_name == "forward"


def test_corner_triggers_big_turn_after_repeated_avoids():
    s = Sim()
    s.up()
    # perete mereu în față: nu înaintează niciodată între ocoliri
    # (după fiecare întoarcere vede liber o clipă, apoi iar perete)
    for _ in range(s.cfg.corner.turns_before_big_turn):
        s.run(0.2, wall_ahead())            # blocat => frână
        s.run(1.0)                          # frână + început de întoarcere
        s.run(0.7, clear())                 # întoarcerea găsește liber, reia mersul scurt
        s.run(0.1, wall_ahead())            # dar iar perete imediat
        if s.brain.state_name == "bigturn":
            break
    s.run(1.0, wall_ahead())
    assert s.brain.stats.big_turns == 1
    assert s.brain.state_name == "bigturn"
    expected = math.radians(s.cfg.corner.big_turn_degrees) / s.cfg.motion.big_turn_speed
    t0 = s.t
    s.run(expected + 0.3, wall_ahead())
    assert s.brain.state_name != "bigturn"
    assert s.t - t0 > expected
    # cât a durat, nu a mers înainte
    assert all(c.vx == 0.0 for _, n, c in s.log if n == "bigturn" and c.kind == "move")


def test_progress_resets_the_corner_counter():
    s = Sim()
    s.up()
    for _ in range(5):  # mult mai multe decât turns_before_big_turn
        s.run(0.2, wall_ahead())
        s.run(1.0)
        s.run(0.6, clear())
        s.run(s.cfg.corner.progress_time + 0.5, clear())  # a mers destul => nu e colț
    assert s.brain.stats.big_turns == 0
    assert s.brain.stats.avoids == 5


def test_turn_that_never_clears_becomes_big_turn():
    s = Sim()
    s.up()
    s.run(s.cfg.obstacle.max_turn_time + 3.0, corner())
    assert s.brain.stats.turn_timeouts >= 1
    assert s.brain.stats.big_turns >= 1


# --- siguranță ---------------------------------------------------------------
def test_fall_stops_and_sends_no_walk_commands_until_upright():
    s = Sim()
    s.up()
    s.log.clear()
    cmd = s.run(0.1, safe=False)
    assert s.brain.state_name == "hold"
    s.run(5.0)
    kinds = [c.kind for _, _, c in s.log]
    assert "move" not in kinds, "nu trebuie trimisă nicio comandă de mișcare cât e căzut"
    assert kinds.count("stop") == 1, "stop o singură dată, nu în buclă"
    assert s.brain.stats.holds_unsafe == 1


def test_resumes_only_after_resume_delay():
    s = Sim()
    s.up()
    s.run(1.0, safe=False)
    s.log.clear()
    s.run(s.cfg.safety.resume_delay - 0.3, safe=True)
    assert all(c.kind != "move" for _, _, c in s.log)
    s.run(0.6)
    assert s.brain.state_name == "forward"


def test_flapping_health_restarts_the_delay():
    s = Sim()
    s.up()
    s.run(0.5, safe=False)
    s.run(1.0, safe=True)
    s.run(0.1, safe=False)       # iar nesigur înainte de resume_delay
    s.run(1.0, safe=True)
    assert s.brain.state_name == "hold"


def test_fall_during_turn_also_holds():
    s = Sim()
    s.up()
    s.run(1.5, wall_ahead())
    assert s.brain.state_name == "turn"
    s.log.clear()
    s.run(2.0, safe=False)
    assert s.brain.state_name == "hold"
    assert all(c.kind != "move" for _, _, c in s.log)


def test_missing_depth_frames_stop_the_robot():
    s = Sim()
    s.up()
    s.log.clear()
    s.frame = None
    s.run(1.0)
    assert s.brain.state_name == "hold"
    assert all(c.kind != "move" for _, _, c in s.log)
    assert s.brain.stats.holds_no_depth == 1


# --- configurație -------------------------------------------------------------
def test_unknown_config_key_is_rejected():
    with pytest.raises(ValueError, match="necunoscut"):
        from_dict({"motion": {"forward_sped": 0.1}})


def test_bad_config_values_are_rejected():
    with pytest.raises(ValueError):
        from_dict({"obstacle": {"stop_distance": 0.5, "clear_distance": 0.3}})
    with pytest.raises(ValueError):
        from_dict({"motion": {"forward_speed": "repede"}})


def test_shipped_brain_toml_equals_defaults():
    import tomllib
    from pathlib import Path

    data = tomllib.loads((Path(__file__).parent.parent / "brain.toml").read_text())
    assert from_dict(data) == Config()


def test_speed_comes_from_config():
    s = Sim(from_dict({"motion": {"forward_speed": 0.2}}))
    s.up()
    assert s.run(0.2).vx == pytest.approx(0.2)
