"""Teste v2: obiecte joase (cuburi de 7 cm, mingi de 8 cm) în rândurile de jos ale cadrului.

Trei surse de cadre:
* ``tests/fixtures/mujoco_low.json`` — cadre REALE din senzorul simulat (MuJoCo), cu poza raportată;
* ``tests/lowgeo.py`` — un ray-caster scris aici, verificat să dea aceleași zone ca MuJoCo;
* o lume simplă în buclă închisă (robotul merge, cubul rămâne pe loc).

Distanțele ``x`` ale obiectelor sunt de la ORIGINEA TRUNCHIULUI (senzorul e cu 6.4 cm mai în față).
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest

from microduck_brain.brain import Brain, Inputs
from microduck_brain.config import Config, from_dict
from microduck_brain.depth import DepthFrame, Pose, floor_model, low_obstacles
from microduck_brain.hazards import LOW

from .lowgeo import SENSOR_X, pose_for, render
from .test_brain import clear

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "mujoco_low.json").read_text())


def fx_frame(f) -> DepthFrame:
    return DepthFrame(8, 8, tuple(f["distance_mm"]), tuple(f["status"]))


def fx_pose(f) -> Pose:
    p = f["pose"]
    return Pose(tuple(p["sensor_pos"]), tuple(p["sensor_quat"]), tuple(p["gravity"]), p["trunk_z"])


def where(shape, x=None, pitch=None, lateral=None):
    return [
        f for f in FIXTURES
        if f["shape"] == shape
        and (x is None or f["x"] == x)
        and (pitch is None or f["pitch_deg"] == pitch)
        and (lateral is None or f["lateral"] == lateral)
    ]


CFG = Config()


# ── modelul podelei ──────────────────────────────────────────────────────────────
def test_floor_model_matches_mujoco_within_noise():
    rel = []
    for f in where("none"):
        m = floor_model(fx_pose(f), CFG.low)
        for i in range(40, 64):
            if f["status"][i] == 5 and m.expected(i) < 9:
                rel.append(f["distance_mm"][i] / 1000 / m.expected(i) - 1)
    assert len(rel) > 1000
    assert max(abs(v) for v in rel) < 0.04      # zgomotul senzorului, nu eroare de model


def test_nominal_floor_matches_the_measured_standing_pose():
    m = floor_model(None, CFG.low)
    # măsurat în MuJoCo (STAND): rândurile 5, 6, 7 -> 1.68, 1.01, 0.73 m
    assert [round(m.expected(r * 8 + 3), 2) for r in (5, 6, 7)] == [1.68, 1.01, 0.73]


def test_my_raycaster_agrees_with_mujoco_on_the_cube():
    f = where("cube", x=0.6, pitch=0.0, lateral=0.0)[0]
    mine = render(("cube", 0.6, 0.07))
    diffs = [abs(a - b) for a, b in zip(mine.distance_mm, f["distance_mm"])]
    assert sum(1 for d in diffs if d > 60) == 0     # zgomotul MuJoCo e de ordinul cm


# ── zero alarme false: podea goală, trunchi înclinat ──────────────────────────────
def test_empty_floor_mujoco_frames_tilted_never_alarm():
    floor = where("none")
    assert len(floor) >= 80
    for f in floor:
        assert low_obstacles(fx_frame(f), fx_pose(f), CFG) == [], (f["pitch_deg"], f["roll_deg"])
        assert low_obstacles(fx_frame(f), None, CFG) == [], ("fără poză", f["pitch_deg"], f["roll_deg"])


@pytest.mark.parametrize("pitch", [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
@pytest.mark.parametrize("roll", [-3.0, 0.0, 3.0])
def test_empty_floor_synthetic_tilted_never_alarm(pitch, roll):
    rng = random.Random(1234)
    for _ in range(25):
        frame = render(noise=0.02, rng=rng, pitch_deg=pitch, roll_deg=roll)
        assert low_obstacles(frame, pose_for(pitch, roll), CFG) == []
        assert low_obstacles(frame, None, CFG) == []


def test_empty_floor_pose_compensation_is_what_allows_the_tight_margin():
    """Dovada că poza contează: aceeași toleranță strânsă FĂRĂ poză ar da alarme false la ±3°."""
    tight = from_dict({"low": {"tilt_tolerance_nopose_deg": 1.5}})
    frames = [render(pitch_deg=3.0), render(pitch_deg=-3.0)]
    assert any(low_obstacles(f, None, tight) for f in frames)
    assert not any(low_obstacles(f, pose_for(p), tight) for f, p in zip(frames, (3.0, -3.0)))


def test_walls_and_far_floor_in_low_rows_are_not_low_obstacles():
    frame = DepthFrame(8, 8, tuple([0] * 64), tuple([255] * 64))     # nimic în rază
    assert low_obstacles(frame, None, CFG) == []


# ── cubul de 7 cm și mingea de 8 cm ───────────────────────────────────────────────
@pytest.mark.parametrize("shape,size", [("cube", 0.07), ("ball", 0.08)])
@pytest.mark.parametrize("x", [0.55, 0.6, 0.65, 0.7])
def test_object_ahead_is_detected_on_real_mujoco_frames(shape, size, x):
    got = 0
    frames = where(shape, x=x, pitch=0.0)
    for f in frames:
        found = low_obstacles(fx_frame(f), fx_pose(f), CFG)
        if found:
            got += 1
            o = min(found, key=lambda o: o.forward)
            assert o.ratio < 0.95
            assert abs(o.lateral - f["lateral"]) < 0.08, "obiectul trebuie localizat lateral"
    # o minge de 8 cm la 0.7 m poate cădea ÎNTRE raze (zonele sunt la ~7 cm una de alta acolo):
    # din cele 3 poziții laterale cel puțin 2 trebuie să fie văzute
    assert got >= 2, f"{shape} la {x}: văzut în {got}/3"


@pytest.mark.parametrize("pitch", [-3.0, 3.0])
def test_object_is_still_detected_when_the_body_sways(pitch):
    for shape, x in (("cube", 0.65), ("ball", 0.65)):
        hits = [low_obstacles(fx_frame(f), fx_pose(f), CFG) for f in where(shape, x=x, pitch=pitch)]
        assert sum(1 for h in hits if h) >= 2, (shape, pitch)


@pytest.mark.parametrize("x", [0.2, 0.3, 0.4])
@pytest.mark.parametrize("shape,size", [("cube", 0.07), ("ball", 0.08)])
def test_object_closer_than_the_lowest_beam_is_physically_invisible(shape, size, x):
    """Limita geometrică: cu capul drept, senzorul e la 24.6 cm și cea mai de jos rază coboară 19.7°.
    Un cub de 7 cm la 20-40 cm de trunchi e SUB acea rază: cadrul e identic cu podea goală. Nicio
    metodă bazată doar pe cadru nu îl poate vedea; de aceea există memoria."""
    floor = where("none", pitch=0.0)[0]
    for f in where(shape, x=x, pitch=0.0):
        close = [abs(a - b) for a, b in zip(f["distance_mm"][40:], floor["distance_mm"][40:])]
        assert max(close) < 120      # doar zgomot (seed diferit), nicio zonă scurtată de obiect
        assert low_obstacles(fx_frame(f), fx_pose(f), CFG) == []
    # și ray-casterul meu confirmă: același cadru ca podeaua
    mine, bare = render((shape, x, size)), render()
    assert mine.distance_mm == bare.distance_mm


def test_ignore_is_a_config_switch_not_a_geometry_change():
    f = where("cube", x=0.6, pitch=0.0, lateral=0.0)[0]
    assert low_obstacles(fx_frame(f), fx_pose(f), CFG)              # detectorul vede obiectul...
    b = Brain(from_dict({"obstacle": {"low_obstacles": "ignore"}}))  # ...dar în "ignore" creierul nu-l folosește
    b.tick(Inputs(0.0, fx_frame(f), True, "", 0.0, fx_pose(f), (0.0, 0.0)))
    assert b.low_items == []


# ── comportament în Brain ─────────────────────────────────────────────────────────
class World:
    """Robot într-o lume cu un singur obiect fix; robotul rulează Brain-ul.

    Rata reală = ``eff_v`` x viteza cerută și ``eff_w`` x rotirea cerută (valori măsurate de tine:
    0.24 cerut ≈ 0.10 m/s, rotire cam la jumătate)."""

    def __init__(self, cfg, obj, eff_v=0.42, eff_w=0.5, shape="cube", size=0.07):
        self.cfg, self.brain = cfg, Brain(cfg)
        self.x = self.y = self.yaw = 0.0
        self.obj, self.shape, self.size = obj, shape, size       # obj = (x, y) centrul, lume
        self.eff_v, self.eff_w = eff_v, eff_w
        self.t = 0.0
        self.min_clearance = math.inf    # cel mai mic spațiu între amprenta robotului și obiect (< 0 = contact)
        self.log = []

    def rel(self):
        dx, dy = self.obj[0] - self.x, self.obj[1] - self.y
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return c * dx + s * dy, -s * dx + c * dy       # înainte, stânga (cadrul trunchiului)

    # amprenta presupusă a robotului în cadrul trunchiului: 20 cm lungime, 20 cm lățime (centrat pe origine)
    FOOT_X, FOOT_Y = 0.10, 0.10

    def clearance(self, fx, fy):
        """Distanța de la marginea obiectului la amprenta robotului (negativ = se ating)."""
        dx = max(abs(fx) - self.FOOT_X, 0.0)
        dy = max(abs(fy) - self.FOOT_Y, 0.0)
        return math.hypot(dx, dy) - self.size / 2

    def frame(self):
        fx, fy = self.rel()
        face = fx - self.size / 2
        if face < 0.02 or fx > 3.0 or abs(fy) > 0.5:
            return render()
        return render((self.shape, face, self.size, fy - 0.0224 + 0.0224))

    def run(self, seconds, dt=0.05):
        end = self.t + seconds
        cmd = None
        while self.t < end - 1e-9:
            fx, fy = self.rel()
            self.min_clearance = min(self.min_clearance, self.clearance(fx, fy))
            pose = pose_for()
            cmd = self.brain.tick(Inputs(self.t, self.frame(), True, "", self.yaw, pose, (self.x, self.y)))
            self.log.append((self.t, self.brain.state_name, cmd))
            if cmd.kind == "move":
                self.yaw += self.eff_w * cmd.vyaw * dt
                v = self.eff_v * cmd.vx * dt
                self.x += v * math.cos(self.yaw)
                self.y += v * math.sin(self.yaw)
            self.t += dt
        return cmd


def start_world(cfg, obj):
    w = World(cfg, obj)
    w.run(cfg.safety.resume_delay + 0.3)           # ieșire din hold
    return w


def test_avoid_steers_around_a_cube_it_cannot_see_at_the_end():
    """Cub la 1.3 m drept în față. Se vede doar la 0.45-0.7 m, apoi dispare sub raza de jos; fără
    memorie robotul l-ar împinge. Cu 'avoid' trebuie să-l ocolească fără contact."""
    w = start_world(Config(), (1.3, 0.0))
    w.run(60.0)
    assert w.brain.stats.low_seen >= 1
    assert w.min_clearance > 0.03, f"cub atins sau la limită: spațiu {w.min_clearance:.3f} m"
    assert w.brain.stats.brakes == 0, "trebuia să vireze din mers, nu să frâneze"


def test_ignore_pushes_the_cube_proving_the_test_world_is_meaningful():
    cfg = from_dict({"obstacle": {"low_obstacles": "ignore"}})
    w = start_world(cfg, (1.3, 0.0))
    w.run(60.0)
    assert w.brain.stats.low_seen == 0
    assert w.min_clearance <= 0.0, "fără detecție, robotul trebuie să ajungă la cub"


@pytest.mark.parametrize("lateral", [-0.08, 0.0, 0.08])
@pytest.mark.parametrize("shape,size", [("cube", 0.07), ("ball", 0.08)])
def test_avoid_clears_objects_slightly_off_axis(shape, size, lateral):
    w = World(Config(), (1.3, lateral), shape=shape, size=size)
    w.run(Config().safety.resume_delay + 0.3)
    w.run(60.0)
    assert w.min_clearance > 0.02, f"{shape} la {lateral}: spațiu {w.min_clearance:.3f} m"


def test_extra_clearance_makes_the_reaction_firm_enough_for_slow_turning():
    """Fără extra_clearance, cu o rotire reală lentă (30% din cea cerută), robotul abia scapă de cub."""
    def run(extra):
        w = World(from_dict({"low": {"extra_clearance": extra}}), (1.3, 0.0), eff_w=0.3)
        w.run(Config().safety.resume_delay + 0.3)
        w.run(60.0)
        return w.min_clearance
    assert run(0.0) < 0.01
    assert run(0.15) > 0.03


def test_memory_saves_the_robot_when_it_turns_slowly_and_walks_fast():
    """Singurul scenariu din 24 încercate unde memoria face diferența (rotire reală 25%, mers 60%)."""
    def run(memory):
        w = World(from_dict({"low": {"memory_seconds": memory}}), (1.3, 0.0), eff_w=0.25, eff_v=0.6)
        w.run(Config().safety.resume_delay + 0.3)
        w.run(40.0)
        return w.min_clearance
    assert run(0.0) < 0.0
    assert run(12.0) > 0.0


def test_object_remembered_after_it_leaves_the_field_of_view():
    w = start_world(Config(), (1.3, 0.0))
    seen_visible = seen_memory = False
    for _ in range(int(25 / 0.05)):
        w.run(0.05)
        fx, fy = w.rel()
        visible = bool(low_obstacles(w.frame(), pose_for(), w.cfg))
        if w.brain.low_items and visible:
            seen_visible = True
        if w.brain.low_items and not visible and 0.0 < fx - 0.035 < 0.45:
            seen_memory = True
    assert seen_visible and seen_memory, "trebuie să fie văzut întâi și ținut minte după aceea"


def test_memory_expires():
    cfg = from_dict({"low": {"memory_seconds": 2.0}})
    b = Brain(cfg)
    obstacle = render(("cube", 0.65, 0.07))
    t = 0.0
    while t < 1.0:   # văzut
        b.tick(Inputs(t, obstacle, True, "", 0.0, pose_for(), (0.0, 0.0)))
        t += 0.05
    assert b.low_items
    while t < 1.0 + 1.0:   # dispare din cadru, încă ținut minte
        b.tick(Inputs(t, render(), True, "", 0.0, pose_for(), (0.0, 0.0)))
        t += 0.05
    assert b.low_items
    while t < 4.5:
        b.tick(Inputs(t, render(), True, "", 0.0, pose_for(), (0.0, 0.0)))
        t += 0.05
    assert b.low_items == []


def test_memory_is_dropped_once_the_object_is_behind_the_robot():
    b = Brain(Config())
    obstacle = render(("cube", 0.65, 0.07))
    t = 0.0
    while t < 1.0:
        b.tick(Inputs(t, obstacle, True, "", 0.0, pose_for(), (0.0, 0.0)))
        t += 0.05
    x = 0.0
    while x < 1.2:
        x += 0.01
        b.tick(Inputs(t, render(), True, "", 0.0, pose_for(), (x, 0.0)))
        t += 0.05
    assert b.low_items == []


def test_memory_cleared_after_a_fall():
    b = Brain(Config())
    t = 0.0
    obstacle = render(("cube", 0.65, 0.07))
    while t < 2.0:                                   # iese din hold
        b.tick(Inputs(t, clear(), True, "", 0.0, pose_for(), (0.0, 0.0)))
        t += 0.05
    assert b.state_name == "forward"
    while t < 3.0:
        b.tick(Inputs(t, obstacle, True, "", 0.0, pose_for(), (0.0, 0.0)))
        t += 0.05
    assert b.hazards.remembered(LOW)
    b.tick(Inputs(t, obstacle, False, "căzut", 0.0, pose_for(), (0.0, 0.0)))
    assert b.state_name == "hold" and b.hazards.remembered(LOW) == 0 and b.low_items == []
    for _ in range(40):                              # cât e căzut nu se adună nimic nou
        t += 0.05
        b.tick(Inputs(t, obstacle, False, "căzut", 0.0, pose_for(), (0.0, 0.0)))
    assert b.hazards.remembered(LOW) == 0


def test_no_memory_without_position_but_visible_objects_still_count():
    b = Brain(Config())
    t = 0.0
    while t < 1.0:
        b.tick(Inputs(t, render(("cube", 0.65, 0.07)), True, "", 0.0, pose_for(), None))
        t += 0.05
    assert b.low_items and b.hazards.remembered(LOW) == 0


def test_memory_can_be_switched_off():
    cfg = from_dict({"low": {"memory_seconds": 0.0}})
    b = Brain(cfg)
    t = 0.0
    while t < 1.0:
        b.tick(Inputs(t, render(("cube", 0.65, 0.07)), True, "", 0.0, pose_for(), (0.0, 0.0)))
        t += 0.05
    assert b.low_items
    b.tick(Inputs(t, render(), True, "", 0.0, pose_for(), (0.0, 0.0)))
    assert b.low_items == []


def test_a_one_frame_blip_does_not_steer():
    b = Brain(Config())
    t = 0.0
    while t < 2.0:   # hold -> forward
        b.tick(Inputs(t, clear(), True, "", 0.0, pose_for(), (0.0, 0.0)))
        t += 0.05
    b.tick(Inputs(t, render(("cube", 0.65, 0.07)), True, "", 0.0, pose_for(), (0.0, 0.0)))
    t += 0.05
    cmd = b.tick(Inputs(t, render(), True, "", 0.0, pose_for(), (0.0, 0.0)))
    assert b.low_items == [] and cmd.vyaw == 0.0


def test_avoid_vs_ignore_on_the_same_frame():
    def run(mode):
        b = Brain(from_dict({"obstacle": {"low_obstacles": mode}}))
        t, cmd = 0.0, None
        while t < 2.0:
            b.tick(Inputs(t, clear(), True, "", 0.0, pose_for(), (0.0, 0.0)))
            t += 0.05
        while t < 3.5:
            cmd = b.tick(Inputs(t, render(("cube", 0.6, 0.07, 0.0)), True, "", 0.0, pose_for(), (0.0, 0.0)))
            t += 0.05
        return cmd

    assert abs(run("avoid").vyaw) > 0.05
    assert run("ignore").vyaw == 0.0


def test_low_obstacle_on_the_left_steers_right_and_vice_versa():
    for lateral, sign in ((0.06, -1), (-0.06, +1)):
        b = Brain(Config())
        t, cmd = 0.0, None
        while t < 2.0:
            b.tick(Inputs(t, clear(), True, "", 0.0, pose_for(), (0.0, 0.0)))
            t += 0.05
        while t < 4.0:
            cmd = b.tick(Inputs(t, render(("cube", 0.58, 0.07, lateral)), True, "", 0.0, pose_for(), (0.0, 0.0)))
            t += 0.05
        assert cmd.vyaw * sign > 0, (lateral, cmd)


# ── configurare ───────────────────────────────────────────────────────────────────
def test_low_obstacles_switch_validation():
    assert from_dict({"obstacle": {"low_obstacles": "ignore"}}).obstacle.low_obstacles == "ignore"
    assert Config().obstacle.low_obstacles == "avoid"
    with pytest.raises(ValueError, match="low_obstacles"):
        from_dict({"obstacle": {"low_obstacles": "push"}})
    with pytest.raises(ValueError):
        from_dict({"low": {"range_margin": 0.9}})
    with pytest.raises(ValueError):
        from_dict({"low": {"tilt_tolerance_deg": 5.0, "tilt_tolerance_nopose_deg": 2.0}})


# ── poza din robot.state ───────────────────────────────────────────────────────────
def state(pos=(0.064, 0.022, 0.126), z=0.12, grav=(0, 0, -1)):
    return {
        "frames": {"tof": {"pos": list(pos), "quat": [1, 0, 0, 0]}},
        "safety": {"gravity": list(grav)},
        "odom": {"position": [1.0, 2.0, z], "yaw": 0.3},
    }


def test_pose_parsed_from_robot_state():
    p = Pose.from_state(state())
    assert p and p.trunk_z == 0.12 and p.sensor_pos[0] == 0.064
    m = floor_model(p, CFG.low)
    assert m.height == pytest.approx(0.246, abs=1e-6)


@pytest.mark.parametrize("broken", [{}, {"frames": {}}, {"safety": {"gravity": [0, 0, 0]}, **{k: v for k, v in state().items() if k != "safety"}}])
def test_pose_missing_or_unusable_gives_none(broken):
    assert Pose.from_state(broken) is None


def test_pose_with_bad_trunk_height_falls_back_to_nominal_height():
    p = Pose.from_state(state(z=0.0))
    assert floor_model(p, CFG.low).height == pytest.approx(0.246, abs=1e-6)
