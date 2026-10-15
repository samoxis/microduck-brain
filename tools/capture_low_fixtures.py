"""Generate the test fixtures from the REAL simulated ToF sensor (microduck_rl, MuJoCo).

Run it with microduck_rl's venv (it is not part of the regular test suite):

    cd ~/microduck_rl && .venv/bin/python ~/microduck-brain/tools/capture_low_fixtures.py          # cubes, balls, tilt
    cd ~/microduck_rl && .venv/bin/python ~/microduck-brain/tools/capture_low_fixtures.py --drop   # the staircase

Every frame comes with the pose the robot would report in robot.state: the sensor's position and
orientation in the trunk (frames.tof, [w,x,y,z]), gravity projected into the trunk, and the trunk's
height. The robot stands in the STAND keyframe on a flat floor; the trunk can be tilted (pitch/roll)
to imitate the gait's sway. A 7 cm cube or an 8 cm ball is placed x metres ahead of the trunk.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

_args = [a for a in sys.argv[1:] if not a.startswith("--")]
RL = Path(_args[0]) if _args else Path.home() / "microduck_rl"
sys.path.insert(0, str(RL / "src"))
from mjlab_microduck.sim.tof import Tof  # noqa: E402

D = RL / "src/mjlab_microduck/robot/microduck"
OUT = Path(__file__).resolve().parent.parent / "tests/fixtures/mujoco_low.json"
OUT_DROP = Path(__file__).resolve().parent.parent / "tests/fixtures/mujoco_drop.json"


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2]


def capture(shape, x, size, pitch_deg=0.0, roll_deg=0.0, lateral=0.0, seed=0):
    spec = mujoco.MjSpec.from_file(str(D / "scene_walk.xml"))
    if shape != "none":
        gtype = mujoco.mjtGeom.mjGEOM_BOX if shape == "cube" else mujoco.mjtGeom.mjGEOM_SPHERE
        spec.worldbody.add_geom(name="obj", type=gtype, size=[size / 2] * 3,
                                pos=[x + size / 2, 0.0224 + lateral, size / 2])
    m = spec.compile()
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_KEY, i) for i in range(m.nkey)]
    d = mujoco.MjData(m)
    d.qpos[:] = m.key_qpos[names.index("STAND")]
    # trunk tilt: pitch + = nose down (rotation about +y)
    p, r = np.radians(pitch_deg) / 2, np.radians(roll_deg) / 2
    qp, qr = [np.cos(p), 0, np.sin(p), 0], [np.cos(r), np.sin(r), 0, 0]
    d.qpos[3:7] = quat_mul(qr, qp)
    mujoco.mj_forward(m, d)
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tof")
    trunk = m.body("trunk_base").id if "trunk_base" in [m.body(i).name for i in range(m.nbody)] else 1
    Rt = d.xmat[trunk].reshape(3, 3)            # trunk -> world
    Rs = d.site_xmat[site].reshape(3, 3)        # sensor -> world
    R_ts = Rt.T @ Rs                            # sensor -> trunk
    pos_t = Rt.T @ (d.site_xpos[site] - d.xpos[trunk])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R_ts.flatten())
    gravity = Rt.T @ np.array([0.0, 0.0, -1.0])
    t = Tof(m, site, seed=seed)
    dist, status = t.frame(d)
    return {
        "shape": shape, "x": x, "size": size, "pitch_deg": pitch_deg, "roll_deg": roll_deg,
        "lateral": lateral, "seed": seed,
        "pose": {"sensor_pos": pos_t.tolist(), "sensor_quat": quat.tolist(),
                 "gravity": gravity.tolist(), "trunk_z": float(d.xpos[trunk][2])},
        "distance_mm": [int(v) for v in dist], "status": [int(v) for v in status],
    }


def capture_apartment(model, x, y, yaw_deg, label, seed=0, pitch_deg=0.0):
    """The robot in STAND in the apartment scene, at (x, y) facing yaw. For the stairwell."""
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i) for i in range(model.nkey)]
    d = mujoco.MjData(model)
    d.qpos[:] = model.key_qpos[names.index("STAND")]
    d.qpos[0], d.qpos[1] = x, y
    a, p = np.radians(yaw_deg) / 2, np.radians(pitch_deg) / 2
    d.qpos[3:7] = quat_mul([np.cos(a), 0, 0, np.sin(a)], [np.cos(p), 0, np.sin(p), 0])
    mujoco.mj_forward(model, d)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tof")
    trunk = 1
    Rt = d.xmat[trunk].reshape(3, 3)
    R_ts = Rt.T @ d.site_xmat[site].reshape(3, 3)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R_ts.flatten())
    t = Tof(model, site, seed=seed)
    dist, status = t.frame(d)
    return {
        "label": label, "x": x, "y": y, "yaw_deg": yaw_deg, "pitch_deg": pitch_deg, "seed": seed,
        "pose": {"sensor_pos": (Rt.T @ (d.site_xpos[site] - d.xpos[trunk])).tolist(), "sensor_quat": quat.tolist(),
                 "gravity": (Rt.T @ np.array([0.0, 0.0, -1.0])).tolist(), "trunk_z": float(d.xpos[trunk][2])},
        "distance_mm": [int(v) for v in dist], "status": [int(v) for v in status],
    }


def main_drop():
    """The apartment's stairwell: x in [-0.40, 0.00], y in [-1.40, -0.70]; the first step at -10 cm."""
    model = mujoco.MjModel.from_xml_path(str(D / "scene_apartment.xml"))
    out = []
    # approaches: d = distance from the trunk origin to the edge of the hole
    for dd in (1.4, 1.2, 1.0, 0.85, 0.75, 0.6, 0.45, 0.3, 0.2):
        for pitch in (0.0, 3.0, -3.0):
            out.append(capture_apartment(model, -0.2, -0.7 + dd, -90, f"north d={dd}", pitch_deg=pitch))
    for dd in (0.6, 0.5, 0.4, 0.3, 0.2):
        out.append(capture_apartment(model, -0.4 - dd, -1.05, 0, f"west d={dd}"))
    for dd in (0.35, 0.3, 0.2):
        out.append(capture_apartment(model, 0.0 + dd, -1.05, 180, f"east d={dd}"))
    # from the south: the bottom step is 40 cm below the floor and the beams hit the steps (longer ranges)
    for dd in (1.2, 0.9, 0.6, 0.4, 0.2):
        out.append(capture_apartment(model, -0.2, -1.4 - dd, 90, f"south d={dd}"))
    # walking past the hole, heading south, 20 cm east of its edge
    for y in (0.0, -0.3, -0.6, -0.9, -1.2):
        out.append(capture_apartment(model, 0.2, y, -90, f"alongside x=0.2 y={y}"))
    # ordinary floor in the apartment (rooms, rugs, furniture, walls): must never read as a drop
    for (x, y) in ((-2.5, -1.0), (-2.5, 1.8), (2.2, 1.5), (2.2, 0.0), (2.2, -2.0), (-0.2, 1.5), (-1.5, -2.0), (-0.6, 2.2)):
        for yaw in (0, 90, 180, 270):
            for seed in (0, 1):
                out.append(capture_apartment(model, x, y, yaw, f"room ({x},{y}) yaw={yaw}", seed=seed))
    OUT_DROP.write_text(json.dumps(out, separators=(",", ":")))
    print(f"{len(out)} frames -> {OUT_DROP} ({OUT_DROP.stat().st_size // 1024} KiB)")


def main():
    out = []
    for seed in range(12):
        for pitch in (-3.0, -1.5, 0.0, 1.5, 3.0):
            out.append(capture("none", 0, 0, pitch_deg=pitch, seed=seed))
        for roll in (-3.0, 3.0):
            out.append(capture("none", 0, 0, roll_deg=roll, seed=seed))
        out.append(capture("none", 0, 0, pitch_deg=3.0, roll_deg=3.0, seed=seed))
    for shape, size in (("cube", 0.07), ("ball", 0.08)):
        for x in (0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7):
            for lateral in (-0.04, 0.0, 0.04):
                for pitch in (0.0, 3.0, -3.0):
                    out.append(capture(shape, x, size, pitch_deg=pitch, lateral=lateral, seed=1))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print(f"{len(out)} frames -> {OUT} ({OUT.stat().st_size // 1024} KiB)")


if __name__ == "__main__":
    if "--drop" in sys.argv:
        main_drop()
    else:
        main()
