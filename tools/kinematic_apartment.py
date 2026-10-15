"""Closed-loop run of microduck-brain in the MuJoCo apartment scene, with the body moved kinematically.

What is real here: the scene (six rooms, furniture, 4 cubes of 7 cm, 3 balls of 8 cm, the descending
staircase in the corridor), the simulated ToF sensor (microduck_rl's own ``sim/tof.py``: same beams,
noise and status codes), the sensor pose the robot would report, and the brain, unchanged.

What is NOT real: walking. There is no physics step and no walking policy (those need the policy
files from the Hugging Face Hub). The body is moved ideally along the commanded velocities, scaled by
the real/commanded ratios measured in the full simulator (forward ~42%, turning ~46-69%), with a
small synthetic trunk sway. So this exercises perception and decisions in the real geometry, not
balance or gait. Odometry is perfect here; on the robot it drifts.

Contacts are counted from MuJoCo's own collision detection between the robot's collision geometry and
the scene (floors excluded); entering the stairwell is checked against the hole's rectangle.

Run with microduck_rl's venv:

    cd ~/microduck_rl && .venv/bin/python ~/microduck-brain/tools/kinematic_apartment.py \
        --seconds 300 --out ~/microduck-brain/docs/apartment.jsonl.gz --svg ~/microduck-brain/docs/apartment.svg
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from microduck_brain.brain import Brain, Inputs  # noqa: E402
from microduck_brain.config import load  # noqa: E402
from microduck_brain.depth import DepthFrame, Pose  # noqa: E402
from microduck_brain.record import Recorder, metrics, read  # noqa: E402

HOLE = (-0.40, 0.00, -1.40, -0.70)   # stairwell: x0, x1, y0, y1
DT = 0.05                            # brain at 20 Hz
TOF_PERIOD = 1.0 / 15.0              # sensor at 15 Hz


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl", default=str(Path.home() / "microduck_rl"), help="microduck_rl checkout")
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--start", type=float, nargs=3, default=(-2.0, -0.3, -90.0), metavar=("X", "Y", "YAW_DEG"))
    ap.add_argument("--forward-efficiency", type=float, default=0.42)
    ap.add_argument("--turn-efficiency", type=float, default=0.55)
    ap.add_argument("--sway-deg", type=float, default=2.0, help="synthetic trunk pitch sway while walking")
    ap.add_argument("--config", default=str(REPO / "brain.toml"))
    ap.add_argument("--out", default="apartment.jsonl.gz")
    ap.add_argument("--svg", default="")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.rl) / "src"))
    from mjlab_microduck.sim.tof import Tof

    scene = Path(args.rl) / "src/mjlab_microduck/robot/microduck/scene_apartment.xml"
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    keys = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i) for i in range(model.nkey)]
    data.qpos[:] = model.key_qpos[keys.index("STAND")]
    stand_z = float(data.qpos[2])
    trunk = int(model.jnt_bodyid[0])                 # the robot's free joint is the first joint
    robot_bodies = {b for b in range(model.nbody) if model.body_rootid[b] == model.body_rootid[trunk]}
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tof")
    tof = Tof(model, site, seed=0)
    cfg = load(args.config)
    brain = Brain(cfg)
    rec = Recorder(args.out, {"source": "kinematic MuJoCo apartment", "args": vars(args)})

    x, y, yaw = args.start[0], args.start[1], math.radians(args.start[2])
    frame: DepthFrame | None = None
    pose: Pose | None = None
    next_frame = 0.0
    phase = 0.0
    touching: set[str] = set()
    contacts: dict[str, int] = {}
    events: list[dict] = []
    in_hole_ticks = 0
    t = 0.0
    vx = vyaw = 0.0
    while t < args.seconds:
        # place the body: STAND pose, at (x, y, yaw), with a small sway while walking
        if vx > 0:
            phase += DT
        pitch = math.radians(args.sway_deg) * math.sin(2 * math.pi * 1.8 * phase) if vx > 0 else 0.0
        roll = math.radians(args.sway_deg * 0.6) * math.sin(2 * math.pi * 0.9 * phase) if vx > 0 else 0.0
        qz = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        qp = [math.cos(pitch / 2), 0, math.sin(pitch / 2), 0]
        qr = [math.cos(roll / 2), math.sin(roll / 2), 0, 0]
        data.qpos[0], data.qpos[1], data.qpos[2] = x, y, stand_z
        data.qpos[3:7] = quat_mul(qz, quat_mul(qp, qr))
        mujoco.mj_forward(model, data)

        # contacts between the robot and anything that is not a floor
        now_touching = set()
        for c in data.contact[: data.ncon]:
            b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
            if (b1 in robot_bodies) == (b2 in robot_bodies):
                continue
            other = c.geom2 if b1 in robot_bodies else c.geom1
            mine = c.geom1 if b1 in robot_bodies else c.geom2
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or f"geom{other}"
            if name.startswith(("floor", "rug")):
                continue
            now_touching.add(name)
            if name not in touching:
                # where on the body, in the trunk frame (x forward, y left)
                rel = data.xmat[trunk].reshape(3, 3).T @ (np.array(c.pos) - data.xpos[trunk])
                part = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[mine])
                events.append({"t": round(t, 2), "state": brain.state_name, "object": name, "robot_part": part,
                               "at_trunk_xyz": [round(float(v), 3) for v in rel], "commanded": [round(vx, 3), round(vyaw, 3)]})
        for name in now_touching - touching:
            contacts[name] = contacts.get(name, 0) + 1
        touching = now_touching
        if HOLE[0] < x < HOLE[1] and HOLE[2] < y < HOLE[3]:
            in_hole_ticks += 1

        # the sensor at 15 Hz, with the pose the robot would report for that instant
        if t >= next_frame:
            next_frame += TOF_PERIOD
            dist, status = tof.frame(data)
            frame = DepthFrame(8, 8, tuple(int(v) for v in dist), tuple(int(v) for v in status))
            Rt = data.xmat[trunk].reshape(3, 3)
            R_ts = Rt.T @ data.site_xmat[site].reshape(3, 3)
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, R_ts.flatten())
            pose = Pose(tuple(Rt.T @ (data.site_xpos[site] - data.xpos[trunk])), tuple(q),
                        tuple(Rt.T @ np.array([0.0, 0.0, -1.0])), float(data.xpos[trunk][2]))

        wrapped = (yaw + math.pi) % (2 * math.pi) - math.pi
        inp = Inputs(t, frame, True, "", wrapped, pose, (x, y))
        cmd = brain.tick(inp)
        rec.write(inp, cmd, brain.state_name)
        vx, vyaw = (cmd.vx, cmd.vyaw) if cmd.kind == "move" else (0.0, 0.0)
        yaw += args.turn_efficiency * vyaw * DT
        x += args.forward_efficiency * vx * math.cos(yaw) * DT
        y += args.forward_efficiency * vx * math.sin(yaw) * DT
        t += DT
    rec.close()

    ticks = list(read(args.out))
    st = brain.stats
    toys = {k: v for k, v in contacts.items() if k.startswith(("obj_", "ball_"))}
    other = {k: v for k, v in contacts.items() if k not in toys}
    out = {
        "seconds": args.seconds,
        "start": args.start,
        "efficiency": {"forward": args.forward_efficiency, "turn": args.turn_efficiency},
        "coverage": metrics(ticks),
        "avoidances": st.avoids,
        "avoidances_with_stop": st.brakes,
        "big_turns": st.big_turns,
        "low_objects_seen": st.low_seen,
        "drops_seen": st.drops_seen,
        "toy_contacts": toys,
        "wall_or_furniture_contacts": other,
        "ticks_over_the_stairwell": in_hole_ticks,
        "contact_events": events,
    }
    print(json.dumps(out, indent=2))
    if args.svg:
        from microduck_brain.plot import render_svg

        title = f"Kinematic run in the MuJoCo apartment ({args.seconds:.0f} s, real ToF model, real scene)"
        Path(args.svg).write_text(render_svg(ticks, cfg, args.out, title), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
