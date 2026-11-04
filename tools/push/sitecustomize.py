# Test hook for the MuJoCo body: when the file $DUCK_PUSH_FILE appears, give the trunk a sideways
# shove (adds to its linear velocity once), then delete the file. With $DUCK_TRUTH_FILE, log where
# the trunk really is (ground truth, to measure odometry drift). With $DUCK_CONTACT_FILE, log every new
# contact of the robot with anything but the floor (furniture, walls, toys, the stairs). With
# $DUCK_HAND_FILE, when that file appears ("DIST SECONDS"), hold the scene's mocap body "hand" (a
# 9 x 9 cm palm, see tools/scenes/make_hand_apartment.py) DIST m in front of the depth sensor, facing
# it, for SECONDS, then take it away and delete the file. Patches mujoco.mj_step lazily,
# when mujoco is first imported, so nothing is imported at interpreter start-up.
import os
import sys
import importlib.abc
import importlib.machinery

PUSH = os.environ.get("DUCK_PUSH_FILE")
TRUTH = os.environ.get("DUCK_TRUTH_FILE")    # also: "monotonic_s x y yaw sim_time" of the trunk every 0.25 s
CONTACTS = os.environ.get("DUCK_CONTACT_FILE")  # also: append "sim_time kind other robot_geom" per new contact
HAND = os.environ.get("DUCK_HAND_FILE")
AWAY = (40.0, 40.0, 5.0)                      # where the hand waits, far outside the flat


def _patch(mujoco):
    import math
    import time

    step = mujoco.mj_step
    state = {"n": 0, "dof": None, "adr": None, "next": 0.0}
    out = open(TRUTH, "a", buffering=1) if TRUTH else None
    cout = open(CONTACTS, "a", buffering=1) if CONTACTS else None
    touching: set = set()

    def classify(model):
        """robot geoms; and for every other geom its kind: floor (ignored), stair, toy, furniture."""
        trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        root = model.body_rootid[trunk]
        robot = {g for g in range(model.ngeom) if model.body_rootid[model.geom_bodyid[g]] == root}
        kind = {}
        for g in range(model.ngeom):
            if g in robot:
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if name.startswith("floor") or model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
                kind[g] = "floor"
            elif name.startswith("stair"):
                kind[g] = "stair"
            elif model.geom_bodyid[g] != 0:
                kind[g] = "toy"                  # a free body: the cubes and balls
            else:
                kind[g] = "furniture"            # walls and furniture, fixed to the world
        return robot, kind

    def log_contacts(model, data):
        if "robot" not in state:
            state["robot"], state["kind"] = classify(model)
        robot, kind = state["robot"], state["kind"]
        now = set()
        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in robot and g2 not in robot:
                r, o = g1, g2
            elif g2 in robot and g1 not in robot:
                r, o = g2, g1
            else:
                continue
            if kind.get(o) == "floor":
                continue
            now.add((r, o))
            if (r, o) not in touching:
                oname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, o) or str(o)
                rname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, r) or str(r)
                cout.write(f"{data.time:.3f} {kind.get(o)} {oname} {rname}\n")
        touching.clear()
        touching.update(now)

    def hand(model, data):
        if "hand" not in state:
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
            state["hand"] = int(model.body_mocapid[b]) if b >= 0 else -1
            state["tof"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tof")
            state["until"] = None
        m = state["hand"]
        if m < 0 or state["tof"] < 0:
            return
        if state["until"] is None and state["n"] % 20 == 0 and os.path.exists(HAND):
            try:
                dist, secs = (float(v) for v in open(HAND).read().split()[:2])
            except ValueError:
                dist, secs = 0.10, 1.5
            state["dist"], state["until"] = dist, data.time + secs
            print(f"== hand: {dist:.2f} m in front for {secs:.1f} s at t={data.time:.2f}", flush=True)
        if state["until"] is not None:
            if data.time < state["until"]:
                site, rot = data.site_xpos[state["tof"]], data.site_xmat[state["tof"]].reshape(3, 3)
                data.mocap_pos[m] = site + rot[:, 0] * state["dist"]     # along the sensor's +x
                import numpy as np

                q = np.zeros(4)
                mujoco.mju_mat2Quat(q, data.site_xmat[state["tof"]])
                data.mocap_quat[m] = q
            else:
                data.mocap_pos[m] = AWAY
                state["until"] = None
                try:
                    os.remove(HAND)
                except OSError:
                    pass

    def mj_step(model, data, *a, **k):
        state["n"] += 1
        if HAND:
            hand(model, data)
        if cout is not None and state["n"] % 4 == 0:
            log_contacts(model, data)
        if out is not None and data.time >= state["next"]:
            if state["adr"] is None:
                j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
                state["adr"] = int(model.jnt_qposadr[j])
            q = data.qpos[state["adr"]: state["adr"] + 7]
            w, x, y, z = q[3:7]
            yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            out.write(f"{time.monotonic():.3f} {q[0]:.4f} {q[1]:.4f} {yaw:.4f} {data.time:.3f}\n")
            state["next"] = data.time + 0.25
        if PUSH and state["n"] % 20 == 0 and os.path.exists(PUSH):
            try:
                speed = float(open(PUSH).read().strip() or "1.5")
            except ValueError:
                speed = 1.5
            os.remove(PUSH)
            if state["dof"] is None:
                j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
                state["dof"] = int(model.jnt_dofadr[j])
            data.qvel[state["dof"] + 1] += speed
            print(f"== push: trunk +{speed} m/s sideways at t={data.time:.2f}", flush=True)
        return step(model, data, *a, **k)

    mujoco.mj_step = mj_step


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name != "mujoco":
            return None
        sys.meta_path.remove(self)
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            _patch(module)

        spec.loader.exec_module = exec_and_patch
        return spec


if PUSH or TRUTH or CONTACTS or HAND:
    sys.meta_path.insert(0, _Finder())
