# Test hook for the MuJoCo body: when the file $DUCK_PUSH_FILE appears, give the trunk a sideways
# shove (adds to its linear velocity once), then delete the file. Patches mujoco.mj_step lazily,
# when mujoco is first imported, so nothing is imported at interpreter start-up.
import os
import sys
import importlib.abc
import importlib.machinery

PUSH = os.environ.get("DUCK_PUSH_FILE")


def _patch(mujoco):
    step = mujoco.mj_step
    state = {"n": 0, "dof": None}

    def mj_step(model, data, *a, **k):
        state["n"] += 1
        if state["n"] % 20 == 0 and os.path.exists(PUSH):
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


if PUSH:
    sys.meta_path.insert(0, _Finder())
