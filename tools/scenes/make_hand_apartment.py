"""Write the apartment scene plus a "hand": a mocap body (a 9 x 9 cm, 2 cm thick palm) that touches
nothing (contype 0) but the depth sensor's rays see. tools/push/sitecustomize.py moves it in front of
the sensor on demand ($DUCK_HAND_FILE); used by tools/gesture_trials.sh.

    ~/microduck_rl/.venv/bin/python tools/scenes/make_hand_apartment.py \\
        ~/microduck_rl/src/mjlab_microduck/robot/microduck/scene_apartment_hand.xml

OUT.xml must sit next to scene_apartment.xml (it is read from OUT's directory).
"""

import sys
from pathlib import Path

import mujoco

out = sys.argv[1]
spec = mujoco.MjSpec.from_file(str(Path(out).resolve().parent / "scene_apartment.xml"))
body = spec.worldbody.add_body(name="hand", mocap=True, pos=[40.0, 40.0, 5.0])
# thin along x: the sensor's +x points at its face
body.add_geom(name="hand_palm", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.01, 0.045, 0.045],
              contype=0, conaffinity=0, rgba=[0.95, 0.8, 0.7, 1.0])
Path(out).write_text(spec.to_xml())
print(f"wrote {out}")
