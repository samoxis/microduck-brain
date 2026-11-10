"""Write a copy of the apartment scene with the flat moved so the duck, which body_server.py always
places at the origin facing +x, starts at a chosen pose in it (x, y in m, heading in deg).

    ~/microduck_rl/.venv/bin/python tools/scenes/make_shifted_apartment.py X Y HEADING_DEG \
        ~/microduck_rl/src/mjlab_microduck/robot/microduck/scene_apartment_desk_leg.xml

OUT.xml must sit next to scene_apartment.xml (it is read from OUT's directory). Used by
tools/desk_leg_trials.sh: by default 3.06 0.48 31, 6 s (true pose) before the desk-leg wedge of run
ab-head-pitch-b62de6c-turnonly/20261104-022552-off (2.70 0.27 26: 11 s before), walking towards
leg off_desk_l3 at (3.40, 0.36).
"""

import math
import sys
from pathlib import Path

import mujoco
import numpy as np

x0, y0, h, out = float(sys.argv[1]), float(sys.argv[2]), math.radians(float(sys.argv[3])), sys.argv[4]
here = Path(out).resolve().parent
spec = mujoco.MjSpec.from_file(str(here / "scene_apartment.xml"))
c, s = math.cos(-h), math.sin(-h)
qrot = np.array([math.cos(-h / 2), 0, 0, math.sin(-h / 2)])


def move(el):
    px, py = el.pos[0] - x0, el.pos[1] - y0
    el.pos = [c * px - s * py, s * px + c * py, el.pos[2]]
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, qrot, np.array(el.quat, dtype=float))
    el.quat = q


robot = {"trunk_base"}
n = 0
for b in spec.worldbody.bodies:
    if b.name not in robot:
        move(b); n += 1
for kind in ("geoms", "sites", "cameras"):
    for e in getattr(spec.worldbody, kind):
        move(e); n += 1
Path(out).write_text(spec.to_xml())
print(f"moved {n} elements; duck starts at ({x0}, {y0}) heading {math.degrees(h):.0f} deg")
