"""Odometry correction against old walls (localize.py)."""

from __future__ import annotations

import math

import pytest

from microduck_brain.brain import Brain, Inputs
from microduck_brain.config import Config, from_dict
from microduck_brain.localize import Localizer

from . import world2d

ON = {"localize": {"enabled": True, "every": 1}}


def mapped_room():
    """A localizer that has seen the 4x3 room all around from (2.0, 1.5), long ago."""
    w = world2d.worlds()[0]
    loc = Localizer(from_dict(ON))
    for a in (0.6, 1.6, 2.6, 3.6, 4.6, 5.6):
        loc.update(w.render(2.0, 1.5, a), (2.0, 1.5), a, 0.064, 0.0)
        loc._last_frame = None
    return w, loc


@pytest.mark.parametrize("off,improves", [((0.05, 0.03, -0.02), True), ((0.0, 0.0, 0.02), True),
                                         ((0.03, -0.02, 0.0), False)])
def test_a_wrong_pose_is_pulled_towards_the_walls_seen_before(off, improves):
    """Large errors shrink. A small one (3.6 cm) is left alone: the +-4 cm steps can't resolve it,
    and the thresholds (2 cm and 50% better) are set so that it isn't "corrected" into a worse one,
    which is what made good odometry worse before."""
    w, loc = mapped_room()
    x, y, yaw = 2.0, 1.5, 0.6
    ox, oy, oa = x + off[0], y + off[1], yaw + off[2]
    before = math.hypot(off[0], off[1]) + abs(off[2]) * 1.0
    for k in range(30):
        loc.update(w.render(x, y, yaw), (ox, oy), oa, 0.064, 100.0 + k)
        loc._last_frame = None
    (cx, cy), ca = loc.correct((ox, oy), oa)
    after = math.hypot(cx - x, cy - y) + abs(ca - yaw) * 1.0
    if improves:
        assert loc.corrections > 0 and after < before
    else:
        assert math.hypot(cx - x, cy - y) <= math.hypot(off[0], off[1]) + 1e-9


def test_new_walls_do_not_count_until_min_age():
    w, loc = mapped_room()
    pts = loc._points(w.render(2.0, 1.5, 0.6), 0.064, 0.0)
    assert loc._cost(pts, 2.0, 1.5, 0.6, 100.0)[1] > 0          # old enough: they match
    assert loc._cost(pts, 2.0, 1.5, 0.6, 10.0)[1] == 0          # 10 s after: not yet


def test_strong_drift_is_reduced_in_the_living_room():
    """Measured over 6 fresh seeds x 2 rooms x 300 s: strong drift 0.56-1.10 m -> 0.30-0.54 m;
    but with perfect odometry the correction itself wanders 0.13-0.33 m. Off by default."""
    cfg = from_dict({"localize": {"enabled": True}})
    r = world2d.run(world2d.worlds()[3], cfg, 300, 60, odom_yaw_drift=0.002, odom_scale=1.03)
    assert r.corrected_error < r.odom_error


def test_off_by_default_and_a_fall_resets_it():
    assert not Config().localize.enabled
    from microduck_brain.recover import FALLEN

    b = Brain(from_dict({"localize": {"enabled": True}}))
    b.localizer.dx, b.localizer.dyaw = 0.3, 0.1
    b.tick(Inputs(0.0, None, False, FALLEN, 0.0, None, (0.0, 0.0)))
    assert b.localizer.dx == 0.0 and b.localizer.dyaw == 0.0


def test_localize_config_validation():
    for bad in ({"every": 0}, {"min_gain_m": 0.0}, {"search_step": 0.0}):
        with pytest.raises(ValueError):
            from_dict({"localize": bad})


def test_icp_pulls_a_wrong_pose_towards_the_walls_seen_before():
    """method = "icp": point-to-point alignment, no grid. On the full-simulator recordings with
    odometry degraded like a slipping robot's it lowered the mean error in 5 of 6 held-out runs
    (-8% medium drift, -3% strong), and added 2-5 cm with good odometry."""
    w = world2d.worlds()[0]
    cfg = from_dict({"localize": {"enabled": True, "every": 1, "method": "icp"}})
    loc = Localizer(cfg)
    for a in (0.6, 1.6, 2.6, 3.6, 4.6, 5.6):
        loc.update(w.render(2.0, 1.5, a), (2.0, 1.5), a, 0.064, 0.0)
        loc._last_frame = None
        loc._prev = None
    x, y, yaw = 2.0, 1.5, 0.6
    ox, oy = x + 0.05, y + 0.03
    for k in range(60):
        loc.update(w.render(x, y, yaw), (ox, oy), yaw, 0.064, 100.0 + k)
        loc._last_frame = None
        loc._prev = None
    (cx, cy), _ = loc.correct((ox, oy), yaw)
    assert loc.corrections > 0 and math.hypot(cx - x, cy - y) < math.hypot(0.05, 0.03)


def test_localize_method_is_validated():
    with pytest.raises(ValueError):
        from_dict({"localize": {"method": "magic"}})
