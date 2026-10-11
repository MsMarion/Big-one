"""pnp_core.wrist_refine: the close look from wrist depth, on synthetic points (desk + a brick seen from an angle)."""
import numpy as np

from pnp_core.catalog import load_object
from pnp_core.object_grasp import object_yaw, rotation_error_deg, upright_pose
from pnp_core.wrist_refine import refine_on_desk

BRICK = load_object("lego_2x4")
DESK = -0.0024


def brick_points(xy, yaw, h=0.0096, step=0.0007, seen_side=(0.0, -1.0)):
    """Top face + the one long side facing the camera (a wrist view from ~30 deg off vertical)."""
    L, W = BRICK["dims_m"][0], BRICK["dims_m"][1]
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s], [s, c]])
    u, v = np.meshgrid(np.arange(-L / 2, L / 2, step), np.arange(-W / 2, W / 2, step))
    top = np.column_stack([np.c_[u.ravel(), v.ravel()] @ Rz.T + xy, np.full(u.size, DESK + h)])
    side_v = -W / 2 if np.dot(Rz @ [0, 1], seen_side) < 0 else W / 2
    a, z = np.meshgrid(np.arange(-L / 2, L / 2, step), np.arange(DESK, DESK + h, step))
    side = np.column_stack([np.c_[a.ravel(), np.full(a.size, side_v)] @ Rz.T + xy, z.ravel()])
    return np.vstack([top, side])


def desk_points(center, half=0.05, step=0.001, holes=()):
    g = np.arange(-half, half, step)
    x, y = np.meshgrid(center[0] + g, center[1] + g)
    P = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, DESK)])
    for xy in holes:                                            # the desk under a brick isn't seen
        P = P[np.hypot(P[:, 0] - xy[0], P[:, 1] - xy[1]) > 0.012]
    return P


def scene(true_xy, true_yaw, noise=0.0003, extra=()):
    rng = np.random.default_rng(3)
    P = np.vstack([desk_points(true_xy, holes=[true_xy, *[e[0] for e in extra]]), brick_points(true_xy, true_yaw)]
                  + [brick_points(xy, yaw) for xy, yaw in extra])
    return P + rng.normal(0, noise, P.shape)


def test_refines_center_and_yaw_from_a_rough_estimate():
    true_xy, true_yaw = np.array([0.191, 0.058]), 0.35
    T_init = upright_pose(BRICK, [0.196, 0.064], DESK + 0.006, 0.50)     # 8 mm and 9 deg off
    r = refine_on_desk(scene(true_xy, true_yaw), T_init, BRICK, desk_z=DESK)
    assert r.T_base_obj is not None, r.message
    T = r.T_base_obj
    assert np.hypot(*(T[:2, 3] - true_xy)) < 0.0015
    assert abs(T[2, 3] - (DESK + BRICK["dims_m"][2] / 2)) < 5e-4          # center height: the model on the measured desk
    assert rotation_error_deg(T[:3, :3], upright_pose(BRICK, true_xy, 0, true_yaw)[:3, :3], BRICK) < 2.0


def test_picks_the_symmetric_twin_nearest_the_estimate():
    """A 2x4 brick looks the same turned 180 deg: the answer stays near the first estimate's yaw."""
    true_xy, true_yaw = np.array([0.18, -0.02]), 0.2
    T_init = upright_pose(BRICK, true_xy, DESK + 0.006, 0.2 + np.pi + 0.1)
    r = refine_on_desk(scene(true_xy, true_yaw), T_init, BRICK, desk_z=DESK)
    assert r.T_base_obj is not None, r.message
    assert abs(np.degrees(np.angle(np.exp(1j * (object_yaw(r.T_base_obj[:3, :3], BRICK) - (0.2 + np.pi)))))) < 2.0


def test_a_touching_neighbour_is_rejected():
    true_xy, true_yaw = np.array([0.19, 0.06]), 0.0
    neighbour = (true_xy + [BRICK["dims_m"][0], 0.0], 0.0)              # end to end: one 64 mm blob
    r = refine_on_desk(scene(true_xy, true_yaw, extra=[neighbour]), upright_pose(BRICK, true_xy, 0.0, 0.0), BRICK,
                       desk_z=DESK)
    assert r.T_base_obj is None and "footprint" in r.message


def test_a_neighbour_a_little_apart_does_not_disturb_it():
    true_xy, true_yaw = np.array([0.19, 0.06]), 0.3
    neighbour = (true_xy + [0.0, 0.035], 1.2)
    T_init = upright_pose(BRICK, true_xy + [0.004, -0.003], 0.0, 0.25)
    r = refine_on_desk(scene(true_xy, true_yaw, extra=[neighbour]), T_init, BRICK, desk_z=DESK)
    assert r.T_base_obj is not None, r.message
    assert np.hypot(*(r.T_base_obj[:2, 3] - true_xy)) < 0.0015


def test_no_object_or_wrong_desk_gives_no_answer():
    xy = np.array([0.19, 0.06])
    T = upright_pose(BRICK, xy, 0.0, 0.0)
    assert refine_on_desk(desk_points(xy), T, BRICK, desk_z=DESK).T_base_obj is None
    assert "desk" in refine_on_desk(scene(xy, 0.0), T, BRICK, desk_z=DESK + 0.03).message


def test_the_jaws_above_the_brick_are_not_part_of_it():
    """At the pre-grasp pose the open jaws hang 3-9 cm above the brick, right over it, in the wrist camera's view."""
    true_xy, true_yaw = np.array([0.19, 0.06]), 0.4
    g = np.arange(-0.02, 0.02, 0.001)
    jx, jz = np.meshgrid(g, np.arange(0.035, 0.09, 0.001))
    jaws = np.column_stack([true_xy[0] + jx.ravel(), np.full(jx.size, true_xy[1] + 0.012), DESK + jz.ravel()])
    P = np.vstack([scene(true_xy, true_yaw), jaws, jaws - [0, 0.024, 0]])
    r = refine_on_desk(P, upright_pose(BRICK, true_xy + [0.003, 0.002], 0.0, 0.3), BRICK, desk_z=DESK)
    assert r.T_base_obj is not None, r.message
    assert np.hypot(*(r.T_base_obj[:2, 3] - true_xy)) < 0.0015
