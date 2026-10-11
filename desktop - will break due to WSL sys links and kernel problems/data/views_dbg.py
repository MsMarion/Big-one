"""How many hand-board views does the eye-to-hand sampler find for a given camera pose? (tuning aid)"""
import sys

import numpy as np

sys.path.insert(0, "/repo")
from pnp_core import robot_description as rdesc  # noqa: E402
from pnp_core.calibration import joint_space_board_views, mount_T  # noqa: E402
from pnp_core.catalog import load_fixed_camera, load_robot, load_scene, load_yaml, repo_root  # noqa: E402
from pnp_core.geometry import look_at_optical, make_transform, rpy_to_matrix  # noqa: E402
from pnp_core.urdf_kin import Chain  # noqa: E402

cfg = load_robot("so101")
urdf = rdesc.urdf(cfg, repo_root(), "mock")
scene = load_scene("config/scenes/so101_lego.yaml")
cam = load_fixed_camera(scene)
K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1.0]])
size = (cam["width"], cam["height"])
b = load_yaml("config/calibration_targets_so101.yaml")["hand_board"]
hand = Chain(urdf, "base_link", b["mount"]["parent"]).clamp_limits(cfg["arm"]["position_limits"], cfg["arm"]["limit_margin"])
m = cam["mount"]
T_nom = make_transform(look_at_optical(m["position"], m["look_at"]), m["position"])
e = load_yaml("config/sim_reality.yaml")["profiles"]["misplaced_camera"]["fixed_camera_error"]
T_mis = T_nom @ make_transform(rpy_to_matrix(*np.radians(e["rpy_deg"])), np.array(e["xyz_mm"]) / 1000)
pz = b["poses"]
for name, T in (("nominal", T_nom), ("misplaced (true)", T_mis)):
    for label, rng_ in (("config ranges", pz.get("ranges")), ("full joint limits", None)):
        for vis in (1.0, 0.6, 0.5):
            qs = joint_space_board_views(hand.fk, hand.lower, hand.upper, 50, np.random.default_rng(0), T, K, size,
                                         mount_T(b), b, dist=tuple(pz["dist"]), max_view_angle_deg=pz["max_view_angle_deg"],
                                         ranges=rng_, tries=100000, min_visible=vis)
            print(f"{name:18s} {label:18s} visible >= {vis}: {len(qs)} views")
print("camera nominal at", np.round(T_nom[:3, 3], 3), " misplaced at", np.round(T_mis[:3, 3], 3))
