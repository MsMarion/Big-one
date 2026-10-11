"""Where the PLAN expects each tag in the fixed camera's image (the calibrated camera pose + the real intrinsics)."""
import sys
import numpy as np
import yaml
sys.path.insert(0, "/repo")
from pnp_core.calibration import flat_tag_T, tag_corners  # noqa: E402
from pnp_core.geometry import make_transform, quat_to_matrix  # noqa: E402
c = yaml.safe_load(open("/repo/config/calibration/so101_plate/real_d435i.yaml"))["T_parent_optical"]
X = make_transform(quat_to_matrix(c["quat_xyzw"]), c["xyz"])
K = np.array([[605.4, 0, 320], [0, 605.4, 240], [0, 0, 1.0]])
a = yaml.safe_load(open("/repo/config/calibration_targets_so101_plate.yaml"))["apriltags"]
Xi = np.linalg.inv(X)
for t in a["tags"]:
    P = tag_corners(a["size_m"]) @ flat_tag_T(t["sim_xy"], t["sim_yaw_deg"], -0.0024)[:3, :3].T \
        + flat_tag_T(t["sim_xy"], t["sim_yaw_deg"], -0.0024)[:3, 3]
    pc = P @ Xi[:3, :3].T + Xi[:3, 3]
    uv = (pc @ K.T)[:, :2] / pc[:, 2:3]
    m = uv.mean(axis=0)
    print(f"tag {t['id']}: planned centre pixel ({m[0]:4.0f}, {m[1]:4.0f}), {np.linalg.norm(uv[1] - uv[0]):3.0f} px across")
