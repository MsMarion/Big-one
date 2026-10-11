"""Which tags are off their spots? Needs NO camera calibration: each tag's centre on the board from the image
(the board plane from all tags' poses, every centre's ray onto it), then the best rigid fit (shift + turn) of those
to the layout plan, ignoring the worst tags while fitting -> how far each tag is from its spot, in the board frame
(x forward from the base, y to its left). The set's shift / turn as a whole comes out too (the camera's view of it).
    docker exec pnp_dev bash /repo/data/rospy.sh /repo/data/tag_placement_check.py"""
import sys
import time

import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

sys.path.insert(0, "/repo")
from pnp_core.calibration import detect_tags, image_to_gray  # noqa: E402
from pnp_core.catalog import calibration_targets_path, load_yaml  # noqa: E402

rclpy.init()
node = rclpy.create_node("tag_placement_check")
got = {}
node.create_subscription(Image, "/camera/camera/color/image_raw", lambda m: got.__setitem__("img", m), qos_profile_sensor_data)
node.create_subscription(CameraInfo, "/camera/camera/color/camera_info", lambda m: got.__setitem__("info", m), qos_profile_sensor_data)
a = load_yaml(calibration_targets_path())["apriltags"]
plan = {int(t["id"]): np.array(t["sim_xy"], float) for t in a["tags"]}
acc = {}
for _ in range(6):
    got.pop("img", None)
    end = time.time() + 5
    while any(k not in got for k in ("img", "info")) and time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    K = np.array(got["info"].k).reshape(3, 3)
    tags = detect_tags(image_to_gray(got["img"]), a["dictionary"], a["size_m"], K, None, with_corners=True)
    tags = {k: v for k, v in tags.items() if k in plan}
    if len(tags) < 4:
        continue
    # the board plane (camera frame): normal = median tag z axis, offset = median n . centre
    n = np.median([T[:3, 2] for T, _ in tags.values()], axis=0)
    n /= np.linalg.norm(n)
    dist = np.median([n @ T[:3, 3] for T, _ in tags.values()])
    for tid, (T, c) in tags.items():
        ray = np.linalg.inv(K) @ np.r_[c.mean(axis=0), 1.0]
        acc.setdefault(tid, []).append(ray * dist / (n @ ray))    # the centre's ray onto the plane
ids = sorted(acc)
P = {i: np.median(acc[i], axis=0) for i in ids}
# 2-D board coordinates in the plane: any in-plane basis (the rigid fit below finds the turn)
n = np.median([np.cross(P[ids[1]] - P[ids[0]], P[ids[2]] - P[ids[0]])], axis=0)
n /= np.linalg.norm(n)
e1 = P[ids[1]] - P[ids[0]]
e1 -= n * (n @ e1)
e1 /= np.linalg.norm(e1)
e2 = np.cross(n, e1)
Q = {i: np.array([P[i] @ e1, P[i] @ e2]) for i in ids}


def fit(use):
    """Rigid 2-D (with a possible mirror: the plane's basis handedness is arbitrary) of measured -> plan."""
    best = None
    for flip in (1.0, -1.0):
        A = np.array([[Q[i][0], flip * Q[i][1]] for i in use])
        B = np.array([plan[i] for i in use])
        ma, mb = A.mean(axis=0), B.mean(axis=0)
        U, _, Vt = np.linalg.svd((A - ma).T @ (B - mb))
        R = (U @ Vt).T
        if np.linalg.det(R) < 0:
            continue
        res = {i: np.array([Q[i][0], flip * Q[i][1]]) @ R.T + mb - ma @ R.T for i in ids}
        err = np.mean([np.linalg.norm(res[i] - plan[i]) for i in use])
        if best is None or err < best[0]:
            best = (err, res)
    return best[1]


use = list(ids)
for _ in range(len(ids) - 4):                    # drop the worst tag while it's far worse than the rest
    res = fit(use)
    e = {i: np.linalg.norm(res[i] - plan[i]) for i in use}
    worst = max(e, key=e.get)
    if e[worst] < 0.006:
        break
    use.remove(worst)
res = fit(use)
print(f"tags measured: {ids}; the fit uses {use} (the ones that agree with each other)")
for i in ids:
    d = (res[i] - plan[i]) * 1000
    flag = "OK" if np.hypot(*d) <= 5 else "MOVE"
    print(f"  tag {i}: {flag:4s} off its spot by x {d[0]:+5.0f} mm, y {d[1]:+5.0f} mm  ({d[0] / 25.4:+.2f} / {d[1] / 25.4:+.2f} in)"
          f"  -> {'' if flag == 'OK' else 'move it ' + ('back toward the base' if d[0] > 0 else 'forward') + f' {abs(d[0]) / 10:.1f} cm, ' + ('to the robot RIGHT' if d[1] > 0 else 'to the robot LEFT') + f' {abs(d[1]) / 10:.1f} cm'}")

# calibration-free facts to check with a tape: centre-to-centre distances between tags, measured vs the plan
print("centre-to-centre distances (measured on the board vs the layout sheet):")
pairs = [(3, 4), (1, 2), (5, 6), (7, 8), (3, 7), (4, 8), (1, 3), (2, 4), (5, 7), (6, 8), (3, 5), (4, 6), (7, 9), (8, 9)]
for i, j in pairs:
    if i in P and j in P:
        m_, p_ = np.linalg.norm(P[i] - P[j]), np.linalg.norm(plan[i] - plan[j])
        print(f"  {i}-{j}: {m_ * 100:5.1f} cm (plan {p_ * 100:5.1f}, {(m_ - p_) * 100:+5.1f})")
