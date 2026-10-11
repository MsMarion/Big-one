"""FoundationPose (NGC ONNX networks) vs the simulator's ground truth, on live sim frames.

    /opt/fp_venv/bin/python /repo/tools/fp_eval.py [--frames 5] [--swap-outputs]     (pnp:pose container,
                                                                                    sim + stack running)
Per frame: color + depth + camera_info + the detector's mask (/perception/mask) -> FoundationPose register()
(252 hypotheses, refine, score) -> pose of the can's mesh frame in the camera. Ground truth: TF camera <- `target`
(Isaac publishes the can prim's pose; the mesh was exported in exactly that frame by sim/export_mesh.py).

Reported: translation error [mm], axis error [deg] (angle between the true and estimated can axis: what matters
for grasping a can), full rotation error [deg] (includes the spin about the axis, only defined thanks to the
label texture), runtime. --swap-outputs tests the other assignment of the refine network's two outputs.
"""

import argparse
import sys
import time

sys.path[:0] = ["/repo", "/repo/ros2_ws/src/pnp_perception"]

import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from rclpy.qos import qos_profile_sensor_data  # noqa: E402
from rclpy.time import Time  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image  # noqa: E402
from tf2_ros import Buffer, TransformListener  # noqa: E402

from pnp_core.geometry import make_transform, quat_to_matrix  # noqa: E402

from pnp_core.catalog import load_robot as _load_robot  # noqa: E402
BASE = _load_robot()["frames"]["base"]     # the active cell's robot base frame


def img_to_np(m):
    dt = {"rgb8": np.uint8, "bgr8": np.uint8, "mono8": np.uint8, "16UC1": np.uint16, "32FC1": np.float32}[m.encoding]
    a = np.frombuffer(m.data, dt).reshape(m.height, m.width, -1).squeeze()
    if m.encoding == "bgr8":
        a = a[..., ::-1]
    if m.encoding == "16UC1":
        a = a.astype(np.float32) / 1000.0
    return a.copy()


def rot_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--object", default="tomato_soup_can")
    ap.add_argument("--swap-outputs", action="store_true")
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--profile", action="store_true", help="time the pipeline stages")
    a = ap.parse_args()

    from pnp_perception import foundationpose_onnx as F
    if a.swap_outputs:
        F.CFG["onnx"]["refine_outputs"] = list(reversed(F.CFG["onnx"]["refine_outputs"]))
    t0 = time.time()
    est = F.make_estimator(a.object)
    if a.profile:                                     # time each stage (GPU-synchronized), printed per frame
        import torch
        import estimater as E
        import learning.training.predict_pose_refine as PR
        import learning.training.predict_score as PS
        prof = {}

        def timed(name, fn):
            def w(*args, **kw):
                torch.cuda.synchronize()
                t = time.time()
                out = fn(*args, **kw)
                torch.cuda.synchronize()
                prof[name] = prof.get(name, 0.0) + time.time() - t
                return out
            return w
        est.refiner.model._run = timed("refine network", est.refiner.model._run)
        est.scorer.model._run = timed("score network", est.scorer.model._run)
        PR.make_crop_data_batch = timed("refine: render + crops", PR.make_crop_data_batch)
        PS.make_crop_data_batch = timed("score: render + crops", PS.make_crop_data_batch)
        E.erode_depth = timed("depth erode", E.erode_depth)
        E.bilateral_filter_depth = timed("depth bilateral", E.bilateral_filter_depth)
        E.FoundationPose.guess_translation = timed("guess translation", E.FoundationPose.guess_translation)
    print(f"estimator ready in {time.time() - t0:.1f} s (refine outputs = {F.CFG['onnx']['refine_outputs']})", flush=True)
    up_axis = 1                                       # the YCB mesh is Y-up: the can's axis is the mesh's y axis

    rclpy.init()
    n = rclpy.create_node("pnp_fp_eval")
    tf = Buffer()
    TransformListener(tf, n)
    latest = {}
    for topic, typ, key in (("/camera/camera/color/image_raw", Image, "rgb"),
                            ("/camera/camera/aligned_depth_to_color/image_raw", Image, "depth"),
                            ("/camera/camera/color/camera_info", CameraInfo, "info"),
                            ("/perception/mask", Image, "mask")):
        n.create_subscription(typ, topic, lambda m, k=key: latest.__setitem__(k, m), qos_profile_sensor_data)

    def T(parent, child):
        t = tf.lookup_transform(parent, child, Time()).transform
        return make_transform(quat_to_matrix([t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]),
                              [t.translation.x, t.translation.y, t.translation.z])

    rows = []
    for k in range(a.frames):
        latest.clear()
        end = time.time() + 10
        while time.time() < end and not (len(latest) == 4 and
                                         abs(Time.from_msg(latest["mask"].header.stamp).nanoseconds -
                                             Time.from_msg(latest["depth"].header.stamp).nanoseconds) < 5e7):
            rclpy.spin_once(n, timeout_sec=0.05)
        if len(latest) < 4:
            print("no synchronized color/depth/mask (is the detector publishing /perception/mask?)")
            return 1
        rgb, depth, mask = img_to_np(latest["rgb"]), img_to_np(latest["depth"]), img_to_np(latest["mask"])
        K = np.array(latest["info"].k).reshape(3, 3)
        if not (mask > 0).any():
            print(f"[{k}] detector found no can in this frame")
            continue
        end = time.time() + 10                        # the latched static camera TF can arrive after the images
        while time.time() < end and not (tf.can_transform(BASE, "camera_color_optical_frame", Time()) and
                                         tf.can_transform(BASE, "target", Time())):
            rclpy.spin_once(n, timeout_sec=0.05)
        T_gt = np.linalg.inv(T(BASE, "camera_color_optical_frame")) @ T(BASE, "target")
        t1 = time.time()
        pose = est.register(K=K, rgb=rgb, depth=depth.astype(np.float32), ob_mask=(mask == 1).astype(np.uint8) * 255,
                            iteration=a.iterations)
        dt = time.time() - t1
        if a.profile:
            print("    " + ", ".join(f"{k} {v * 1000:.0f} ms" for k, v in sorted(prof.items(), key=lambda kv: -kv[1])) +
                  f", other {(dt - sum(prof.values())) * 1000:.0f} ms", flush=True)
            prof.clear()
        terr = np.linalg.norm(pose[:3, 3] - T_gt[:3, 3]) * 1000
        axis_err = np.degrees(np.arccos(np.clip(abs(pose[:3, up_axis] @ T_gt[:3, up_axis]), -1, 1)))
        rerr = rot_angle_deg(pose[:3, :3].T @ T_gt[:3, :3])
        rows.append((terr, axis_err, rerr, dt))
        print(f"[{k}] translation err {terr:6.1f} mm   axis err {axis_err:5.1f} deg   full rot err {rerr:6.1f} deg   "
              f"{dt * 1000:5.0f} ms", flush=True)
    if rows:
        r = np.array(rows)
        print(f"MEAN translation {r[:, 0].mean():.1f} mm, axis {r[:, 1].mean():.1f} deg, rotation {r[:, 2].mean():.1f} deg, "
              f"time {r[1:, 3].mean() * 1000 if len(r) > 1 else r[0, 3] * 1000:.0f} ms (first frame excluded: warm-up)")
    rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
