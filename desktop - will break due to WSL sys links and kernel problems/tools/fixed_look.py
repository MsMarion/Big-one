"""What does the FIXED camera (D435i) see on the desk? One RGB-D frame -> the vision pipeline (config/vision.yaml
profile, default geometry: depth blobs + catalog size) -> each object's name, base-frame xy, size. No motion.
    python3 /repo/tools/fixed_look.py [--region 0.08 0.40 -0.25 0.25] [--profile geometry] [--frames 5]
Saves <repo>/data/jetson/fixed_look.png (the masks + names drawn on the color image).
"""
import argparse
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener

sys.path.insert(0, "/repo")
from pnp_core.catalog import load_robot, load_scene, load_yaml  # noqa: E402
from pnp_core.geometry import make_transform, quat_to_matrix  # noqa: E402
from pnp_core.vision import Frame, build_pipeline  # noqa: E402

CAM = "/camera/camera"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", type=float, nargs=4, default=[0.08, 0.40, -0.25, 0.25], help="x0 x1 y0 y1 (m)")
    ap.add_argument("--profile", default="geometry")
    ap.add_argument("--frames", type=int, default=5, help="median depth over this many frames (less noise)")
    ap.add_argument("--out", default="/repo/data/jetson/fixed_look.png")
    a = ap.parse_args()
    scene = load_scene()
    table_z = scene["table"]["center"][2] + scene["table"]["size"][2] / 2
    base = load_robot()["frames"]["base"]
    rclpy.init()
    n = rclpy.create_node("pnp_fixed_look")
    ex = MultiThreadedExecutor()
    ex.add_node(n)
    threading.Thread(target=ex.spin, daemon=True).start()
    tfb = Buffer()
    TransformListener(tfb, n)
    got = {"depth": []}
    n.create_subscription(Image, CAM + "/color/image_raw", lambda m: got.__setitem__("rgb", m), qos_profile_sensor_data)
    n.create_subscription(Image, CAM + "/aligned_depth_to_color/image_raw", lambda m: got["depth"].append(m),
                          qos_profile_sensor_data)
    n.create_subscription(CameraInfo, CAM + "/color/camera_info", lambda m: got.__setitem__("info", m),
                          qos_profile_sensor_data)
    end = time.time() + 10
    while (len(got["depth"]) < a.frames or "rgb" not in got or "info" not in got) and time.time() < end:
        time.sleep(0.05)
    if "rgb" not in got or not got["depth"]:
        print("no frames from the fixed camera")
        return 1
    m = got["rgb"]
    rgb = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
    rgb = rgb if m.encoding == "rgb8" else rgb[..., ::-1]
    def meters(d):       # the real D435i: 16UC1 mm; the sim (twin): 32FC1 m
        if d.encoding == "32FC1":
            return np.frombuffer(d.data, np.float32).reshape(d.height, d.width).astype(np.float64)
        return np.frombuffer(d.data, np.uint16).reshape(d.height, d.width).astype(np.float64) / 1000.0
    st = np.stack([meters(d) for d in got["depth"][-a.frames:]])
    st[~np.isfinite(st)] = 0
    st[st == 0] = np.nan
    depth = np.nan_to_num(np.nanmedian(st, axis=0), nan=0.0)
    t = tfb.lookup_transform(base, m.header.frame_id, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=3))
    tr, q = t.transform.translation, t.transform.rotation
    T = make_transform(quat_to_matrix([q.x, q.y, q.z, q.w]), [tr.x, tr.y, tr.z])
    K = np.array(got["info"].k).reshape(3, 3)
    pipe = build_pipeline(load_yaml("config/vision.yaml")["profiles"][a.profile], load_yaml("config/objects.yaml"),
                          table_z, tuple(a.region))
    est = pipe.run(Frame(rgb.copy(), depth, K, T, camera="d435i"))
    vis = rgb[..., ::-1].copy()
    for e in est:
        p = e.proposal
        print(f"{e.identity.name:10s} at ({p.center_xy[0]:.3f}, {p.center_xy[1]:.3f}) m, "
              f"{p.footprint[0] * 1000:.0f} x {p.footprint[1] * 1000:.0f} x {p.height * 1000:.0f} mm, "
              f"yaw {np.degrees(p.yaw):.0f} deg")
        if p.mask is not None:
            cs, _ = cv2.findContours(p.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cs, -1, (0, 255, 0), 1)
            if cs:
                x, y, _, _ = cv2.boundingRect(max(cs, key=cv2.contourArea))
                cv2.putText(vis, e.identity.name.replace("lego_", ""), (x, y - 4), 0, 0.5, (0, 255, 0), 1)
    if not est:
        print("nothing found in the region", a.region)
    cv2.imwrite(a.out, vis)
    print("image:", a.out)
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
