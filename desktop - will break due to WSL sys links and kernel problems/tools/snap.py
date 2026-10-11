"""Save one color frame from each camera (no motion): to check a mount, an angle, what's in view.
    python3 /repo/tools/snap.py [--out /repo/data/jetson/snap] [--topics /camera/camera /wrist_camera/camera]
-> <out>_<camera>.png, and a side-by-side <out>.png
"""
import argparse
import sys
import time

import cv2
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


def to_bgr(msg: Image) -> np.ndarray:
    a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
    return a[..., ::-1].copy() if msg.encoding == "rgb8" else a.copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/repo/data/jetson/snap")
    ap.add_argument("--topics", nargs="+", default=["/camera/camera", "/wrist_camera/camera"])
    ap.add_argument("--timeout", type=float, default=8.0)
    a = ap.parse_args()
    rclpy.init()
    n = rclpy.create_node("pnp_snap")
    got = {}
    for t in a.topics:
        n.create_subscription(Image, t + "/color/image_raw", lambda m, t=t: got.setdefault(t, m), qos_profile_sensor_data)
    t0 = time.time()
    while len(got) < len(a.topics) and time.time() - t0 < a.timeout:
        rclpy.spin_once(n, timeout_sec=0.1)
    imgs = []
    for t in a.topics:
        if t not in got:
            print(f"{t}: no image within {a.timeout:.0f} s")
            continue
        im = to_bgr(got[t])
        name = t.strip("/").split("/")[0]
        cv2.imwrite(f"{a.out}_{name}.png", im)
        print(f"{t}: {im.shape[1]}x{im.shape[0]} -> {a.out}_{name}.png")
        imgs.append(cv2.resize(im, (640, int(640 * im.shape[0] / im.shape[1]))))
    if imgs:
        h = max(i.shape[0] for i in imgs)
        cv2.imwrite(f"{a.out}.png", np.hstack([np.pad(i, ((0, h - i.shape[0]), (0, 0), (0, 0))) for i in imgs]))
    rclpy.shutdown()
    return 0 if len(imgs) == len(a.topics) else 1


if __name__ == "__main__":
    sys.exit(main())
