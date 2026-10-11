"""Record one pick & place as an animated GIF: the overhead camera and the wrist camera side by side, with the task
state written on each frame, plus where the object ended up vs the goal (sim ground truth).

    python3 /repo/tools/record_pick.py                       -> data/pick_place.gif (+ data/pick_place_<state>.png)
    python3 /repo/tools/record_pick.py --randomize --obstacles --out data/pick2.gif

Uses whatever stack is running (sim, any robot / scene). The frames are the simulated cameras' own images (what
the robot sees), so the GIF shows the grasp from both of its viewpoints.
"""

import argparse
import sys
import time

import cv2
import numpy as np
from PIL import Image as PILImage
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

sys.path.insert(0, "/repo/tools")
from run_benchmark import Bench  # noqa: E402
import rclpy  # noqa: E402

sys.path.insert(0, "/repo")
from pnp_core.catalog import load_robot  # noqa: E402
from pnp_core.wrist import wrist_camera  # noqa: E402


def to_bgr(m):
    a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)[:, :, :3]
    return cv2.cvtColor(a, cv2.COLOR_RGB2BGR) if m.encoding.lower().startswith("rgb") else a.copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/pick_place.gif")
    ap.add_argument("--randomize", action="store_true", help="new random object + goal positions first")
    ap.add_argument("--obstacles", action="store_true", help="random obstacle boxes too")
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--profile", default="dev")
    a = ap.parse_args()

    rclpy.init()
    b = Bench(name="pnp_record_pick")
    scene, obj = b.scene()
    wcfg = wrist_camera(load_robot(scene.get("robot", "fr3")))
    latest = {}
    b.create_subscription(Image, "/camera/camera/color/image_raw", lambda m: latest.__setitem__("top", m),
                          qos_profile_sensor_data)
    if wcfg:
        b.create_subscription(Image, wcfg["topics"]["color_image"], lambda m: latest.__setitem__("wrist", m),
                              qos_profile_sensor_data)
    b.sim.wait_ready(timeout=5.0)
    if a.randomize:
        b.sim.call("randomize_scene")
    if a.obstacles:
        b.sim.call("randomize_obstacles")
    b.spin_for(2.0)
    goal = b.sim.call("state")["goal"]["xy"]
    frames, keys, last = [], {}, 0.0

    def grab():
        nonlocal last
        if time.time() - last < 1.0 / a.fps or "top" not in latest:
            return
        last = time.time()
        state = b.states[-1][0] if b.states else "START"
        top = cv2.resize(to_bgr(latest["top"]), (480, 360))
        panels = [top]
        if "wrist" in latest:
            panels.append(cv2.resize(to_bgr(latest["wrist"]), (640, 360)))
        f = np.hstack(panels)
        cv2.rectangle(f, (0, 0), (f.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(f, f"{state}   (left: overhead D435i   right: wrist D405)", (8, 21), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
        frames.append(f)
        keys[state] = f

    # Bench.pick_place spins the node itself; grab frames from a timer on the same node
    timer = b.create_timer(0.05, grab)
    label = obj["yolo_class"]
    print(f"pick & place: {scene['target']['object']} -> goal {np.round(goal, 3).tolist()} ...", flush=True)
    res = b.pick_place(goal, label, a.profile)
    b.spin_for(1.5)                                   # a few frames of the result
    timer.cancel()
    outcome = (res.outcome_name or str(res.outcome)) if res is not None else "TIMEOUT"
    gt, _ = b.gt_pose("target")
    err = float(np.hypot(gt[0] - goal[0], gt[1] - goal[1]))
    print(f"outcome {outcome}; object now at {np.round(gt[:2], 4).tolist()}, {err * 1000:.1f} mm from the goal; "
          f"{len(frames)} frames", flush=True)
    out = a.out if a.out.startswith("/") else f"/repo/{a.out}"
    imgs = [PILImage.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).resize((f.shape[1] * 3 // 4, f.shape[0] * 3 // 4))
            for f in frames]
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=int(1000 / a.fps), loop=0, optimize=True)
    print(f"saved {out}")
    for st in ("LIFT", "MOVE_PREPLACE", "RETREAT"):
        if st in keys:
            p = out.rsplit(".", 1)[0] + f"_{st.lower()}.png"
            cv2.imwrite(p, keys[st])
            print(f"saved {p}")
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
