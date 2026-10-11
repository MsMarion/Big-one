"""Live fixed-camera (D435i) view in a browser, with the AprilTags it finds: for aiming the camera / placing tags.

    docker exec pnp_dev bash /repo/data/rospy.sh /repo/data/tag_live_view.py        then open http://localhost:8088 (the workcell page's port: not while it runs)

Each detected tag: a green outline, its id, a red dot on its TOP-LEFT corner (it should be the tag's top-right in the
image: TOP edge toward the robot's left). Banner: how many of the layout's tags are seen, which are missing, the
camera's tilt (from the tags: no calibration needed). An orange band marks the 20 px next to the image edges: a tag
touching it is about to fall out. No motion, nothing written. Ctrl+C stops it.
"""
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

sys.path.insert(0, "/repo")
from pnp_core.calibration import detect_tags  # noqa: E402
from pnp_core.catalog import calibration_targets_path, load_yaml  # noqa: E402

# the container publishes only 7447 (zenoh) and 8088 (the workcell page) to Windows: 8088 while no workcell runs
PORT = int(os.environ.get("PNP_VIEW_PORT", "8088"))
A = load_yaml(calibration_targets_path())["apriltags"]
WANT = sorted(int(t["id"]) for t in A["tags"])
latest = {"jpg": None}
lock = threading.Condition()


def annotate(m, info):
    ch = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}[m.encoding]
    img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)[:, :m.width * ch].reshape(m.height, m.width, ch)[:, :, :3]
    vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if m.encoding.startswith("rgb") else img.copy()
    K = np.array(info.k).reshape(3, 3)
    tags = {k: v for k, v in detect_tags(cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY), A["dictionary"], A["size_m"], K, None,
                                         with_corners=True).items() if k in WANT}
    h, w = vis.shape[:2]
    band = vis.copy()
    cv2.rectangle(band, (0, 0), (w - 1, h - 1), (0, 140, 255), 40)
    vis = cv2.addWeighted(band, 0.35, vis, 0.65, 0)
    for tid, (T, c) in tags.items():
        cv2.polylines(vis, [c.astype(np.int32)], True, (0, 255, 0), 2)
        cv2.circle(vis, tuple(c[0].astype(int)), 5, (0, 0, 255), -1)
        cv2.putText(vis, str(tid), tuple(c.mean(axis=0).astype(int) + [-8, 8]), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255), 2)
    missing = [t for t in WANT if t not in tags]
    tilt = (f", tilt {np.median([np.degrees(np.arccos(np.clip(T[2, 2], -1, 1))) for T, _ in tags.values()]):.1f} deg"
            if tags else "")
    text = ("ALL 9 IN VIEW" if not missing else f"{len(tags)}/9  missing {missing}") + tilt
    cv2.rectangle(vis, (0, 0), (w, 30), (0, 0, 0), -1)
    cv2.putText(vis, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0) if not missing else (0, 200, 255), 2)
    return cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/stream"):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with lock:
                        lock.wait(timeout=2.0)
                        jpg = latest["jpg"]
                    if jpg:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                return
        else:
            body = (b"<html><head><title>Tag view</title></head><body style='margin:0;background:#111'>"
                    b"<img src='/stream' style='width:100%;max-width:1272px'></body></html>")
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(body)


def main():
    rclpy.init()
    node = rclpy.create_node("tag_live_view")
    got = {}
    node.create_subscription(Image, "/camera/camera/color/image_raw", lambda m: got.__setitem__("img", m),
                             qos_profile_sensor_data)
    node.create_subscription(CameraInfo, "/camera/camera/color/camera_info", lambda m: got.__setitem__("info", m),
                             qos_profile_sensor_data)
    threading.Thread(target=ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever, daemon=True).start()
    print(f"live tag view: http://localhost:{PORT}  (Ctrl+C stops)", flush=True)
    last = None
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.05)
        m = got.get("img")
        if m is None or m is last or "info" not in got:
            continue
        last = m
        jpg = annotate(m, got["info"])
        with lock:
            latest["jpg"] = jpg
            lock.notify_all()
        time.sleep(0.1)                                   # ~7 fps is plenty for aiming


if __name__ == "__main__":
    main()
