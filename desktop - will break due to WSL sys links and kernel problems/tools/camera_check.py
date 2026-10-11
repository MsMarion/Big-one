"""Is the camera stream healthy? Works the same for the sim, a local D435i and a Jetson-streamed D435i.

    python3 /repo/tools/camera_check.py [--seconds 5]

Per topic (config/interfaces.yaml + the compressed versions when they exist):
  rate [Hz], bandwidth [Mbit/s], image size/encoding, and LATENCY = receive time - header stamp.
Latency only means something when the sender's clock is synced with this PC (real cameras: chrony/NTP on the
Jetson; see docs/learning/08_real_sensors.md). In the sim both sides use /clock, so it is skipped.
Depth: fraction of valid pixels (a real D435i has holes; < 50 % valid usually means bad exposure/range).

Exit code 0 when color + depth + camera_info arrive at >= 5 Hz.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, CompressedImage, Image

REPO = Path(__file__).resolve().parents[1]


class Stat:
    def __init__(self):
        self.n, self.bytes, self.lat, self.first, self.last, self.msg = 0, 0, [], None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=5.0)
    a = ap.parse_args()
    topics = yaml.safe_load((REPO / "config/interfaces.yaml").read_text())["topics"]
    color, depth, info = topics["color_image"]["name"], topics["aligned_depth"]["name"], topics["color_info"]["name"]
    subs = {color: Image, depth: Image, info: CameraInfo,
            color + "/compressed": CompressedImage, depth + "/compressedDepth": CompressedImage}
    rclpy.init()
    node = rclpy.create_node("pnp_camera_check")
    use_sim = any(n == "/clock" for n, _ in node.get_topic_names_and_types())
    stats = {t: Stat() for t in subs}

    def cb(topic):
        def f(msg):
            s, now = stats[topic], time.time()
            s.n += 1
            s.bytes += len(msg.data) if hasattr(msg, "data") else 0
            s.first = s.first or now
            s.last, s.msg = now, msg
            if not use_sim:
                s.lat.append(now - Time.from_msg(msg.header.stamp).nanoseconds * 1e-9)
        return f

    for t, typ in subs.items():
        node.create_subscription(typ, t, cb(t), qos_profile_sensor_data)
    end = time.time() + a.seconds
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.05)

    ok = True
    print(f"{'topic':62s} {'Hz':>6s} {'Mbit/s':>7s} {'latency ms':>11s}  details")
    for t, s in stats.items():
        required = t in (color, depth, info)
        if s.n < 2:
            print(f"{t:62s} {'-':>6s} {'-':>7s} {'-':>11s}  {'MISSING' if required else '(not published)'}")
            ok &= not required
            continue
        dt = max(s.last - s.first, 1e-6)
        hz, mbit = (s.n - 1) / dt, s.bytes * 8 / dt / 1e6
        if not s.lat:
            lat = "  (sim)"
        elif abs(np.median(s.lat)) > 60:          # stamps aren't this PC's wall clock (bag replay / unsynced clock)
            lat = " (clock?)"
        else:
            lat = f"{np.median(s.lat) * 1000:7.0f}"
        detail = ""
        m = s.msg
        if isinstance(m, Image):
            detail = f"{m.width}x{m.height} {m.encoding} frame={m.header.frame_id}"
            if t == depth:
                d = np.frombuffer(m.data, dtype=np.uint16 if m.encoding == "16UC1" else np.float32)
                detail += f" valid={np.mean((d > 0) & np.isfinite(d)) * 100:.0f}%"
        elif isinstance(m, CompressedImage):
            detail = f"{m.format}  {s.bytes / s.n / 1024:.0f} KB/frame"
        elif isinstance(m, CameraInfo):
            detail = f"fx={m.k[0]:.1f} fy={m.k[4]:.1f} cx={m.k[2]:.1f} cy={m.k[5]:.1f}"
        print(f"{t:62s} {hz:6.1f} {mbit:7.1f} {lat:>11s}  {detail}")
        ok &= (hz >= 5.0) or not required
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    rclpy.try_shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
