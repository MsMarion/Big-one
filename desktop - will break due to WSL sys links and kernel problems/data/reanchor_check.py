"""Tier 0.2 validation (sim): the drift monitor + its AprilTag re-anchor, against deliberate camera bumps.
  1. small bump (+3 mm, the sim command camera_error): the monitor re-anchors -> the believed camera (TF) is back within
     ~1 mm / 0.1 deg of the TRUE one (sim camera_truth); an event in the trial log
  2. the camera back where it was: it re-anchors back
  3. big bump (+20 mm): no re-anchor (reanchor_refused event), DRIFT; with calibration_policy stop a pick refuses
     (CALIBRATION_DRIFT)
  4. restored: the camera back, the original calibration file restored, policies back to their defaults
Needs: the so101_plate_tags scene with the tag set, a calibration with a tag reference (calibrate.py locate +
reference), the realistic profile (its camera error is the starting point).
    docker exec pnp_dev bash /repo/data/rospy.sh /repo/data/reanchor_check.py
"""
import json
import shutil
import subprocess
import sys
import time
import urllib.request

import numpy as np
import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

sys.path.insert(0, "/repo")
sys.path.insert(0, "/repo/tools")
from pnp_core import trials  # noqa: E402
from pnp_core.calibration import pose_error  # noqa: E402
from pnp_core.catalog import calibration_dir, load_yaml  # noqa: E402
from pnp_core.geometry import make_transform, quat_to_matrix  # noqa: E402
from sim_client import SimClient  # noqa: E402

rclpy.init()
node = rclpy.create_node("reanchor_check")
sim = SimClient(node)
tf = Buffer()
TransformListener(tf, node)
status = {}
q = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
node.create_subscription(String, "/calibration/status", lambda m: status.update(json.loads(m.data)), q)
BASE, OPT = "base_link", "camera_color_optical_frame"
REAL = load_yaml("config/sim_reality.yaml")["profiles"]["realistic"]["fixed_camera_error"]


def spin(s):
    end = time.time() + s
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.05)


def believed():
    spin(0.3)
    end = time.time() + 10
    while not tf.can_transform(BASE, OPT, rclpy.time.Time()) and time.time() < end:
        spin(0.2)
    t = tf.lookup_transform(BASE, OPT, rclpy.time.Time()).transform
    return make_transform(quat_to_matrix([t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]),
                          [t.translation.x, t.translation.y, t.translation.z])


def err():
    T_true = np.array(sim.call("camera_truth")["cameras"]["camera"])
    return pose_error(believed(), T_true)


def param(node_name, name, value):
    r = subprocess.run(["ros2", "param", "set", node_name, name, str(value)], capture_output=True, text=True)
    return r.stdout.strip()


def bump(dx_mm):
    xyz = list(REAL["xyz_mm"])
    xyz[0] += dx_mm
    sim.call("camera_error", xyz_mm=xyz, rpy_deg=REAL["rpy_deg"])


def wait_event(kind, t0, timeout):
    end = time.time() + timeout
    while time.time() < end:
        spin(0.5)
        c = trials.connect()
        r = c.execute("SELECT * FROM events WHERE kind = ? AND t > ? ORDER BY id DESC LIMIT 1", (kind, t0)).fetchone()
        c.close()
        if r:
            return json.loads(r["data"])
    return None


def api(path, body=None):
    req = urllib.request.Request("http://localhost:8088" + path, json.dumps(body).encode() if body is not None else None,
                                 {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=10).read())


def main():
    spin(2.0)
    cal = calibration_dir() / f"{status.get('set', 'sim_realistic')}_d435i.yaml"
    backup = cal.with_suffix(".yaml.bak_reanchor_check")
    shutil.copy(cal, backup)
    e0 = err()
    print(f"start: monitor {status.get('state')} ({status.get('px')} px); believed vs TRUE {e0[0]:.2f} mm / {e0[1]:.3f} deg")
    print(param("/pnp_calibration_monitor", "reanchor", "auto"))
    try:
        for label, dx in (("small bump +3 mm", 3.0), ("back to where it was", 0.0)):
            t0 = time.time()
            bump(dx)
            spin(3.0)
            print(f"{label}: monitor {status.get('state')} ({status.get('px')} px), believed vs TRUE "
                  f"{err()[0]:.2f} mm / {err()[1]:.3f} deg before the re-anchor")
            ev = wait_event("reanchor", t0, 40)
            spin(2.0)
            e = err()
            print(f"  -> re-anchored: {ev}" if ev else "  -> NO re-anchor within 40 s")
            print(f"  -> believed vs TRUE now {e[0]:.2f} mm / {e[1]:.3f} deg; monitor {status.get('state')}")
        t0 = time.time()
        bump(20.0)
        ev = wait_event("reanchor_refused", t0, 30)
        spin(6.0)
        print(f"big bump +20 mm: refused {ev}; monitor {status.get('state')} ({status.get('mm')} mm)")
        print(param("/pnp_pick_place", "calibration_policy", "stop"))
        w = api("/api/world")
        if w.get("objects"):
            o = w["objects"][0]
            api("/api/pick", {"id": o["id"], "xy": [0.2, -0.06], "yaw_deg": 0})
            for _ in range(60):
                spin(0.5)
                w = api("/api/world")
                if not w["task"]["busy"]:
                    break
            print(f"  a pick under calibration_policy stop: {w['task'].get('result')}")
        else:
            print("  (no scanned object to try a pick with)")
    finally:
        print(param("/pnp_calibration_monitor", "reanchor", "off"))
        print(param("/pnp_pick_place", "calibration_policy", "warn"))
        bump(0.0)
        shutil.copy(backup, cal)
        backup.unlink()
        subprocess.run(["ros2", "service", "call", "/calibration/reload", "std_srvs/srv/Trigger"], capture_output=True)
        spin(2.0)
        e = err()
        print(f"restored: the camera at the realistic error, {cal.name} restored; believed vs TRUE {e[0]:.2f} mm / "
              f"{e[1]:.3f} deg")


if __name__ == "__main__":
    main()
