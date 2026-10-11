"""Scan the desk with the wrist camera (/wrist_map/scan) and compare the flat map's objects with the sim's ground
truth: position error, identity, face. Saves the top view to data/wrist_map_topview.png.

    bash /repo/data/rospy.sh /repo/data/wrist_map_check.py [--randomize]
"""
import json
import math
import sys
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger

sys.path.insert(0, "/repo/tools")
sys.path.insert(0, "/repo")
from sim_client import SimClient  # noqa: E402
from pnp_core.catalog import load_scene  # noqa: E402

rclpy.init()
n = rclpy.create_node("wrist_map_check")
sim = SimClient(n)
if "--randomize" in sys.argv:
    sim.call("randomize_scene")
    time.sleep(2.0)
latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
got = {}
n.create_subscription(String, "/wrist_map/objects", lambda m: got.__setitem__("obj", json.loads(m.data)), latched)
n.create_subscription(Image, "/wrist_map/topview", lambda m: got.__setitem__("img", m), latched)
cli = n.create_client(Trigger, "/wrist_map/scan")
cli.wait_for_service(timeout_sec=10.0)
t_req = time.time()
fut = cli.call_async(Trigger.Request())
while not fut.done():
    rclpy.spin_once(n, timeout_sec=0.1)
print("scan:", fut.result().message)
end = time.time() + 5
while time.time() < end and (got.get("obj", {}).get("time", 0) < t_req or "img" not in got):
    rclpy.spin_once(n, timeout_sec=0.1)
scene = load_scene()
name_of = {"target": scene["target"]["object"], **{e["name"]: e["object"] for e in scene.get("extra_objects", [])}}
truth = {name_of[k]: v for k, v in sim.call("state")["objects"].items() if k in name_of}
objs = got["obj"]["objects"]
print(f"coverage {got['obj']['coverage'] * 100:.0f}%, {len(objs)} objects in the map, {len(truth)} on the desk")
errs = []
for name, xy in truth.items():
    near = min(objs, key=lambda o: math.hypot(o["xy"][0] - xy[0], o["xy"][1] - xy[1]), default=None)
    if near is None or math.hypot(near["xy"][0] - xy[0], near["xy"][1] - xy[1]) > 0.03:
        print(f"  {name:10s} at ({xy[0]:.3f}, {xy[1]:.3f}): NOT in the map")
        continue
    e = math.hypot(near["xy"][0] - xy[0], near["xy"][1] - xy[1])
    errs.append(e)
    ok = "OK" if near.get("object") == name else f"WRONG ({near.get('object')})"
    print(f"  {name:10s} at ({xy[0]:.3f}, {xy[1]:.3f}): map ({near['xy'][0]:.3f}, {near['xy'][1]:.3f}) "
          f"{e * 1000:4.1f} mm off, {near['footprint'][0] * 1000:.1f} x {near['footprint'][1] * 1000:.1f} x "
          f"{near['height'] * 1000:.1f} mm, identity {ok}, face {near.get('rest')}")
extra = [o for o in objs if all(math.hypot(o["xy"][0] - xy[0], o["xy"][1] - xy[1]) > 0.03 for xy in truth.values())]
for o in extra:
    print(f"  extra blob at ({o['xy'][0]:.3f}, {o['xy'][1]:.3f}) {o['footprint'][0] * 1000:.0f} x "
          f"{o['footprint'][1] * 1000:.0f} x {o['height'] * 1000:.0f} mm ({o.get('object', 'unmatched')})")
if errs:
    print(f"position error: mean {np.mean(errs) * 1000:.1f} mm, max {np.max(errs) * 1000:.1f} mm")
img = CvBridge().imgmsg_to_cv2(got["img"], "bgr8")
cv2.imwrite("/repo/data/wrist_map_topview.png", cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
print("saved data/wrist_map_topview.png")
rclpy.shutdown()
