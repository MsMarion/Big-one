"""The no-motion wrist map (/wrist_map/look) in the sim: put the arm in ONE looking-down pose (the best single view of
pnp_core/wrist_scan.choose_views), ask for a look, compare the objects with the sim's truth (the ones in view).
On the real arm the pose comes from your hands instead (docs/learning/22_jetson_d405.md).

    bash /repo/data/rospy.sh /repo/data/wrist_look_check.py
"""
import json
import math
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener

sys.path.insert(0, "/repo/tools")
from sim_client import SimClient  # noqa: E402
from pnp_core.catalog import load_robot, load_scene  # noqa: E402
from pnp_core.geometry import make_transform, quat_to_matrix  # noqa: E402
from pnp_core.wrist import wrist_camera  # noqa: E402
from pnp_core.wrist_scan import choose_views  # noqa: E402
from pnp_manipulation.robot_interface import RobotInterface  # noqa: E402

rclpy.init()
n = rclpy.create_node("wrist_look_check")
cfg, scene = load_robot(), load_scene()
wc = wrist_camera(cfg)
# the sim's truth first: SimClient spins the node itself, so before the executor takes it over
name_of = {"target": scene["target"]["object"], **{e["name"]: e["object"] for e in scene.get("extra_objects", [])}}
truth = {name_of[k]: v for k, v in SimClient(n).call("state")["objects"].items() if k in name_of}
ex = MultiThreadedExecutor(num_threads=4)
ex.add_node(n)
threading.Thread(target=ex.spin, daemon=True).start()
tf = Buffer()
TransformListener(tf, n)
info = {}
n.create_subscription(CameraInfo, wc["topics"]["color_info"], lambda m: info.__setitem__("k", m),
                       qos_profile_sensor_data)
got = {}
latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
n.create_subscription(String, "/wrist_map/objects", lambda m: got.__setitem__("obj", json.loads(m.data)), latched)
R = RobotInterface(n, cfg)
assert R.wait_ready(timeout=20.0), "robot not ready"
end = time.time() + 10
while ("k" not in info or R.ik_chain is None) and time.time() < end:
    time.sleep(0.1)
t = tf.lookup_transform(cfg["frames"]["tcp"], wc["frames"]["color_optical"], rclpy.time.Time(),
                        timeout=rclpy.duration.Duration(seconds=5)).transform
T_tcp_opt = make_transform(quat_to_matrix([t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]),
                           [t.translation.x, t.translation.y, t.translation.z])
tb = scene["table"]
table_z = tb["center"][2] + tb["size"][2] / 2
reg = scene.get("obstacles", {}).get("region") or scene["target"]["spawn_region"]
region = (reg["x"][0], reg["x"][1], reg["y"][0], reg["y"][1])
ci, ch = info["k"], R.ik_chain
views, cov = choose_views(lambda q: ch.fk(q) @ T_tcp_opt, ch.lower, ch.upper, region, np.array(ci.k).reshape(3, 3),
                          (ci.width, ci.height), table_z, np.random.default_rng(3), dist=(0.15, 0.32),
                          keep=(0.0, 0.0, 1.0, 0.67), max_range=0.5, fk_tcp=ch.fk, max_views=1)
print(f"one view planned: covers {cov * 100:.0f}% of the region; joints {np.round(views[0], 2).tolist()} rad; moving there")
assert R.move_joints(views[0]) == 1, "move failed"
time.sleep(1.0)
cli = n.create_client(Trigger, "/wrist_map/look")
cli.wait_for_service(timeout_sec=10.0)
t_req = time.time()
f = cli.call_async(Trigger.Request())
while not f.done():
    time.sleep(0.05)
print("look:", f.result().message)
end = time.time() + 5
while time.time() < end and got.get("obj", {}).get("time", 0) < t_req:
    time.sleep(0.1)
objs, errs = got["obj"]["objects"], []
for name, xy in truth.items():
    near = min(objs, key=lambda o: math.hypot(o["xy"][0] - xy[0], o["xy"][1] - xy[1]), default=None)
    if near is None or math.hypot(near["xy"][0] - xy[0], near["xy"][1] - xy[1]) > 0.03:
        print(f"  {name:10s} at ({xy[0]:.3f}, {xy[1]:.3f}): not in this view")
        continue
    errs.append(math.hypot(near["xy"][0] - xy[0], near["xy"][1] - xy[1]))
    print(f"  {name:10s} at ({xy[0]:.3f}, {xy[1]:.3f}): map ({near['xy'][0]:.3f}, {near['xy'][1]:.3f}) "
          f"{errs[-1] * 1000:4.1f} mm off, identity {'OK' if near.get('object') == name else near.get('object')}")
if errs:
    print(f"{len(errs)} seen in one view, position error mean {np.mean(errs) * 1000:.1f} mm, max {np.max(errs) * 1000:.1f} mm")
R.move_joints(np.array(cfg["arm"]["home"], float))
ex.shutdown()
rclpy.try_shutdown()
