"""MoveIt's allowed collision matrix as move_group has it now (which robot link pairs may touch). No motion.
    docker exec pnp_dev bash /repo/data/realpy.sh /repo/data/acm_dump.py
"""
import time

import rclpy
from moveit_msgs.msg import PlanningSceneComponents
from moveit_msgs.srv import GetPlanningScene

rclpy.init()
n = rclpy.create_node("acm_dump")
c = n.create_client(GetPlanningScene, "/get_planning_scene")
if not c.wait_for_service(timeout_sec=10):
    raise SystemExit("/get_planning_scene not reachable")
req = GetPlanningScene.Request()
req.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX | PlanningSceneComponents.WORLD_OBJECT_NAMES
f = c.call_async(req)
end = time.time() + 10
while not f.done() and time.time() < end:
    rclpy.spin_once(n, timeout_sec=0.1)
s = f.result().scene
acm = s.allowed_collision_matrix
names = list(acm.entry_names)
print(f"{len(names)} entries, {len(acm.default_entry_names)} defaults; world objects: "
      f"{[o.id for o in s.world.collision_objects]}")
allowed = sum(acm.entry_values[i].enabled[j] for i in range(len(names)) for j in range(i + 1, len(names))
              if j < len(acm.entry_values[i].enabled))
print(f"allowed pairs: {allowed}")
for a, b in [("base_link", "shoulder_link"), ("gripper_link", "moving_jaw_so101_v1_link"), ("upper_arm_link", "lower_arm_link")]:
    if a in names and b in names:
        print(a, b, acm.entry_values[names.index(a)].enabled[names.index(b)])
    else:
        print(a, b, "not in the matrix")
print(names)
