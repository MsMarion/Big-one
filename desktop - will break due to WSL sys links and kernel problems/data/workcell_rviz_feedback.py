"""Act like a person in RViz on the workcell's interactive markers (no clicking needed): click an object marker,
drag the goal disc to (x, y), then the goal's right-click menu "Pick & place". Then follow the task on the API.

    bash /repo/data/rospy.sh /repo/data/workcell_rviz_feedback.py soup_can 0.45 -0.20
"""
import json
import sys
import time
import urllib.request

import rclpy
from visualization_msgs.msg import InteractiveMarkerFeedback as F

cls, x, y = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
world = lambda: json.load(urllib.request.urlopen("http://localhost:8088/api/world"))  # noqa: E731
oid = next(o["id"] for o in world()["objects"] if o["class"] == cls)

rclpy.init()
n = rclpy.create_node("fake_rviz_user")
pub = n.create_publisher(F, "/workcell/feedback", 10)
time.sleep(1.0)


def send(marker, event, **kw):
    fb = F(client_id="/fake_rviz", marker_name=marker, event_type=event)
    fb.header.frame_id = "fr3_link0"
    for k, v in kw.items():
        setattr(fb, k, v)
    pub.publish(fb)
    time.sleep(0.5)


send(oid, F.BUTTON_CLICK)                                       # click the object (the node redraws the markers)
fb_pose = F().pose
fb_pose.position.x, fb_pose.position.y, fb_pose.orientation.w = x, y, 1.0
send("goal", F.POSE_UPDATE, pose=fb_pose)                       # drag ...
send("goal", F.MOUSE_UP, pose=fb_pose)                          # ... and let go
w = world()
print(f"selected {w['selected']} (want {oid}), goal {w['goal']}")
send("goal", F.MENU_SELECT, menu_entry_id=1, pose=fb_pose)      # right-click > Pick & place
time.sleep(2.0)
while world()["task"]["busy"]:
    time.sleep(2.0)
w = world()
print("\n".join(line for line in w["task"]["log"] if any(k in line for k in ("pick ", "DETECT]", "VERIFY_PLACE]"))))
print("result:", w["task"]["result"])
rclpy.shutdown()
