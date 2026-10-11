"""Make a pick FAIL on purpose (sim): start a pick from the workcell page API, and once the arm is on its way to the
brick, teleport the brick away. The grasp closes on nothing -> retries -> the task fails. Checks that the failure
lands in the trial log: the attempt row, its events (grip verdicts), and the failure ring-buffer snapshot linked in
artifacts (bringup bag_buffer:=true).

    docker exec pnp_dev bash /repo/data/rospy.sh /repo/data/force_grasp_fail.py [object=lego_2x2]
"""
import json
import sys
import time
import urllib.request

import rclpy

sys.path.insert(0, "/repo")
sys.path.insert(0, "/repo/tools")
from pnp_core import trials  # noqa: E402
from sim_client import SimClient  # noqa: E402

API = "http://localhost:8088"
NAME = sys.argv[1] if len(sys.argv) > 1 else "lego_2x2"


def get(path):
    return json.loads(urllib.request.urlopen(API + path, timeout=10).read())


def post(path, body):
    req = urllib.request.Request(API + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=10).read())


def main():
    rclpy.init()
    sim = SimClient(rclpy.create_node("force_grasp_fail"))
    post("/api/scan", {})
    time.sleep(1.0)
    while get("/api/world")["scan"]["busy"]:
        time.sleep(0.5)
    w = get("/api/world")
    o = next(x for x in w["objects"] if x["name"] == NAME)
    goal = [0.215, 0.025]
    print(f"picking {o['id']} {NAME} at {o['xy']} -> {goal}")
    print(post("/api/pick", {"id": o["id"], "xy": goal, "yaw_deg": 0}))
    t0, moved = time.time(), False
    while time.time() - t0 < 240:
        w = get("/api/world")
        st = w["task"].get("state")
        if not moved and st in ("MOVE_PREGRASP", "LOOK"):
            # the brick leaves while the arm is on its way: the grasp will close on nothing
            print(f"state {st}: teleporting {NAME} away")
            sim.call("place_object", name=NAME, xy=[0.33, -0.25], yaw=0.0)
            moved = True
        if not w["task"]["busy"] and moved:
            break
        time.sleep(0.3)
    print("result:", w["task"].get("result"))
    c = trials.connect()
    a = c.execute("SELECT * FROM attempts ORDER BY id DESC LIMIT 1").fetchone()
    print(f"attempt {a['id']}: {a['result']} / {a['failure']} task {a['task']}")
    for e in c.execute("SELECT kind, severity, data FROM events WHERE task = ? ORDER BY id", (a["task"],)):
        print(f"  event {e['kind']:20s} {e['severity']:5s} {e['data'][:120]}")
    time.sleep(2.0)
    for r in c.execute("SELECT kind, path, bytes, note FROM artifacts WHERE attempt = ?", (a["id"],)):
        print(f"  artifact {r['kind']} {r['path']} {r['bytes'] / 1e6 if r['bytes'] else 0:.1f} MB")
    sim.call("place_object", name=NAME, xy=o["xy"], yaw=float(o["yaw"]))      # put it back for the next run


if __name__ == "__main__":
    main()
