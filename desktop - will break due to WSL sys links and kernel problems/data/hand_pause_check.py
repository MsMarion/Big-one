"""#7 validation (sim): the hand-detection pause.
  1. the stand-in hand (sim command hand_prop, magenta) in the workspace -> /safety/hand present; out of it /
     hidden -> not present
  2. a running pick: the prop appears while the arm moves -> the arm stops (joints still), feedback PAUSED; the prop
     goes -> resumed after ~1 s, the pick finishes; events hand_paused / hand_resumed in the trial log

    docker exec pnp_dev bash /repo/data/rospy.sh /repo/data/hand_pause_check.py [object=lego_2x2]
"""
import json
import sys
import threading
import time
import urllib.request

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState
from std_msgs.msg import String

sys.path.insert(0, "/repo")
sys.path.insert(0, "/repo/tools")
from pnp_core import trials  # noqa: E402
from sim_client import SimClient  # noqa: E402

API = "http://localhost:8088"
NAME = sys.argv[1] if len(sys.argv) > 1 else "lego_2x2"
HAND_XY, HAND_Z = [0.14, 0.22], 0.10          # beside the work area, well inside r 0.45, in the camera's view
state = {"hand": None, "q": None, "q_t": 0.0}


def get(path):
    return json.loads(urllib.request.urlopen(API + path, timeout=10).read())


def post(path, body):
    req = urllib.request.Request(API + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=10).read())


def wait_hand(want, timeout=12.0):
    end = time.time() + timeout
    while time.time() < end:
        h = state["hand"]
        if h is not None and time.time() - h["t"] < 2.0 and h["present"] == want:
            return h
        time.sleep(0.1)
    return None


def joint_speed(dt=0.5):
    q0 = np.array(state["q"])
    time.sleep(dt)
    return float(np.max(np.abs(np.array(state["q"]) - q0)) / dt)


def main():
    rclpy.init()
    node = rclpy.create_node("hand_pause_check")
    sim = SimClient(node)
    # the subscriptions on their OWN node + executor (SimClient spins `node` itself: one node in two spinners
    # starved the callbacks)
    sub = rclpy.create_node("hand_pause_check_sub")
    sub.create_subscription(String, "/safety/hand", lambda m: state.__setitem__("hand", json.loads(m.data)), 10)
    sub.create_subscription(JointState, "/joint_states", lambda m: state.__setitem__("q", list(m.position[:5])), 10)
    ex = MultiThreadedExecutor(num_threads=2)
    ex.add_node(sub)
    threading.Thread(target=ex.spin, daemon=True).start()
    ok = True

    print("1. detection")
    sim.call("hand_prop", present=False)
    h = wait_hand(False)
    print(f"   no prop: {'OK' if h else 'FAIL'} {h}")
    ok &= h is not None
    print(sim.call("hand_prop", present=True, xy=HAND_XY, z=HAND_Z))
    h = wait_hand(True)
    print(f"   prop at {HAND_XY}, {HAND_Z} m up: {'OK' if h else 'FAIL'} {h}")
    ok &= h is not None
    sim.call("hand_prop", present=False)
    wait_hand(False)

    print("2. pause a running pick")
    post("/api/scan", {})
    time.sleep(1.0)
    while get("/api/world")["scan"]["busy"]:
        time.sleep(0.5)
    objs = get("/api/world")["objects"]
    o = next((x for x in objs if x["name"] == NAME), None) or next(x for x in objs if x["name"].startswith("lego"))
    for goal in ([0.215, 0.025], [0.20, -0.12], [0.17, 0.12], [0.24, -0.06], [0.15, -0.16]):
        r = post('/api/pick', {'id': o['id'], 'xy': goal, 'yaw_deg': 0})
        print(f"   picking {o['id']} {o['name']} at {o['xy']} -> {goal}: {r}")
        if r.get("ok", True):
            break
    t0, phase, states = time.time(), "wait_move", []
    while time.time() - t0 < 240:
        w = get("/api/world")
        st = w["task"].get("state")
        if not states or states[-1] != st:
            states.append(st)
        if phase == "wait_move" and st in ("MOVE_PREGRASP", "APPROACH", "MOVE_TO_GOAL", "TRANSPORT") \
                and joint_speed(0.2) > 0.05:
            print(f"   state {st}, arm moving: the hand appears")
            sim.call("hand_prop", present=True, xy=HAND_XY, z=HAND_Z)
            t_hand = time.time()
            time.sleep(2.5)
            sp = joint_speed(0.5)
            st = get("/api/world")["task"].get("state")
            print(f"   {time.time() - t_hand:.1f} s later: joint speed {sp:.3f} rad/s, state {st} "
                  f"({'OK: stopped' if sp < 0.02 else 'FAIL: still moving'})")
            ok &= sp < 0.02
            sim.call("hand_prop", present=False)
            t_gone, phase = time.time(), "wait_resume"
        elif phase == "wait_resume" and joint_speed(0.2) > 0.05:
            print(f"   moving again {time.time() - t_gone:.1f} s after the hand left")
            phase = "finish"
        if not w["task"]["busy"] and phase != "wait_move":
            break
        time.sleep(0.2)
    res = get("/api/world")["task"].get("result")
    print(f"   states: {states}")
    print(f"   result: {res}")
    ok &= phase == "finish" and bool(res) and "SUCCESS" in json.dumps(res)
    c = trials.connect()
    a = c.execute("SELECT * FROM attempts ORDER BY id DESC LIMIT 1").fetchone()
    ev = [e["kind"] for e in c.execute("SELECT kind FROM events WHERE task = ? ORDER BY id", (a["task"],))]
    print(f"   events of task {a['task']}: {ev}")
    ok &= "hand_paused" in ev and "hand_resumed" in ev
    print("PASS" if ok else "FAIL")
    rclpy.try_shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
