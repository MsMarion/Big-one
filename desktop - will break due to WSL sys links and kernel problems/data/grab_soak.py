"""The grab soak: many grabs, fast, logged. Each attempt teleports one brick (round robin) to a seeded random spot /
yaw in the pick area -- including its edges and the far corners -- parks the others out of reach, and runs
PickPlace grab_only (detect -> close look -> grab -> lift -> grip check -> put it down, home) at the `fast` speed.
Every attempt lands in the trial log (pnp_core/trials.py, /data/trials.sqlite): the pick node writes what it did,
this script adds the truth (where the brick really was). Compare runs with tools/trials_report.py.

    docker exec pnp_dev bash /repo/data/rospy.sh /repo/data/grab_soak.py [--tier quick|gate] [--n N] [--seed S]
        [--set grasp_scoring=true ...]  (pick node parameters for this run, recorded in its flags) [--note TEXT]

Tiers: quick = 10 grabs (~3 min), gate = 30 (~8 min). The stack should run with servo_error typical and the sim's
reality realistic (setup/stack.ps1 Ensure-Stack -ServoError typical -Reality realistic): the flags record both.
"""
import json
import math
import random
import subprocess
import sys
import time

import rclpy
from rclpy.action import ActionClient

sys.path.insert(0, "/repo")
sys.path.insert(0, "/repo/tools")
from pnp_core import trials  # noqa: E402
from pnp_core.catalog import load_scene, load_yaml  # noqa: E402
from pnp_interfaces.action import PickPlace  # noqa: E402
from sim_client import SimClient  # noqa: E402

NODE = "/pnp_pick_place"
FLAG_PARAMS = ("grasp_scoring", "closed_loop_approach", "grasp_source", "wrist_refine")


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def ros_param(name):
    r = subprocess.run(["ros2", "param", "get", NODE, name], capture_output=True, text=True, timeout=20)
    out = r.stdout.strip()
    return out.split(":", 1)[1].strip() if r.returncode == 0 and ":" in out else None


def set_param(kv):
    k, v = kv.split("=", 1)
    r = subprocess.run(["ros2", "param", "set", NODE, k, v], capture_output=True, text=True, timeout=20)
    print(f"  set {k}={v}: {r.stdout.strip() or r.stderr.strip()}")


def bringup_arg(name):
    r = subprocess.run(["pgrep", "-fa", "ros2 launch pnp_bringup [b]ringup.launch.py"], capture_output=True, text=True)
    for tok in r.stdout.split():
        if tok.startswith(name + ":="):
            return tok.split(":=", 1)[1]
    return None


def spots(scene, n, seed):
    """n (x, y, yaw, zone): 60 % anywhere in the pick area, 25 % on its edges, 15 % in its far corners."""
    rnd = random.Random(seed)
    rg = scene["target"]["spawn_region"]
    (x0, x1), (y0, y1) = rg["x"], rg["y"]
    r_max = float(scene.get("ui", {}).get("reach_m", [0.0, 0.25])[1]) - 0.005
    out = []
    while len(out) < n:
        u = rnd.random()
        if u < 0.60:
            x, y, zone = rnd.uniform(x0, x1), rnd.uniform(y0, y1), "inside"
        elif u < 0.85:
            side = rnd.randrange(4)
            t = rnd.random()
            x, y = [(x0, y0 + t * (y1 - y0)), (x1, y0 + t * (y1 - y0)), (x0 + t * (x1 - x0), y0),
                    (x0 + t * (x1 - x0), y1)][side]
            zone = "edge"
        else:
            x, y, zone = x1, rnd.choice((y0, y1)), "far_corner"
        if math.hypot(x, y) > r_max:          # the far corners sit at the reach limit: pull in along the ray
            s = r_max / math.hypot(x, y)
            x, y = x * s, y * s
        out.append((round(x, 4), round(y, 4), round(rnd.uniform(-math.pi / 2, math.pi / 2), 3), zone))
    return out


def main():
    tier = arg("--tier", "quick")
    n = int(arg("--n", {"quick": 10, "gate": 30}.get(tier, 10)))
    seed = int(arg("--seed", 7))
    rclpy.init()
    node = rclpy.create_node("grab_soak")
    sim = SimClient(node)
    pp = ActionClient(node, PickPlace, "/pick_place")
    if not pp.wait_for_server(timeout_sec=20):
        sys.exit("no /pick_place")
    for i, a in enumerate(sys.argv):
        if a == "--set":
            set_param(sys.argv[i + 1])
    scene = load_scene()
    cat = load_yaml("config/objects.yaml")
    objs = [("target", scene["target"]["object"])] + [(e["name"], e["object"]) for e in scene.get("extra_objects", [])]
    flags = {"speed_profile": "fast", "servo_error": bringup_arg("servo_error") or "none",
             "reality": arg("--reality", "realistic")}
    flags.update({p: ros_param(p) for p in FLAG_PARAMS})
    c = trials.connect()
    run = trials.start_run(c, scene.get("name", "?"), seed, tier, flags, arg("--note", ""))
    print(f"run {run}: {tier}, {n} grabs, seed {seed}, flags {json.dumps(flags)}")
    sim.call("clear_obstacles")
    t = scene["table"]
    park = [(round(t["center"][0] + t["size"][0] / 2 - 0.05, 3), y) for y in (-0.16, 0.0, 0.16, 0.08, -0.08)]
    tally, t_run = {}, time.time()
    for k, (x, y, yaw, zone) in enumerate(spots(scene, n, seed)):
        name, objname = objs[k % len(objs)]
        for j, (o, _) in enumerate(x for x in objs if x[0] != name):
            sim.call("place_object", name=o, xy=list(park[j]), yaw=0.0)
        sim.call("place_object", name=name, xy=[x, y], yaw=yaw)
        time.sleep(1.5)
        t0 = time.time()
        g = PickPlace.Goal(object_class=cat[objname]["yolo_class"], use_default_place=True, speed_profile="fast",
                           use_pick_hint=True, pick_x=x, pick_y=y, grab_only=True)
        fut = pp.send_goal_async(g)
        rclpy.spin_until_future_complete(node, fut, timeout_sec=20)
        res = None
        if fut.result() is not None and fut.result().accepted:
            rf = fut.result().get_result_async()
            rclpy.spin_until_future_complete(node, rf, timeout_sec=240)
            res = rf.result().result if rf.result() else None
        outcome = res.outcome_name if res else "TIMEOUT"
        msg = res.message if res else ""
        last = trials.last_attempt(c, run)
        if last is not None and last["t"] >= t0 - 1:
            trials.update_attempt(c, last["id"], true_x=x, true_y=y, true_yaw=yaw)
        cls = trials.classify_failure(outcome, msg)
        tally[cls] = tally.get(cls, 0) + 1
        print(f"  {k + 1:2d}/{n} {objname:9s} ({x:.3f}, {y:.3f}) yaw {math.degrees(yaw):5.0f} {zone:10s} -> "
              f"{outcome:13s} {cls:16s} {time.time() - t0:5.1f} s  {msg[:90]}", flush=True)
    trials.close_run(c, run)
    ok = tally.get("ok", 0)
    print(f"run {run}: {ok}/{n} grabbed ({100 * ok / n:.0f} %) in {(time.time() - t_run) / 60:.1f} min; {tally}")


if __name__ == "__main__":
    main()
