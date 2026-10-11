"""Interactive playground: drive the sim + pick & place stack from a small keyboard menu.

Started by demo.ps1 (in its own window), or by hand in the pnp container while the sim and bringup run:
    python3 /repo/tools/demo_menu.py

Everything here goes through the same interfaces the benchmark uses:
  sim commands  /sim/request -> /sim/response   (tools/sim_client.py; sim/sim_services.py lists them)
  the task      /pick_place action              (pnp_interfaces/PickPlace)
  judging       ground-truth TF from Isaac + obstacle displacement from the sim
Ctrl+C during a task cancels it (the robot puts the can back down if it was holding it).
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rcl_interfaces.srv import GetParameters
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_benchmark import COLLISION_MOVE_M, OUTCOMES, REPO, Bench  # noqa: E402

MENU = """
 ---------------------------------------------------------------- speed: {speed:4s}  obstacles: {obs} ({src})
 map: {map_mode}   change policy: {policy}   wrist close look: {wrist}
 calibration: {cal}   policy on drift: {cal_policy}
  p  pick & place once  (can -> green marker)       l  loop: randomize + pick N times
  r  randomize can + goal (+ boxes if on)           d  reset can + goal to default positions
  o  obstacles ON: new random box layout            c  obstacles OFF (clear the table)
  b  block the goal with a tower (impossible task -> expect UNREACHABLE)
  s  toggle speed dev / fast                        t  show state (can, goal, boxes)
  m  measure the boxes with the camera vs their true size (demo.ps1 -Sensors)
  f  map mode: continuous <-> freeze per task      g  change policy: warn -> refresh -> stop
  x  pick & place, and put a tower on the goal while lifting (does the change monitor react?)
  w  close look with the wrist camera before grasping: on / off
  n  sensor realism: ideal -> realistic -> stress  (depth/color noise + camera placement errors)
  k  calibration check (both cameras vs robot vs each other, ~30 s)
  a  calibrate a camera (hand-eye, ~2 min)          y  on calibration drift: warn <-> stop
  q  quit
"""


class Demo:
    def __init__(self):
        self.b = Bench(name="pnp_demo_menu", verbose=True)
        scene, obj = self.b.scene()                  # whatever scene the running stack uses (demo.ps1 -Scene)
        self.label, self.height = obj["yolo_class"], obj["dims_m"][2]
        self.table_z = scene["table"]["center"][2] + scene["table"]["size"][2] / 2
        print(f"scene '{scene['name']}': picking the {scene['target']['object']} ('{self.label}')")
        self.speed, self.obstacles = "dev", False
        self.calibration = {}                        # latest /calibration/status (tools: calibrate / calibration_check)
        self.b.create_subscription(String, "/calibration/status",
                                   lambda m: setattr(self, "calibration", json.loads(m.data)),
                                   QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                              reliability=ReliabilityPolicy.RELIABLE))
        self.known = self.uses_known_obstacles()

    def uses_known_obstacles(self) -> bool:
        """Ask the pick & place node how it gets obstacles: known (sim box list -> MoveIt boxes) or perceived
        (depth camera -> Octomap; demo.ps1 -Perceived / -Sensors). Defaults to known if it doesn't answer."""
        cli = self.b.create_client(GetParameters, "/pnp_pick_place/get_parameters")
        if cli.wait_for_service(timeout_sec=5.0):
            res = self.b.wait_future(cli.call_async(GetParameters.Request(names=["use_known_obstacles"])), 5.0)
            if res is not None and res.values:
                return res.values[0].bool_value
        return True

    def state(self):
        st = self.b.sim.call("state")
        print(f"  can  at ({st['target']['xy'][0]:.3f}, {st['target']['xy'][1]:.3f})"
              f"   goal at ({st['goal']['xy'][0]:.3f}, {st['goal']['xy'][1]:.3f})")
        for o in st.get("obstacles", []):
            print(f"  box {o['id']:6s} at ({o['center'][0]:.3f}, {o['center'][1]:.3f})  size {o['size']}  "
                  f"yaw {np.degrees(o['yaw']):5.0f} deg  moved {o['displacement_m'] * 1000:5.1f} mm")
        return st

    def sim(self, cmd):
        self.b.sim.call(cmd)
        self.b.spin_for(0.3)
        self.state()

    def pick(self, on_state=None):
        st = self.b.sim.call("state")
        boxes = st.get("obstacles", [])
        if not self.known:                  # perceived: no box list in MoveIt; rebuild the voxel map from the camera
            if not self.b.refresh_octomap():
                print("  (warning: couldn't clear the Octomap; old voxels may remain)")
        elif not self.b.wait_obstacles_in_moveit(boxes):
            print("  (warning: MoveIt doesn't have the current box layout yet)")
        print(f"  pick & place -> goal ({st['goal']['xy'][0]:.3f}, {st['goal']['xy'][1]:.3f}), speed {self.speed}")
        t0 = time.time()
        try:
            res = self.b.pick_place(st["goal"]["xy"], self.label, self.speed, on_state=on_state)
        except KeyboardInterrupt:
            print("  canceling ... (the robot finishes recovering first)")
            if self.b.goal_handle is not None:
                self.b.wait_future(self.b.goal_handle.cancel_goal_async(), 10.0)
                res = self.b.wait_future(self.b.goal_handle.get_result_async(), 120.0)
                res = res.result if res else None
            else:
                res = None
        self.b.spin_for(0.3)
        gt = self.b.gt("target")
        err = float(np.hypot(gt[0] - st["goal"]["xy"][0], gt[1] - st["goal"]["xy"][1]))
        upright = abs((gt[2] - self.table_z) - self.height / 2) < 0.01
        moved = {o["id"]: o["displacement_m"] for o in self.b.sim.call("state").get("obstacles", [])}
        hit = [i for i, m in moved.items() if m > COLLISION_MOVE_M]
        name = OUTCOMES.get(res.outcome, "?") if res else "NO RESULT"
        print(f"  => {name}: {res.message if res else ''}")
        print(f"     ground truth: can {err * 1000:.1f} mm from the goal center, upright={upright}"
              + (f", COLLISION: moved {hit}" if hit else (", no box touched" if moved else ""))
              + f", {time.time() - t0:.1f} s")
        times = self.b.state_durations()
        if times:
            print("     time per state: " + ", ".join(f"{k} {v:.1f}" for k, v in times.items()))
        return name == "SUCCESS" and err < 0.02 and upright and not hit

    def toggle(self, name, values):
        """Switch a pick & place parameter to the next value (live: no restart)."""
        cur = self.b.get_pick_param(name)
        nxt = values[(values.index(cur) + 1) % len(values)] if cur in values else values[0]
        print(f"  {name}: {cur} -> {nxt}" if self.b.set_pick_param(name, nxt) else f"  could not set {name}")

    def next_reality(self):
        order = ["ideal", "realistic", "stress"]
        self.reality = order[(order.index(getattr(self, "reality", "ideal")) + 1) % len(order)]
        subprocess.run([sys.executable, str(REPO / "tools/set_reality.py"), self.reality])

    def calibrate(self):
        which = input("  which camera? fixed / wrist / both [both] ").strip() or "both"
        for cam in (["fixed", "wrist"] if which == "both" else [which]):
            subprocess.run([sys.executable, str(REPO / "tools/calibrate.py"), cam])

    def cal_status(self) -> str:
        self.b.spin_for(0.2)
        c = self.calibration
        if not c:
            return "no monitor status yet"
        return c["state"] + (f" (tag offset {c['px']} px ~ {c['mm']} mm)" if "px" in c else "")

    def toggle_bool(self, name):
        new = not self.b.get_pick_param(name)
        print(f"  {name} -> {new}" if self.b.set_pick_param(name, new) else f"  could not set {name}")

    def loop(self):
        try:
            n = int(input("  how many? [5] ") or 5)
        except ValueError:
            return
        ok = 0
        for i in range(n):
            print(f"\n  --- {i + 1}/{n} ---")
            self.b.sim.call("randomize_scene")
            self.b.spin_for(0.3)
            ok += self.pick()
        print(f"\n  {ok}/{n} succeeded")

    def run(self):
        print(__doc__.split("\n\n")[0])
        self.state()
        actions = {"p": self.pick, "l": self.loop, "t": self.state,
                   "r": lambda: self.sim("randomize_scene"), "d": lambda: self.sim("reset_target"),
                   "b": lambda: self.sim("block_goal"),
                   "x": lambda: self.pick(on_state={"LIFT": lambda: (print("  >>> putting a tower on the goal"),
                                                                     self.b.sim.call("block_goal"))}),
                   "f": lambda: self.toggle("map_mode", ["continuous", "freeze_per_task"]),
                   "g": lambda: self.toggle("change_policy", ["warn", "refresh", "stop"]),
                   "w": lambda: self.toggle_bool("wrist_refine"),
                   "n": self.next_reality,
                   "k": lambda: subprocess.run([sys.executable, str(REPO / "tools/calibration_check.py")]),
                   "a": self.calibrate,
                   "y": lambda: self.toggle("calibration_policy", ["warn", "stop"]),
                   "m": lambda: subprocess.run([sys.executable, str(REPO / "tools/scene_check.py")])}
        while True:
            print(MENU.format(speed=self.speed, obs="ON" if self.obstacles else "off",
                              map_mode=self.b.get_pick_param("map_mode"), policy=self.b.get_pick_param("change_policy"),
                              wrist="on" if self.b.get_pick_param("wrist_refine") else "off",
                              cal=self.cal_status(), cal_policy=self.b.get_pick_param("calibration_policy"),
                              src="known boxes" if self.known else "perceived: Octomap"))
            try:
                key = input("> ").strip()[:1]
            except (EOFError, KeyboardInterrupt):
                break
            if key == "q":
                break
            if key == "s":
                self.speed = "fast" if self.speed == "dev" else "dev"
            elif key in ("o", "c"):
                self.obstacles = key == "o"
                self.sim("randomize_obstacles" if self.obstacles else "clear_obstacles")
            elif key in actions:
                try:
                    actions[key]()
                except Exception as e:  # noqa: BLE001  (keep the menu alive)
                    print(f"  error: {e}")


def main():
    rclpy.init()
    d = Demo()
    try:
        d.run()
    finally:
        d.b.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
