"""Pick & place benchmark: randomized can AND goal positions (optionally box obstacles), judged by sim ground truth.

Run in the pnp container while run_sim.py + bringup.launch.py are running:
    python3 /repo/tools/run_benchmark.py --tier smoke        #  5 trials, after any change      (~1-2 min)
    python3 /repo/tools/run_benchmark.py --tier regression   # 20 trials, before a commit
    python3 /repo/tools/run_benchmark.py --tier gate         # 50 trials, phase gates
    ... --obstacles   (Phase 5: 1-3 random boxes per trial; any box pushed > 5 mm counts as a collision)
    ... --profile dev (watchable speed; tests default to "fast")
    ... --grasp-source auto|pose|detector|ab  (sets /pnp_pick_place grasp_source; ab alternates auto/detector; each trial records which
                                            source DETECT actually used: "6-D pose" or "detector (why)")

Each trial:
  1. sim "randomize_scene": can -> random upright pose (spawn region), green goal marker -> random position
     (goal region, >= 15 cm from the can)
  2. /pick_place with place = the goal marker position
  3. ground truth (TF fr3_link0 -> target, from Isaac): where did the can REALLY end up? Is it upright?

A trial succeeds only if the action reports SUCCESS **and** ground truth agrees: can within 2 cm of the goal
center and standing upright. (The action's own placement check uses perception; ground truth keeps it honest.)

Gates: success rate >= 90 %, ground-truth placement error p95 < 20 mm (Phase 4), zero collisions (Phase 5:
a trial that moved a box is also not a success).
summary.json also has state_mean_s: mean seconds per state (DETECT, MOVE_PREGRASP, ...) = where the time goes.
Outputs: /data/runs/<run_id>/episodes.jsonl (one line per trial), summary.json.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rclpy
import yaml
from moveit_msgs.msg import PlanningSceneComponents
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
from moveit_msgs.srv import GetPlanningScene
from std_srvs.srv import Empty
from pnp_interfaces.action import PickPlace
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from sim_client import SimClient  # noqa: E402

from pnp_core.geometry import quat_to_matrix  # noqa: E402
from pnp_core.object_grasp import object_yaw  # noqa: E402

from pnp_core.catalog import load_robot as _load_robot  # noqa: E402
BASE = _load_robot()["frames"]["base"]     # the active cell's robot base frame

OUTCOMES = {getattr(PickPlace.Result, k): k for k in dir(PickPlace.Result)
            if k.isupper() and isinstance(getattr(PickPlace.Result, k), int)}
TIERS = {"smoke": 5, "regression": 20, "gate": 50}
YAW_TOL_DEG = 10.0           # non-symmetric objects: placed yaw within this of the goal (modulo symmetry)
COLLISION_MOVE_M = 0.005     # an obstacle box displaced more than this = the robot (or the can) hit it


class Bench(Node):
    def __init__(self, name="pnp_benchmark", verbose=False):
        super().__init__(name)
        self.verbose = verbose            # print each state change live (demo menu)
        self.goal_handle = None           # the running goal (to cancel it)
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.sim = SimClient(self)
        self.pp = ActionClient(self, PickPlace, "/pick_place")
        self.get_scene = self.create_client(GetPlanningScene, "/get_planning_scene")
        self.clear_octomap = self.create_client(Empty, "/clear_octomap")
        self.states = []
        self.all_feedback = []            # (state, detail) over all tasks of this Bench

    def refresh_octomap(self, settle=2.0) -> bool:
        """Layer B: forget the old Octomap (boxes moved) and let the camera fill a fresh one."""
        if not self.clear_octomap.wait_for_service(timeout_sec=5.0):
            return False
        ok = self.wait_future(self.clear_octomap.call_async(Empty.Request()), 5.0) is not None
        self.spin_for(settle)
        return ok

    def wait_obstacles_in_moveit(self, boxes, timeout=10.0) -> bool:
        """Block until MoveIt's planning scene holds exactly these boxes (the pick & place node mirrors
        /scene/known_obstacles asynchronously; starting a task before that would plan with the OLD layout)."""
        want = {f"obstacle_{b['id']}" for b in boxes}
        end = time.time() + timeout
        while time.time() < end:
            if self.get_scene.wait_for_service(timeout_sec=1.0):
                req = GetPlanningScene.Request()
                req.components.components = PlanningSceneComponents.WORLD_OBJECT_NAMES
                res = self.wait_future(self.get_scene.call_async(req), 2.0)
                if res is not None:
                    have = {o.id for o in res.scene.world.collision_objects if o.id.startswith("obstacle_")}
                    if have == want:
                        return True
            self.spin_for(0.1)
        return False

    def spin_for(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def wait_future(self, fut, timeout):
        end = time.time() + timeout
        while not fut.done() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
        return fut.result() if fut.done() else None

    def pick_place(self, goal_xy, label, profile, timeout=240.0, on_state=None):
        """Run one task. on_state: {feedback state: function} called once when the task first reaches that state
        (e.g. {"LIFT": lambda: sim.call("block_goal")}: change the scene in the middle of a task)."""
        if not self.pp.wait_for_server(timeout_sec=20.0):
            raise RuntimeError("/pick_place not available")
        self.states = []          # (state, wall time it started)
        self.feedback = []        # every (state, detail), e.g. which source the grasp came from
        goal = PickPlace.Goal(object_class=label, use_default_place=False, place_x=float(goal_xy[0]),
                              place_y=float(goal_xy[1]), place_yaw=0.0, speed_profile=profile)

        def on_fb(fb):
            self.feedback.append((fb.feedback.state, fb.feedback.detail))
            self.all_feedback.append((fb.feedback.state, fb.feedback.detail))
            if not self.states or self.states[-1][0] != fb.feedback.state:
                self.states.append((fb.feedback.state, time.time()))
            elif not fb.feedback.detail:
                return
            if self.verbose:
                print(f"   {fb.feedback.state:14s} {fb.feedback.detail}", flush=True)
        handle = self.goal_handle = self.wait_future(self.pp.send_goal_async(goal, feedback_callback=on_fb), 20.0)
        if handle is None or not handle.accepted:
            return None
        fut, end, fired = handle.get_result_async(), time.time() + timeout, set()
        while not fut.done() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            for st, fn in (on_state or {}).items():             # outside any callback: fn may spin (sim calls)
                if st not in fired and any(s == st for s, _ in self.states):
                    fired.add(st)
                    fn()
        res = fut.result() if fut.done() else None
        self.states.append(("END", time.time()))
        if res is None:
            handle.cancel_goal_async()
            return None
        return res.result

    def grasp_source(self):
        """Which pose the last task grasped from ("6-D pose" / "detector (...)"), from the DETECT feedback."""
        src = [d[len("grasp from "):].rsplit(": ", 1)[0] for s, d in self.feedback if s == "DETECT" and d.startswith("grasp from ")]
        return src[-1] if src else None

    def get_pick_param(self, name):
        """A parameter of the running pick & place node (None if it doesn't answer)."""
        cli = self.create_client(GetParameters, "/pnp_pick_place/get_parameters")
        if not cli.wait_for_service(timeout_sec=10.0):
            return None
        res = self.wait_future(cli.call_async(GetParameters.Request(names=[name])), 5.0)
        if not res or not res.values:
            return None
        v = res.values[0]
        return {ParameterType.PARAMETER_STRING: v.string_value, ParameterType.PARAMETER_BOOL: v.bool_value,
                ParameterType.PARAMETER_DOUBLE: v.double_value, ParameterType.PARAMETER_INTEGER: v.integer_value
                }.get(v.type)

    def scene(self):
        """(scene config, target object's catalog entry) of the RUNNING stack: tools follow whatever scene the
        pick & place node was launched with (bringup scene:=...), default basic."""
        path = self.get_pick_param("scene") or "config/scenes/basic.yaml"
        scene = yaml.safe_load((REPO / path).read_text())
        obj = yaml.safe_load((REPO / "config/objects.yaml").read_text())[scene["target"]["object"]]
        return scene, obj

    def gt_pose(self, frame):
        """Ground-truth (position, rotation matrix) of a TF frame in the base frame."""
        tf = self.tf.lookup_transform(BASE, frame, Time()).transform
        q, t = tf.rotation, tf.translation
        return np.array([t.x, t.y, t.z]), quat_to_matrix([q.x, q.y, q.z, q.w])

    def set_pick_param(self, name, value):
        """Set a parameter of the pick & place node (e.g. grasp_source) -> True if it was accepted."""
        cli = self.create_client(SetParameters, "/pnp_pick_place/set_parameters")
        if not cli.wait_for_service(timeout_sec=10.0):
            return False
        if isinstance(value, bool):
            p = Parameter(name=name, value=ParameterValue(type=ParameterType.PARAMETER_BOOL, bool_value=value))
        else:
            p = Parameter(name=name, value=ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=value))
        res = self.wait_future(cli.call_async(SetParameters.Request(parameters=[p])), 5.0)
        return bool(res and res.results and res.results[0].successful)

    def state_durations(self):
        """Seconds spent in each feedback state (a state can occur more than once, e.g. HOME)."""
        d = {}
        for (s, t0), (_, t1) in zip(self.states, self.states[1:]):
            d[s] = round(d.get(s, 0.0) + (t1 - t0), 2)
        return d

    def gt(self, frame):
        t = self.tf.lookup_transform(BASE, frame, Time()).transform.translation
        return np.array([t.x, t.y, t.z])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=list(TIERS), default=None,
                    help="smoke=5 (after any edit), regression=20 (before a commit), gate=50 (phase gate)")
    ap.add_argument("--trials", type=int, default=None, help="overrides --tier")
    ap.add_argument("--profile", default="fast", help="PickPlace speed_profile (config/robot_fr3.yaml)")
    ap.add_argument("--fixed", action="store_true", help="don't randomize: default can + goal positions")
    ap.add_argument("--obstacles", action="store_true",
                    help="randomize box obstacles each trial too (Phase 5); collisions = an obstacle moved > 5 mm")
    ap.add_argument("--perceived", action="store_true",
                    help="Phase 5B: the stack runs with obstacle_source:=perceived (boxes only via the depth camera)")
    ap.add_argument("--grasp-source", choices=["auto", "pose", "detector", "ground_truth", "ab"], default=None,
                    help="where the grasp pose comes from (needs pose_estimation:=true for auto/pose); default: as "
                         "running. ab = alternate auto / detector trial by trial (same mix of scenes for both)")
    ap.add_argument("--map-mode", choices=["continuous", "freeze_per_task"], default=None,
                    help="set /pnp_pick_place map_mode first (default: as running)")
    ap.add_argument("--change-policy", choices=["warn", "refresh", "stop"], default=None)
    ap.add_argument("--wrist-refine", choices=["on", "off", "ab"], default=None,
                    help="close look with the wrist camera before grasping (ab = alternate off / on per trial)")
    ap.add_argument("--settle", type=float, default=0.3)
    a = ap.parse_args()
    a.trials = a.trials or TIERS[a.tier or "gate"]

    contract = yaml.safe_load((REPO / "config/interfaces.yaml").read_text())
    git = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_benchmark" + (f"_{git}" if git else "")
    out = Path(os.environ.get("PNP_DATA", "/data")) / "runs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "episodes.jsonl", "w")

    rclpy.init()
    b = Bench()
    scene, obj = b.scene()
    height, label = obj["dims_m"][2], obj["yolo_class"]
    table_z = scene["table"]["center"][2] + scene["table"]["size"][2] / 2
    # objects whose yaw matters (not rotationally symmetric): the placed yaw is judged too, modulo the object's
    # symmetry period (a box: 180 deg). The benchmark asks for place_yaw 0.
    yaw_period = None if obj["grasp"].get("symmetric_yaw") else np.radians(obj["grasp"].get("yaw_period_deg", 360))
    print(f"[bench] scene {scene['name']}: target {scene['target']['object']} ('{label}')", flush=True)
    # parameters this run changes on the LIVE pick & place node: restored at the end, so a benchmark never leaves
    # the demo in another mode (found: a --map-mode run left the stack frozen)
    restore = {n: b.get_pick_param(n) for n in ("grasp_source", "map_mode", "change_policy", "wrist_refine")}
    if a.wrist_refine in ("on", "off"):
        b.set_pick_param("wrist_refine", a.wrist_refine == "on")
    if a.grasp_source and a.grasp_source != "ab" and not b.set_pick_param("grasp_source", a.grasp_source):
        print(f"could not set grasp_source={a.grasp_source} on /pnp_pick_place")
        return 1
    by_source = {}                         # ab mode: requested source -> list of (success, gt err, pick err)
    for name, val in (("map_mode", a.map_mode), ("change_policy", a.change_policy)):
        if val and not b.set_pick_param(name, val):
            print(f"could not set {name}={val} on /pnp_pick_place")
            return 1
    map_mode = b.get_pick_param("map_mode") or "continuous"
    print(f"[bench] map_mode {map_mode}, change_policy {b.get_pick_param('change_policy')}", flush=True)
    errors, durations, outcomes, ok_count, collisions = [], [], Counter(), 0, 0
    sources = Counter()                    # how many trials grasped from the 6-D pose vs the detector
    pick_errs = []                         # [mm] grasp pose vs the object's true position, per trial
    yaw_errs = []                          # [deg] placed yaw vs goal (non-symmetric objects)
    state_times = {}
    b.sim.call("randomize_obstacles" if a.obstacles else "clear_obstacles")
    for k in range(a.trials):
        b.sim.call("reset_target" if a.fixed else "randomize_scene")
        b.spin_for(a.settle)
        st = b.sim.call("state")
        goal_xy, start_xy, boxes = st["goal"]["xy"], st["target"]["xy"], st.get("obstacles", [])
        if a.perceived:
            if not b.refresh_octomap():
                print(f"[{k + 1:3d}/{a.trials}] WARN could not clear the Octomap", flush=True)
        elif not b.wait_obstacles_in_moveit(boxes):
            print(f"[{k + 1:3d}/{a.trials}] WARN MoveIt's obstacles don't match the sim layout", flush=True)
        requested = a.grasp_source
        if a.grasp_source == "ab":
            requested = "auto" if k % 2 == 0 else "detector"
            b.set_pick_param("grasp_source", requested)
        if a.wrist_refine == "ab":                              # compare per trial: close look off / on
            requested = "wrist_off" if k % 2 == 0 else "wrist_on"
            b.set_pick_param("wrist_refine", requested == "wrist_on")
        t0 = time.time()
        res = b.pick_place(goal_xy, label, a.profile)
        b.spin_for(a.settle)
        moved = {o["id"]: round(o["displacement_m"] * 1000, 1) for o in b.sim.call("state").get("obstacles", [])}
        hit = sorted(i for i, mm in moved.items() if mm > COLLISION_MOVE_M * 1000)
        collisions += bool(hit)
        for name, sec in b.state_durations().items():
            state_times.setdefault(name, []).append(sec)
        gt, gt_R = b.gt_pose(contract["frames"]["object_ground_truth"])
        gt_err = float(np.hypot(gt[0] - goal_xy[0], gt[1] - goal_xy[1]))
        upright = bool(abs((gt[2] - table_z) - height / 2) < 0.01)
        yaw_err = None
        if yaw_period is not None:
            e = object_yaw(gt_R, obj) % yaw_period              # goal yaw 0, modulo the symmetry period
            yaw_err = float(np.degrees(min(e, yaw_period - e)))
        name = OUTCOMES.get(res.outcome, "?") if res else "TIMEOUT"
        success = bool(res and res.outcome == PickPlace.Result.SUCCESS and gt_err < 0.02 and upright and not hit
                       and (yaw_err is None or yaw_err < YAW_TOL_DEG))
        # a box placed closer than the open jaw needs (obstacles.too_close_prob): refusing is the right answer
        too_close = any(o.get("near_target") for o in boxes)
        if too_close and res and res.outcome == PickPlace.Result.BLOCKED and not hit:
            success, name_override = True, "BLOCKED_CORRECTLY"
        else:
            name_override = None
        ok_count += success
        outcomes[name_override or (name if success or name != "SUCCESS" else ("SUCCESS_BUT_COLLIDED" if hit
                                                                                else "SUCCESS_BUT_GT_DISAGREES"))] += 1
        rec = {"trial": k, "start_xy": [round(v, 4) for v in start_xy], "goal_xy": [round(v, 4) for v in goal_xy],
               "outcome": name, "message": res.message if res else "no result", "success": success,
               "gt_final": gt.round(4).tolist(), "gt_place_err_mm": round(gt_err * 1000, 1), "upright": upright,
               "gt_yaw_err_deg": None if yaw_err is None else round(yaw_err, 2),
               "perceived_place_err_mm": (round(res.placement_error_m * 1000, 1)
                                          if res and res.placement_error_m == res.placement_error_m else None),
               "grasp_attempts": res.grasp_attempts if res else None, "grasp_source": b.grasp_source(),
               "grasp_source_requested": requested,
               "look": next((d for s_, d in reversed(b.feedback) if s_ == "LOOK" and d.startswith(("refined", "close"))),
                            None),
               # perception at the pick: the pose the grasp was planned from vs the can's TRUE position (sim)
               "pick_err_mm": (round(float(np.hypot(res.picked_at[0] - start_xy[0], res.picked_at[1] - start_xy[1]))
                                     * 1000, 1) if res and len(res.picked_at) >= 2 else None),
               "duration_s": round(time.time() - t0, 1),
               "last_state": b.states[-2][0] if len(b.states) > 1 else None,
               "state_s": b.state_durations(), "profile": a.profile,
               "obstacles": [{"id": o["id"], "center": [round(v, 3) for v in o["center"]], "yaw": round(o["yaw"], 3)}
                             for o in boxes],
               "obstacle_moved_mm": moved, "collided_with": hit}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if success:
            errors.append(gt_err)
        durations.append(rec["duration_s"])
        sources[(rec["grasp_source"] or "none").split(" (")[0]] += 1
        if rec["pick_err_mm"] is not None:
            pick_errs.append(rec["pick_err_mm"])
        if yaw_err is not None:
            yaw_errs.append(yaw_err)
        by_source.setdefault(requested, []).append((success, rec["gt_place_err_mm"], rec["pick_err_mm"]))
        print(f"[{k + 1:3d}/{a.trials}] {'OK  ' if success else 'FAIL'} {name:18s} gt err {rec['gt_place_err_mm']:6.1f} mm"
              f"  pick err {rec['pick_err_mm']} mm ({(rec['grasp_source'] or '-').split(' (')[0]})"
              + (f"  yaw err {yaw_err:.1f} deg" if yaw_err is not None else "")
              + f"  upright={upright}  attempts={rec['grasp_attempts']}  {rec['duration_s']:5.1f} s"
              + (f"  boxes={[o['id'] for o in boxes]}" if boxes else "") + (f"  HIT {hit}" if hit else "")
              + ("" if success else f"  last={rec['last_state']}: {rec['message']}"), flush=True)
    log.close()

    e = np.array(errors) * 1000
    rate = ok_count / max(a.trials, 1)
    summary = {
        "run_id": run_id, "scene": scene["name"], "target": scene["target"]["object"], "trials": a.trials,
        "randomized": not a.fixed, "profile": a.profile,
        "obstacles": a.obstacles, "obstacle_source": "perceived" if a.perceived else "known", "map_mode": map_mode,
        # frozen tasks only: how often the change monitor raised a (false) alarm while nothing changed
        "scene_change_alarms": sum("scene changed" in d for s, d in b.all_feedback),
        "success_rate": round(rate, 3), "collisions": collisions,
        "grasp_source_requested": a.grasp_source, "grasp_sources_used": dict(sources),
        "gt_yaw_err_deg": ({"mean": round(float(np.mean(yaw_errs)), 2), "max": round(float(np.max(yaw_errs)), 2)}
                           if yaw_errs else None),
        "pick_err_mm": ({"mean": round(float(np.mean(pick_errs)), 1), "p95": round(float(np.percentile(pick_errs, 95)), 1),
                         "max": round(float(np.max(pick_errs)), 1)} if pick_errs else None),
        "outcomes": dict(outcomes),
        "gt_place_err_mm_successes": ({"mean": round(float(e.mean()), 1), "p95": round(float(np.percentile(e, 95)), 1),
                                       "max": round(float(e.max()), 1)} if len(e) else None),
        "duration_s": {"median": round(float(np.median(durations)), 1), "max": round(float(np.max(durations)), 1)},
        # where the time goes (mean seconds per trial in each state): what to optimize next
        "state_mean_s": {k: round(float(np.mean(v)), 2) for k, v in sorted(state_times.items(), key=lambda kv: -np.mean(kv[1]))},
    }
    if a.grasp_source == "ab" or a.wrist_refine == "ab":
        def stats(v):
            v = np.array([x for x in v if x is not None], float)
            return {"mean": round(float(v.mean()), 1), "p95": round(float(np.percentile(v, 95)), 1),
                    "max": round(float(v.max()), 1)} if len(v) else None
        summary["by_grasp_source"] = {src: {"trials": len(v), "success": sum(x[0] for x in v),
                                            "gt_place_err_mm": stats([x[1] for x in v if x[0]]),
                                            "pick_err_mm": stats([x[2] for x in v])} for src, v in by_source.items()}
        b.set_pick_param("grasp_source", "auto")
    summary["gates"] = {"success_rate>=0.9": rate >= 0.9, "place_err_p95<20mm": bool(len(e) and np.percentile(e, 95) < 20),
                        "zero_collisions": collisions == 0}
    summary["passed"] = all(summary["gates"].values())
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"RESULT: {'PASS' if summary['passed'] else 'FAIL'}   (details: {out})")
    for n, v in restore.items():
        if v is not None:
            b.set_pick_param(n, v)
    b.destroy_node()
    rclpy.try_shutdown()
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
