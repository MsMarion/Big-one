"""Sim-only "reset buttons" for tools and benchmarks, as a request/response over TOPICS.

    publish  /sim/request   std_msgs/String  '{"id": 7, "cmd": "randomize_scene"}'
    receive  /sim/response  std_msgs/String  '{"id": 7, "ok": true, "target": {...}, "goal": {...}}'

    cmd: randomize_target   can -> random upright pose in the scene's spawn region
         randomize_goal     green goal marker -> random position in the goal region, away from the can
         randomize_scene    both (can first, then a goal at least min_separation_m away)
         target_away        can -> on the floor behind the robot, out of the camera view (negative tests)
         reset_target       can (and goal marker) -> the scene's default positions
         randomize_obstacles  Phase 5: lay out 1-3 random SOLVABLE boxes (pnp_core/obstacles.py) and keep doing
                            so after every can/goal change, until clear_obstacles
         clear_obstacles    park all boxes off the table, obstacles off
         block_goal         a tall box right on the goal (impossible task -> the action must say UNREACHABLE)
         state              current can + goal positions, and each box's displacement since it was placed
                            (> 5 mm = something hit it)
         (scenes with extra_objects: randomize_scene / reset_target place those too, apart from each other;
          state lists every object's xy under "objects")
         visual_gap         {links: [robot link names]}: how far each link's VISIBLE surface is from the target's
                            visible shape (signed: - = overlapping), e.g. the gripper's fingers on a held object

Also publishes the current box layout as the KNOWN OBSTACLES (std_msgs/String JSON, /scene/known_obstacles) on
every change and once a second; the pick & place node mirrors it into MoveIt's planning scene. (On a real robot
this list would come from a measured map; Layer B replaces it with live depth data.)

Why topics and not ROS services: services do NOT interoperate between DDS vendors (Isaac Sim's Fast DDS vs the
zenoh bridge's CycloneDDS encode request/reply ids differently), so a service hosted in Isaac can't be called
from the container through the bridge (tested: the call hangs). Topics work across vendors.
Client side: tools/sim_client.py (retries with the same id; repeated ids are answered from a cache).

Requests are processed in the sim loop (spin_once between steps), so the scene only changes between physics steps.
"""

import json
import time

import numpy as np
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from scene import place_extra, place_target, set_obstacles

from pnp_core.catalog import load_fixed_camera, load_robot, load_yaml
from pnp_core.geometry import make_transform, quat_to_matrix
from scene import camera_nominal_poses, prim_world_T, rpy_matrix, set_camera_pose, true_camera_pose
from pnp_core.obstacles import _make, sample_layout


class SimServices:
    def __init__(self, h, seed: int = 0):
        self.h = h
        self.rng = np.random.default_rng(seed)
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("pnp_sim")
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.pub = self.node.create_publisher(String, "/sim/response", qos)
        self.node.create_subscription(String, "/sim/request", self._on_request, qos)
        self.handlers = {"randomize_target": self._randomize_target, "randomize_goal": self._randomize_goal,
                         "randomize_scene": self._randomize_scene, "target_away": self._away,
                         "reset_target": self._reset, "state": self._state,
                         "randomize_obstacles": self._randomize_obstacles, "clear_obstacles": self._clear_obstacles,
                         "block_goal": self._block_goal, "set_reality": self._set_reality,
                         "camera_truth": self._camera_truth, "calibration_targets": self._calibration_targets,
                         "link_pose": self._link_pose, "place_object": self._place_object,
                         "set_obstacles": self._set_obstacles, "camera_error": self._camera_error,
                         "hand_prop": self._hand_prop,
                         "visual_gap": self._visual_gap}
        self.done = {}   # id -> response: clients RETRY (a first message can be lost while the bridge builds
                         # its route), so a repeated id is answered from here instead of executing twice
        self.marker = h.world.scene.get_object("goal_marker")
        self.goal_xy = list(h.cfg["goal"]["xy"])
        self.obstacles_on = False
        self.layout = []                  # boxes currently on the table (pnp_core.obstacles dicts)
        self.layout_seq = 0
        self.known_pub = self.node.create_publisher(String, "/scene/known_obstacles", qos)
        self.last_known = 0.0
        # every shape that exists once the scene is built: anything that appears LATER was added by hand (Isaac's
        # GUI: Create > Shape > Cube...) and is reported as truth too (/scene/known_obstacles "hand_boxes")
        self.base_gprims = self._gprim_paths()
        self.camera_pos = load_fixed_camera()["mount"]["position"]
        self.can_height = load_yaml("config/objects.yaml")[h.cfg["target"]["object"]]["dims_m"][2]
        # the target's half size across (its shorter side): a "too close" box's gap is measured from there
        self.target_half = min(load_yaml("config/objects.yaml")[h.cfg["target"]["object"]]["dims_m"][:2]) / 2
        # camera placement errors (config/sim_reality.yaml): the believed poses are what ROS uses (TF); Isaac moves the
        # simulated cameras to believed x error (x drift): a calibration has something real to find
        self.cam_nominal = camera_nominal_poses(h)
        self.reality, self.errors, self.drift, self.drift_t0, self.drift_applied = "ideal", {}, None, 0.0, 0.0

    def spin_once(self):
        rclpy.spin_once(self.node, timeout_sec=0.0)
        if self.drift and time.time() - self.drift_applied > 1.0:     # the fixed camera's stand slowly moves
            self._apply_camera_errors()
        if time.time() - self.last_known > 1.0:
            self._publish_known()

    def _publish_known(self):
        self.known_pub.publish(String(data=json.dumps(
            {"seq": self.layout_seq, "frame": load_robot()["frames"]["base"], "boxes": self.layout,
             "hand_boxes": self._hand_boxes()})))
        self.last_known = time.time()

    @staticmethod
    def _gprim_paths():
        from isaacsim.core.utils.stage import get_current_stage
        from pxr import UsdGeom
        return {str(p.GetPath()) for p in get_current_stage().Traverse() if p.IsA(UsdGeom.Gprim)}

    def _hand_boxes(self):
        """The oriented box of every shape added after the scene was built (by hand): the base frame = the world here.
        Not part of `boxes` (the pool layout the benchmark scores collisions on); the scene model uses both."""
        from isaacsim.core.utils.stage import get_current_stage
        from pxr import Usd, UsdGeom
        out = []
        try:
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
            for prim in get_current_stage().Traverse():
                path = str(prim.GetPath())
                if not prim.IsA(UsdGeom.Gprim) or path in self.base_gprims or not path.startswith("/World"):
                    continue
                bb = cache.ComputeWorldBound(prim)
                rng = bb.GetRange()
                if rng.IsEmpty():
                    continue
                M = np.array(bb.GetMatrix(), float).T                 # Gf: row vectors -> column 4x4
                lo, hi = np.array(rng.GetMin(), float), np.array(rng.GetMax(), float)
                scale = np.linalg.norm(M[:3, :3], axis=0)
                Rm = M[:3, :3] / np.maximum(scale, 1e-12)
                center = (M @ np.r_[(lo + hi) / 2, 1.0])[:3]
                out.append({"id": f"hand:{prim.GetName()}", "path": path,
                            "size": [round(float(v), 4) for v in (hi - lo) * scale],
                            "center": [round(float(v), 4) for v in center],
                            "yaw": round(float(np.arctan2(Rm[1, 0], Rm[0, 0])), 4)})
        except Exception as e:  # noqa: BLE001  (never break the sim loop for a diagnostic)
            print(f"[sim] hand boxes: {e}")
        return out

    def _on_request(self, msg):
        try:
            req = json.loads(msg.data)
            rid = req.get("id")
            if rid in self.done:          # a retry of a request we already executed: just answer again
                self.pub.publish(String(data=json.dumps(self.done[rid])))
                return
            fn = self.handlers[req["cmd"]]
            out = fn(req) if fn.__code__.co_argcount > 1 else fn()      # some commands take arguments
            out.update({"id": rid, "ok": True})
            self.done[rid] = out
            if len(self.done) > 100:
                self.done.pop(next(iter(self.done)))
        except Exception as e:  # noqa: BLE001  (report errors to the caller instead of crashing the sim)
            out = {"id": None, "ok": False, "error": str(e)}
            try:
                out["id"] = json.loads(msg.data).get("id")
            except Exception:  # noqa: BLE001
                pass
        self.pub.publish(String(data=json.dumps(out)))

    # --- helpers --------------------------------------------------------------------------------------
    def _target_xy(self):
        pos, _ = self.h.target.get_world_pose()
        return [float(pos[0]), float(pos[1])]

    def _move_goal(self, xy):
        z = self.h.cfg["table"]["center"][2] + self.h.cfg["table"]["size"][2] / 2 + 0.0005
        self.marker.set_world_pose(position=np.array([xy[0], xy[1], z]))
        self.goal_xy = [float(xy[0]), float(xy[1])]

    def _state(self):
        boxes = []
        for b in self.layout:
            pos, _ = self.h.obstacles[b["id"]].get_world_pose()
            boxes.append({**b, "displacement_m": float(np.linalg.norm(np.asarray(pos) - np.asarray(b["center"])))})
        objs = {"target": self._target_xy()}
        for name, rb in self.h.extras.items():
            pos, _ = rb.get_world_pose()
            objs[name] = [float(pos[0]), float(pos[1])]
        return {"target": {"xy": self._target_xy()}, "goal": {"xy": self.goal_xy}, "obstacles": boxes, "objects": objs}

    def _extra_xy(self):
        out = []
        for rb in self.h.extras.values():
            pos, _ = rb.get_world_pose()
            out.append([float(pos[0]), float(pos[1])])
        return out

    def _footprint_r(self, name):
        objects = load_yaml("config/objects.yaml")
        o = objects[self.h.cfg["target"]["object"]] if name == "target" else \
            objects[next(e["object"] for e in self.h.cfg.get("extra_objects", []) if e["name"] == name)]
        return 0.5 * float(np.hypot(o["dims_m"][0], o["dims_m"][1]))

    def _place_extras_random(self, target_xy):
        """Each extra object: a random spot in its spawn region (default: the target's), clear of everything placed."""
        placed = [(target_xy, self._footprint_r("target"))]
        for e in self.h.cfg.get("extra_objects", []):
            r = e.get("spawn_region", self.h.cfg["target"]["spawn_region"])
            me = self._footprint_r(e["name"])
            for _ in range(500):
                xy = [float(self.rng.uniform(*r["x"])), float(self.rng.uniform(*r["y"]))]
                if all(np.hypot(xy[0] - p[0], xy[1] - p[1]) > me + rr + 0.04 for p, rr in placed):
                    break
            place_extra(self.h, e["name"], xy, float(self.rng.uniform(-np.pi, np.pi)))
            placed.append((xy, me))

    def _set_layout(self, layout):
        set_obstacles(self.h, layout)
        self.layout = layout
        self.layout_seq += 1
        self._publish_known()

    def _relayout(self, can_xy=None):
        """New random boxes for the current can + goal (called after anything moved them)."""
        t = self.h.cfg["table"]
        top = t["center"][2] + t["size"][2] / 2
        set_obstacles(self.h, [])      # park first (not published: listeners only ever see complete layouts)
        can_xy = self._target_xy() if can_xy is None else can_xy
        ocfg = dict(self.h.cfg["obstacles"])          # boxes keep clear of the extra objects too
        ocfg["keep_clear"] = list(ocfg.get("keep_clear", [])) + [{"xy": xy, "r_m": ocfg["clearance_m"]}
                                                                  for xy in self._extra_xy()]
        self._set_layout(sample_layout(self.rng, ocfg, can_xy, self.goal_xy, self.camera_pos, t, top, self.can_height,
                                       target_half_m=self.target_half))

    # --- sensor realism: camera placement errors + drift ------------------------------------------------
    def _set_reality(self, req):
        name = req.get("profile", "ideal")
        prof = load_yaml("config/sim_reality.yaml")["profiles"][name]
        self.reality = name
        self.errors = {"camera": prof.get("fixed_camera_error"), "wrist_camera": prof.get("wrist_camera_error")}
        self.drift = prof.get("fixed_camera_drift")
        self.drift_t0 = self.h.world.current_time
        self._apply_camera_errors()
        return {"profile": name, "errors": self.errors, "drift": self.drift}

    def _apply_camera_errors(self):
        for key, T_nom in self.cam_nominal.items():
            e = self.errors.get(key) or {"xyz_mm": [0, 0, 0], "rpy_deg": [0, 0, 0]}
            if e.get("from_calibration"):
                # the TRUE camera = a real unit's calibrated pose (the twin): whatever it differs from the nominal
                # mount by is what the sim's own calibration has to find
                c = load_yaml(e["from_calibration"])["T_parent_optical"]
                set_camera_pose(self.h, key, make_transform(quat_to_matrix(c["quat_xyzw"]), c["xyz"]))
                continue
            xyz, rpy = np.array(e["xyz_mm"], float) / 1000, np.radians(e["rpy_deg"])
            if key == "camera" and self.drift:
                minutes = (self.h.world.current_time - self.drift_t0) / 60.0
                xyz = xyz + np.array(self.drift["xyz_mm_per_min"]) / 1000 * minutes
                rpy = rpy + np.radians(self.drift["rpy_deg_per_min"]) * minutes
            set_camera_pose(self.h, key, T_nom @ make_transform(rpy_matrix(*rpy), xyz))
        self.drift_applied = time.time()

    def _camera_error(self, req):
        """Move the fixed camera to its nominal pose + {xyz_mm, rpy_deg} (a bump: tests the drift monitor and its
        re-anchor). The believed pose (what ROS uses) is unchanged."""
        self.errors["camera"] = {"xyz_mm": [float(v) for v in req.get("xyz_mm", [0, 0, 0])],
                                 "rpy_deg": [float(v) for v in req.get("rpy_deg", [0, 0, 0])]}
        self._apply_camera_errors()
        return {"camera": self.errors["camera"]}

    def _hand_prop(self, req):
        """A stand-in for a person's hand (tests the hand-detection pause: pnp_perception/hand_detector_node, backend
        sim_marker): {present: bool, xy: [x, y] (base frame, m), z: height above the table (m, default 0.12),
        size: [l, w, h] (m)}. A MAGENTA box, visual only (no collider: it must not push anything), at /SimProps (outside
        /World: not reported as a hand-added obstacle). present false hides it."""
        from isaacsim.core.utils.stage import get_current_stage
        from pxr import Gf, UsdGeom
        stage = get_current_stage()
        path = "/SimProps/hand"
        if not req.get("present", True):
            if stage.GetPrimAtPath(path).IsValid():
                UsdGeom.Imageable(stage.GetPrimAtPath(path)).MakeInvisible()
            return {"present": False}
        size = [float(v) for v in req.get("size", [0.18, 0.09, 0.03])]
        t = self.h.cfg["table"]
        top = t["center"][2] + t["size"][2] / 2
        xy = [float(v) for v in req.get("xy", [0.2, 0.0])]
        z = top + float(req.get("z", 0.12))
        UsdGeom.Xform.Define(stage, "/SimProps")
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        cube.CreateDisplayColorAttr([Gf.Vec3f(0.95, 0.05, 0.85)])
        xf = UsdGeom.XformCommonAPI(cube)
        xf.SetTranslate(Gf.Vec3d(xy[0], xy[1], z))
        xf.SetScale(Gf.Vec3f(*size))
        UsdGeom.Imageable(cube).MakeVisible()
        return {"present": True, "xy": xy, "z": round(z, 4), "size": size}

    def _calibration_targets(self, req):
        """Show / hide the calibration boards: {hand_board: bool, table_board: bool} (missing keys: unchanged).
        They only exist during a calibration run, like the real ones."""
        from isaacsim.core.utils.stage import get_current_stage
        from pxr import UsdGeom
        out = {}
        for key in ("hand_board", "table_board"):
            path = self.h.paths.get(key)
            if not path:
                out[key] = None
                continue
            img = UsdGeom.Imageable(get_current_stage().GetPrimAtPath(path))
            if key in req:
                img.MakeVisible() if req[key] else img.MakeInvisible()
            out[key] = img.ComputeVisibility() != UsdGeom.Tokens.invisible
        return out

    def _visual_gap(self, req):
        """Signed distance from each link's visible surface to the target's visible shape (its convex hull: fine for
        the cans, boxes and bricks we pick) [mm]: > 0 a gap, < 0 the meshes overlap. Physics touches COLLISION shapes,
        so a grasp can look right or wrong while it's physically fine: this measures what you SEE."""
        from scipy.spatial import ConvexHull
        from scene import visual_surface_points
        name = req.get("object", "target")
        obj = visual_surface_points(self.h.paths["target"] if name == "target" else
                                    self.h.paths["extra_objects"][name], spacing=0.001)
        hull = ConvexHull(obj)
        lo, hi = obj.min(0) - 0.01, obj.max(0) + 0.01
        out = {}
        for link in req["links"]:
            P = visual_surface_points(f"{self.h.paths['robot_links']}/{link}", spacing=float(req.get("spacing", 0.0003)))
            P = P[np.all((P > lo) & (P < hi), axis=1)]                 # only the part next to the object
            if len(P) == 0:
                out[link] = None
                continue
            d = np.max(P @ hull.equations[:, :3].T + hull.equations[:, 3], axis=1)    # signed distance (convex)
            out[link] = {"gap_mm": round(float(d.min()) * 1000, 3), "points": int(len(P)),
                         "touching": int(np.sum(np.abs(d) < 0.0005))}
        return {"links": out, "object_points": int(len(obj))}

    def _link_pose(self, req):
        """The TRUE world pose of a robot link in Isaac (sim only): {link: name} -> 4x4. Compare with ROS TF to check
        that the simulated robot matches the robot description (joint axes / zeros / signs after a URDF import)."""
        return {"link": req["link"], "T": prim_world_T(f"{self.h.paths['robot_links']}/{req['link']}").round(6).tolist()}

    def _camera_truth(self):
        """TRUE optical-frame poses of the cameras in the robot base frame (sim only: what calibration should find)."""
        out = {"profile": self.reality, "cameras": {k: true_camera_pose(self.h, k).round(6).tolist()
                                                    for k in self.cam_nominal}}
        if "wrist_camera" in self.cam_nominal:     # eye-in-hand: the camera in its hand link frame
            out["wrist_in_parent"] = true_camera_pose(self.h, "wrist_camera", relative_to_parent=True).round(6).tolist()
        out["targets"] = {k: prim_world_T(self.h.paths[k]).round(6).tolist()      # board corner frames, world
                          for k in ("hand_board", "table_board") if k in self.h.paths}
        return out

    # --- commands -------------------------------------------------------------------------------------
    def _randomize_target(self):
        r = self.h.cfg["target"]["spawn_region"]
        xy = [float(self.rng.uniform(*r["x"])), float(self.rng.uniform(*r["y"]))]
        yaw = float(self.rng.uniform(-np.pi, np.pi))
        place_target(self.h, xy, yaw)
        return {"xy": xy, "yaw": yaw, "target": {"xy": xy, "yaw": yaw}}

    def _randomize_goal(self, avoid_xy=None):
        g = self.h.cfg["goal"]
        avoid = [self._target_xy() if avoid_xy is None else avoid_xy] + self._extra_xy()     # every object
        for _ in range(200):
            xy = [float(self.rng.uniform(*g["region"]["x"])), float(self.rng.uniform(*g["region"]["y"]))]
            if all(np.hypot(xy[0] - a[0], xy[1] - a[1]) >= g["min_separation_m"] for a in avoid):
                break
        self._move_goal(xy)
        return {"goal": {"xy": xy}}

    def _randomize_scene(self):
        if self.layout and self.obstacles_on:
            set_obstacles(self.h, [])  # park the boxes so the can never lands on / inside one (re-laid out below)
        elif self.layout:
            self._set_layout([])       # e.g. after block_goal with obstacles off: clear the table
        t = self._randomize_target()
        self._place_extras_random(t["target"]["xy"])
        g = self._randomize_goal(avoid_xy=t["target"]["xy"])
        if self.obstacles_on:
            self._relayout(t["target"]["xy"])
        return {"target": t["target"], "goal": g["goal"], "obstacles": self.layout}

    def _randomize_obstacles(self):
        self.obstacles_on = True
        self._relayout()
        return {"obstacles": self.layout}

    def _clear_obstacles(self):
        self.obstacles_on = False
        self._set_layout([])
        return {"obstacles": []}

    def _block_goal(self):
        spec = max(self.h.cfg["obstacles"]["pool"], key=lambda s: s["size"][2])       # the tallest box
        t = self.h.cfg["table"]
        top = t["center"][2] + t["size"][2] / 2
        box = {"id": spec["id"], "size": [float(v) for v in spec["size"]],
               "center": [self.goal_xy[0], self.goal_xy[1], top + spec["size"][2] / 2], "yaw": 0.0}
        self._set_layout([box])
        return {"obstacles": self.layout}

    def _set_obstacles(self, req):
        """Boxes at EXACT spots: {"boxes": [{"id": pool id, "xy": [x, y], "yaw": rad}, ...]} (the others parked).
        For tests that need an obstacle at a given distance from an object (data/workcell_stress.py "too close"):
        the random layouts keep obstacles.clearance_m from every object, so they never make that case."""
        t = self.h.cfg["table"]
        top = t["center"][2] + t["size"][2] / 2
        pool = {spec["id"]: spec for spec in self.h.cfg["obstacles"]["pool"]}
        boxes = []
        for b in req.get("boxes", []):
            if b["id"] not in pool:
                raise KeyError(f"no pool box '{b['id']}' ({', '.join(pool)})")
            boxes.append(_make(pool[b["id"]], b["xy"], float(b.get("yaw", 0.0)), top))
        self.obstacles_on = bool(boxes)
        self._set_layout(boxes)
        return {"obstacles": self.layout}

    def _place_object(self, req):
        """Put one object (name: "target" or an extra_objects name) upright at xy [m] / yaw [rad] on the table: tests
        that need an object at a particular spot (e.g. data/workcell_stress.py: the edge of the camera's view)."""
        name, xy, yaw = req.get("name", "target"), [float(v) for v in req["xy"]], float(req.get("yaw", 0.0))
        if name == "target":
            place_target(self.h, xy, yaw)
        elif name in self.h.extras:
            place_extra(self.h, name, xy, yaw)
        else:
            raise KeyError(f"no object '{name}' (target, {', '.join(self.h.extras)})")
        return {"name": name, "xy": xy, "yaw": yaw}

    def _away(self):
        t = self.h.cfg["table"]
        floor = t["center"][2] + t["size"][2] / 2 - t["height_above_floor"]
        place_target(self.h, (-1.5, -1.5), 0.0, z=floor + 0.06)
        return {"xy": [-1.5, -1.5], "away": True}

    def _reset(self):
        if self.layout and self.obstacles_on:
            set_obstacles(self.h, [])
        elif self.layout:
            self._set_layout([])
        place_target(self.h, self.h.cfg["target"]["default_xy"], 0.0)
        for e in self.h.cfg.get("extra_objects", []):
            place_extra(self.h, e["name"], e["default_xy"], float(e.get("yaw", 0.0)))
        self._move_goal(self.h.cfg["goal"]["xy"])
        if self.obstacles_on:
            self._relayout(self.h.cfg["target"]["default_xy"])
        return {"xy": list(self.h.cfg["target"]["default_xy"]), "yaw": 0.0, **self._state()}
