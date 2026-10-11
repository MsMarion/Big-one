"""Can the gripper pick this object here at all? A check BEFORE any motion: the space the open hand needs at a grasp,
tested against everything the camera sees standing on the table (no cut-out around the target).

Why: the sim's random layouts keep obstacles >= obstacles.clearance_m (5.5 cm) from every object, so "an obstacle
is too close to pick it" never came up there. A real desk has it: a cube 1 cm beside a brick, the cut-out around
the target erased the cube's near side, MoveIt planned through the hole, the jaw hit the cube (CONTROL_FAILED).

The space one grasp needs, in the TCP frame (z = approach: toward the object, y = the jaws' closing direction):
  fixed jaw / open jaw   the jaw slabs at the grasp, opened to `open_width`: one moving jaw against a fixed one
                         (gripper.fixed_jaw_y_m: SO-101) or two fingers at +-open/2 (FR3); x = the jaw width, z = the
                         finger box (config gripper.hand_boxes, measured from the collision meshes)
  servo housing / hand   the hand body behind the jaws (hand_boxes "body")
  descent column         all of the above swept back along -z by the approach height: the straight line down
The target's own points (inside its box + ignore_pad) don't count. A grasp is CLEAR when no other point lies in
any of its volumes. pick_check() tries every candidate grasp and says which are clear, and if none, why:
"obstacle 1.8 cm from lego 2x2, the open jaw needs 2.4 cm".
"""
from __future__ import annotations

import numpy as np

from .object_grasp import grasps_from_pose, object_extents, shift_for_fixed_jaw, with_flips


def gripper_volumes(gripper_cfg: dict, open_width: float, approach_height: float, inflate: float = 0.0) -> list:
    """-> [(name, center[3], half[3])] boxes in the TCP frame: the open jaws, the hand body, each swept back by the
    approach (the descent column), grown by `inflate` (the obstacle map's voxels make obstacles look that much
    bigger to MoveIt). From gripper.hand_bands (measured from the meshes, band by band) where the profile has them,
    else the coarse gripper.hand_boxes."""
    hb = gripper_cfg.get("hand_bands")
    if hb:
        out = []
        shift = open_width - float(hb["ref_width_m"])           # the moving jaw at this opening (it slides along +y)
        for part in ("fixed", "moving"):
            for z_hi, z_lo, x_lo, x_hi, y_lo, y_hi in hb.get(part, []):
                if part == "moving":
                    y_lo, y_hi = y_lo + shift, y_hi + shift
                    name = "open jaw"
                else:
                    name = "fixed jaw" if z_hi > -0.045 else "gripper body"
                lo = np.array([x_lo, y_lo, z_lo - approach_height]) - inflate     # swept back: the descent
                hi = np.array([x_hi, y_hi, z_hi]) + inflate
                out.append((name, (lo + hi) / 2, (hi - lo) / 2))
        return out
    boxes = {b["name"]: b for b in gripper_cfg.get("hand_boxes", [])}
    fing, body = boxes.get("fingers"), boxes.get("body")
    out = []

    def add(name, x, y, z):
        z = (z[0] - approach_height, z[1])                     # swept back along the approach: the descent
        lo, hi = np.array([x[0], y[0], z[0]]), np.array([x[1], y[1], z[1]])
        out.append((name, (lo + hi) / 2, (hi - lo) / 2))
    if fing is not None:
        c, h = np.asarray(fing["center"], float), np.asarray(fing["half"], float)
        x, z = (c[0] - h[0], c[0] + h[0]), (c[2] - h[2], c[2] + h[2])
        if "fixed_jaw_y_m" in gripper_cfg:                     # one fixed jaw, one moving jaw (SO-101)
            fy = float(gripper_cfg["fixed_jaw_y_m"])
            t = max(0.003, fy - (c[1] - h[1]))                 # the fixed jaw: from its inner face to the box's side
            add("fixed jaw", x, (fy - t, fy), z)
            add("open jaw", x, (fy + open_width, fy + open_width + t), z)
        else:                                                  # two fingers, symmetric (FR3)
            t = h[1]
            add("finger", x, (-open_width / 2 - t, -open_width / 2), z)
            add("finger", x, (open_width / 2, open_width / 2 + t), z)
    if body is not None:
        c, h = np.asarray(body["center"], float), np.asarray(body["half"], float)
        add("servo housing" if "fixed_jaw_y_m" in gripper_cfg else "hand", (c[0] - h[0], c[0] + h[0]),
            (c[1] - h[1], c[1] + h[1]), (c[2] - h[2], c[2] + h[2]))
    return out


def check_inflate(scene_cfg: dict) -> float:
    """How much the hand's volumes grow for the check -- the same on the workcell page and in the pick, so the two
    agree: the obstacle map's voxel (MoveIt sees obstacles as voxels, up to one voxel beyond the surface) +
    perception.pickability_margin_m (the wrist camera's close look still corrects the pose: ~1 mm but 3-6 deg on
    the SO-101 -- a grasp clearing by less than that was refused at the close look after the page said pickable)."""
    p = scene_cfg.get("perception", {})
    return float(p.get("obstacle_cloud", {}).get("voxel_m", 0.02)) + float(p.get("pickability_margin_m", 0.0))


def _box_sdf(p_local: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Signed distance of points (box frame) to an axis-aligned box: < 0 inside (the depth), > 0 outside."""
    q = np.abs(p_local) - half
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)
    return outside + inside


def target_mask(points: np.ndarray, T_base_obj: np.ndarray, extents, pad: float) -> np.ndarray:
    """Points inside the target's box grown by `pad` (its own points: not obstacles for its pick)."""
    p = (points - T_base_obj[:3, 3]) @ T_base_obj[:3, :3]
    return np.all(np.abs(p) <= np.asarray(extents, float) / 2 + pad, axis=1)


def check_grasp(points: np.ndarray, T_base_tcp: np.ndarray, volumes) -> dict:
    """-> {clear, depth (how deep the worst point lies inside a volume, m), volume (its name), point (base),
    clearance (the nearest point's distance outside every volume, m; 0 when blocked)}."""
    if not len(points):
        return {"clear": True, "depth": 0.0, "volume": None, "point": None, "clearance": 1.0}
    p_tcp = (points - T_base_tcp[:3, 3]) @ T_base_tcp[:3, :3]
    worst = {"clear": True, "depth": 0.0, "volume": None, "point": None}
    nearest = np.inf
    for name, c, h in volumes:
        d = _box_sdf(p_tcp - c, h)
        i = int(np.argmin(d))
        nearest = min(nearest, float(d[i]))
        if d[i] < 0 and -d[i] > worst["depth"]:
            worst = {"clear": False, "depth": float(-d[i]), "volume": name, "point": points[i]}
    worst["clearance"] = max(0.0, nearest)
    return worst


def pick_check(points, obj_cfg: dict, T_base_obj, gripper_cfg: dict, gripper_model, tcp_yaw: float = 0.0,
               yaws=(0.0,), max_tilt_deg: float = 20.0, open_margin=None, approach_height: float = 0.04,
               ignore_pad: float = 0.004, label: str = "", inflate: float = 0.0, flips: bool = False) -> dict:
    """Every grasp candidate of the object at T_base_obj vs the points standing on the table (base frame).
    -> {pickable, clear: [grasp indices], grasps, results, why}. `why` names the obstacle when none is clear."""
    pts = np.asarray(points, float).reshape(-1, 3)
    ext = object_extents(obj_cfg)
    pts = pts[~target_mask(pts, T_base_obj, ext, ignore_pad)] if len(pts) else pts
    grasps, why = grasps_from_pose(T_base_obj, obj_cfg, list(yaws), tcp_yaw, max_tilt_deg,
                                   float(gripper_cfg["max_width_m"]))
    if not grasps:
        return {"pickable": False, "clear": [], "grasps": [], "results": [], "why": f"no grasp: {why}"}
    if flips:                                              # + the hand turned 180 deg (grasp scoring)
        grasps = with_flips(grasps)
    grasps = shift_for_fixed_jaw(grasps, gripper_model.held_offset_y, gripper_model.approach_clearance)
    margin = float(gripper_cfg.get("open_margin_m", 0.0)) if open_margin is None else float(open_margin)
    results = []
    for g in grasps:
        w = (min(float(gripper_cfg["max_width_m"]), g.width_m + 2 * margin) if margin > 0
             else float(gripper_cfg["max_width_m"]))
        results.append(check_grasp(pts, g.T_base_tcp, gripper_volumes(gripper_cfg, w, approach_height, inflate)))
    clear = [i for i, r in enumerate(results) if r["clear"]]
    out = {"pickable": bool(clear), "clear": clear, "grasps": grasps, "results": results, "why": ""}
    if not clear:
        # how far the obstacle is from the object's side, and how far it would have to be for the best grasp to clear:
        # push its points away along the line from the object through it until they leave every volume
        c, a, half = T_base_obj[:3, 3], T_base_obj[:3, :3][:2, :2], ext[:2] / 2

        def side_gap(p):                                   # 2-D distance of points to the object's footprint
            q = np.abs((p[:, :2] - c[:2]) @ a) - half
            return np.linalg.norm(np.maximum(q, 0.0), axis=1)
        best = None
        for g, res in zip(grasps, results):
            w = (min(float(gripper_cfg["max_width_m"]), g.width_m + 2 * margin) if margin > 0
                 else float(gripper_cfg["max_width_m"]))
            vols = gripper_volumes(gripper_cfg, w, approach_height, inflate)
            p_tcp = (pts - g.T_base_tcp[:3, 3]) @ g.T_base_tcp[:3, :3]
            inside = np.zeros(len(pts), bool)
            for _, vc, vh in vols:
                inside |= _box_sdf(p_tcp - vc, vh) < 0
            blk = pts[inside]
            # the obstacle: the points near the blocking ones (its whole near side, not only what's inside)
            obst = pts[np.min(np.linalg.norm(pts[:, None, :2] - blk[None, ::max(1, len(blk) // 50), :2], axis=2),
                              axis=1) < 0.02]
            j = int(np.argmin(side_gap(obst)))
            gap = float(side_gap(obst[j:j + 1])[0])
            d = np.append(obst[j, :2] - c[:2], 0.0)
            d /= max(np.linalg.norm(d), 1e-9)
            push, vol = 0.0, res["volume"]
            for p in p_tcp[inside]:
                s, name = _exit_along(p, g.T_base_tcp[:3, :3].T @ d, vols)
                if s > push:
                    push, vol = s, name
            if best is None or gap + push < best[1]:
                best = (gap, gap + push, vol, obst[j])
        gap, need, vol, pt = best
        out["why"] = (f"obstacle {gap * 100:.1f} cm from {label or 'the object'}, the {vol} needs {need * 100:.1f} cm "
                      f"(every grasp blocked: {len(results)} tried)")
        out["obstacle"] = [float(v) for v in pt]
    return out


def _exit_along(p, d, volumes, rounds: int = 4):
    """How far point p (TCP frame) must move along unit direction d to be outside every volume (it can leave one box
    straight into another: repeat a few rounds). -> (distance, the volume it left last)."""
    s, name = 0.0, None
    for _ in range(rounds):
        moved = False
        for vname, c, h in volumes:
            q = p + s * d - c
            if np.all(np.abs(q) < h):
                t = min(((h[i] if d[i] > 0 else -h[i]) - q[i]) / d[i] for i in range(3) if abs(d[i]) > 1e-9)
                s += max(t, 0.0) + 1e-4
                name, moved = vname, True
        if not moved:
            break
    return s, name
