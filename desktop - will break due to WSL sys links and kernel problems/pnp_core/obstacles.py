"""Known box obstacles (Phase 5, Layer A): geometry checks and random *solvable* layouts.

A box is a dict  {"id": str, "size": [sx, sy, sz], "center": [x, y, z], "yaw": rad}  in the robot base frame,
standing on the table (only rotated about z). The same dicts travel everywhere: the sim spawns them, publishes
them as the "known obstacles" (/scene/known_obstacles), and the pick & place node turns them into MoveIt
collision boxes. On a real robot the list would come from a measured/declared map instead of the simulator.

"Solvable" = what we guarantee so that a failure means the PLANNER failed, not that the task was impossible:
  * the gripper fits around the can and around the goal  (box footprint >= clearance_m from both centers)
  * the boxes stay away from the robot base               (>= robot_clearance_m)
  * the camera can still see the can and the goal        (no box crosses the camera -> can / goal rays)
  * boxes don't overlap each other, and stay on the table
Pure numpy (no ROS / Isaac): unit-tested in tests/test_obstacles.py.
"""

from __future__ import annotations

import numpy as np


def _to_box_frame(p_xy, box) -> np.ndarray:
    c, s = np.cos(box["yaw"]), np.sin(box["yaw"])
    d = np.asarray(p_xy, float)[:2] - np.asarray(box["center"], float)[:2]
    return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])


def footprint_distance(p_xy, box) -> float:
    """Distance in XY from a point to the box footprint (0 if the point is inside it)."""
    q = np.abs(_to_box_frame(p_xy, box)) - np.asarray(box["size"][:2], float) / 2
    return float(np.linalg.norm(np.maximum(q, 0.0)))


def footprint_corners(box) -> np.ndarray:
    hx, hy = box["size"][0] / 2, box["size"][1] / 2
    c, s = np.cos(box["yaw"]), np.sin(box["yaw"])
    local = np.array([[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy]])
    return local @ np.array([[c, s], [-s, c]]) + np.asarray(box["center"][:2])


def segment_hits_box(p0, p1, box, pad: float = 0.0) -> bool:
    """Does the 3D segment p0 -> p1 pass through the box (inflated by `pad` on every side)? Slab test in the
    box frame (the box is only yawed, so z is unchanged)."""
    a = np.r_[_to_box_frame(p0, box), p0[2] - box["center"][2]]
    b = np.r_[_to_box_frame(p1, box), p1[2] - box["center"][2]]
    half = np.asarray(box["size"], float) / 2 + pad
    t0, t1, d = 0.0, 1.0, b - a
    for k in range(3):
        if abs(d[k]) < 1e-12:
            if abs(a[k]) > half[k]:
                return False
            continue
        ta, tb = sorted(((-half[k] - a[k]) / d[k], (half[k] - a[k]) / d[k]))
        t0, t1 = max(t0, ta), min(t1, tb)
        if t0 > t1:
            return False
    return True


def _radius_xy(box) -> float:
    return float(np.hypot(box["size"][0], box["size"][1]) / 2)


def layout_problems(boxes, can_xy, goal_xy, camera_pos, table, table_top, can_height, cfg) -> list[str]:
    """Everything that makes a layout unsolvable or unfair (empty list = OK). Also used by the tests."""
    probs = []
    tc, ts = np.asarray(table["center"][:2]), np.asarray(table["size"][:2])
    lo, hi = tc - ts / 2 + 0.02, tc + ts / 2 - 0.02
    sight = [np.r_[xy, table_top + can_height * f] for xy in (can_xy, goal_xy) for f in (0.5, 1.0)]
    # keep_clear: spots that must stay uncovered AND visible, e.g. the AprilTag the calibration drift monitor watches
    keep = cfg.get("keep_clear", [])
    sight += [np.r_[k["xy"], table_top + 0.005] for k in keep]
    for i, b in enumerate(boxes):
        for k in keep:
            if footprint_distance(k["xy"], b) < k["r_m"]:
                probs.append(f"{b['id']}: covers a keep-clear spot {k['xy']}")
        if footprint_distance(can_xy, b) < cfg["clearance_m"] and not b.get("near_target"):
            probs.append(f"{b['id']}: too close to the can")
        if footprint_distance(goal_xy, b) < cfg["clearance_m"]:
            probs.append(f"{b['id']}: too close to the goal")
        if footprint_distance((0.0, 0.0), b) < cfg["robot_clearance_m"]:
            probs.append(f"{b['id']}: too close to the robot base")
        corners = footprint_corners(b)
        if (corners < lo).any() or (corners > hi).any():
            probs.append(f"{b['id']}: off the table")
        if any(segment_hits_box(np.asarray(camera_pos, float), p, b, pad=0.01) for p in sight):
            probs.append(f"{b['id']}: blocks the camera's view")
        for o in boxes[:i]:
            gap = np.hypot(*(np.asarray(b["center"][:2]) - np.asarray(o["center"][:2])))
            if gap < _radius_xy(b) + _radius_xy(o) + 0.03:
                probs.append(f"{b['id']}: overlaps {o['id']}")
    return probs


def _near_target(rng, cfg, can_xy, goal_xy, camera_pos, table, table_top, can_height, target_half, tries):
    """One pool box (not a wall) with a face too_close_gap_m from the target's side, facing it; or None."""
    lo, hi = cfg.get("too_close_gap_m", [0.004, 0.02])
    pool = [s for s in cfg["pool"] if s.get("role") != "between"] or list(cfg["pool"])
    for _ in range(tries):
        spec = pool[int(rng.integers(len(pool)))]
        a = float(rng.uniform(-np.pi, np.pi))
        d = np.array([np.cos(a), np.sin(a)])
        gap = float(rng.uniform(lo, hi))
        xy = np.asarray(can_xy, float) + d * (target_half + gap + float(spec["size"][0]) / 2)
        b = {**_make(spec, xy, a, table_top), "near_target": True, "gap_m": round(gap, 4)}
        if not layout_problems([b], can_xy, goal_xy, camera_pos, table, table_top, can_height, cfg):
            return b
    return None


def _make(spec, xy, yaw, table_top) -> dict:
    size = [float(v) for v in spec["size"]]
    return {"id": spec["id"], "size": size, "center": [float(xy[0]), float(xy[1]), table_top + size[2] / 2],
            "yaw": float(yaw)}


def sample_layout(rng: np.random.Generator, cfg: dict, can_xy, goal_xy, camera_pos, table, table_top,
                  can_height, tries: int = 300, target_half_m: float = 0.0) -> list[dict]:
    """Random 1..N boxes from cfg["pool"]. A pool entry with role "between" is first tried as a wall across
    the straight line from the can to the goal (the interesting case: the arm must go over or around it).
    Boxes that can't be placed after `tries` samples are left out, so the result may be shorter than asked.

    cfg too_close_prob (default 0): that often ONE box stands right beside the target instead, its face
    too_close_gap_m [lo, hi] from the target's side (target_half_m: the target's half size), marked near_target:
    closer than the open jaw needs, so the right answer is to refuse the pick (BLOCKED), not to try it. The other
    rules (goal, robot, table, camera, other boxes) still hold for it."""
    if cfg.get("too_close_prob", 0.0) > 0 and rng.uniform() < cfg["too_close_prob"]:
        near = _near_target(rng, cfg, can_xy, goal_xy, camera_pos, table, table_top, can_height, target_half_m, tries)
        if near is not None:
            rest = sample_layout(rng, {**cfg, "too_close_prob": 0.0,
                                       "count": [max(0, cfg["count"][0] - 1), max(0, cfg["count"][1] - 1)]},
                                 can_xy, goal_xy, camera_pos, table, table_top, can_height, tries)
            boxes = [near]
            for b in rest:
                if not layout_problems(boxes + [b], can_xy, goal_xy, camera_pos, table, table_top, can_height, cfg):
                    boxes.append(b)
            return boxes
    n_lo, n_hi = cfg["count"]
    n = int(rng.integers(n_lo, n_hi + 1))
    pool = list(cfg["pool"])
    chosen = [pool[i] for i in rng.choice(len(pool), size=min(n, len(pool)), replace=False)]
    chosen.sort(key=lambda s: s.get("role") != "between")         # place the wall first
    can_xy, goal_xy = np.asarray(can_xy, float), np.asarray(goal_xy, float)
    reg = cfg["region"]
    boxes: list[dict] = []
    for spec in chosen:
        for k in range(tries):
            if spec.get("role") == "between" and k < tries // 2:
                u = goal_xy - can_xy
                dist = float(np.linalg.norm(u))
                u = u / max(dist, 1e-9)
                perp = np.array([-u[1], u[0]])
                xy = (can_xy + goal_xy) / 2 + u * rng.uniform(-0.02, 0.02) + perp * rng.uniform(-0.05, 0.05)
                yaw = float(np.arctan2(u[1], u[0]))                 # thin side (size x) along can -> goal
            else:
                xy = np.array([rng.uniform(*reg["x"]), rng.uniform(*reg["y"])])
                yaw = float(rng.uniform(-np.pi, np.pi))
            cand = _make(spec, xy, yaw, table_top)
            if not layout_problems(boxes + [cand], can_xy, goal_xy, camera_pos, table, table_top, can_height, cfg):
                boxes.append(cand)
                break
    return boxes
