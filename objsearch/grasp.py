"""Can the SO-ARM101 gripper pick this object up off the table?

dims_mm are [length, width, height] as the object normally sits (catalog convention), so height is the
third value and the jaws close across the narrower of length/width. grasp_pose="flat" overrides that for
things that end up lying on their largest face (height = smallest dimension). Limits live in
configs/objsearch.yaml -> gripper.

    good        height >= min_height and min_width <= jaw width <= max_open - clearance
    borderline  within the tolerance band below/above those limits (needs a careful grasp)
    no          too flat, too thin to grip, or wider than the jaws open
"""
from . import config

VERDICTS = ("good", "borderline", "no")


def limits():
    g = {"max_open_mm": 55, "clearance_mm": 8, "min_width_mm": 10, "min_height_mm": 10, "tolerance_mm": 3}
    g.update(config.load().get("gripper", {}))
    return g


def assess(dims_mm, pose=None, g=None):
    g = g or limits()
    if not dims_mm or len(dims_mm) != 3:
        return {"verdict": None, "reasons": ["size unknown"]}
    l, w, h = (float(d) for d in dims_mm)
    if pose == "flat":
        lo, mid, _ = sorted((l, w, h))
        height, width = lo, mid
    else:
        height, width = h, min(l, w)
    tol, reasons, worst = g["tolerance_mm"], [], 0

    def flag(level, why):
        nonlocal worst
        worst = max(worst, level)
        reasons.append(why)

    if height < g["min_height_mm"] - tol:
        flag(2, f"too flat: {height:g} mm tall (needs >= {g['min_height_mm']:g})")
    elif height < g["min_height_mm"]:
        flag(1, f"low: {height:g} mm tall, fingertips barely get purchase")
    if width < g["min_width_mm"] - tol:
        flag(2, f"too thin to grip: {width:g} mm across the jaws")
    elif width < g["min_width_mm"]:
        flag(1, f"narrow: {width:g} mm across the jaws, little room for position error")
    if width > g["max_open_mm"]:
        flag(2, f"too wide: {width:g} mm, jaws open ~{g['max_open_mm']:g} mm")
    elif width > g["max_open_mm"] - g["clearance_mm"]:
        flag(1, f"tight fit: {width:g} mm leaves < {g['clearance_mm']:g} mm clearance")
    return {"verdict": VERDICTS[worst], "height_mm": height, "grasp_width_mm": width,
            "pose": pose or "as listed", "reasons": reasons}


FILTERS = {"good": ("good",), "borderline": ("borderline",), "no": ("no",), "ok": ("good", "borderline")}


def annotate(objs, g=None):
    """Add o["grasp"] to each object dict (from list_objects / get_object_full)."""
    g = g or limits()
    for o in objs:
        o["grasp"] = assess(o.get("dims_mm"), o.get("grasp_pose"), g)
    return objs


def counts(objs):
    out = {"good": 0, "borderline": 0, "no": 0, "unknown": 0}
    for o in objs:
        out[o["grasp"]["verdict"] or "unknown"] += 1
    return out
