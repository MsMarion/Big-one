"""The FOCUS object on /perception/focus (std_msgs/String, latched): which object every perception node works on, and
optionally WHERE it is ("this one, near here": with two of a class the hint decides).

    "lego_2x4"                                         the class only (the old form, still accepted)
    {"object": "lego_2x4", "xy": [0.20, 0.07]}         + a near-here hint (base frame, m)

Published by pick_place_node (a pick starting) and workcell_node (an object clicked on the page / in RViz).
Read by the detector (which instance's mask the pose estimator gets), the 6-D pose estimator (its mesh; a new hint ->
register afresh), the obstacle cloud (which detection to carve out of the map: the others stay obstacles), the wrist
refine node.
"""
import json
import math

HINT_RADIUS_M = 0.06            # an instance within this of the hint is "the one" (pick_place_node pick_hint_radius_m)


def focus_message(name: str, xy=None) -> str:
    if xy is None:
        return name
    return json.dumps({"object": name, "xy": [round(float(xy[0]), 4), round(float(xy[1]), 4)]})


def parse_focus(data: str):
    """-> (object name, (x, y) or None)."""
    s = (data or "").strip()
    if s.startswith("{"):
        try:
            d = json.loads(s)
            xy = d.get("xy")
            return str(d.get("object", "")).strip(), (tuple(map(float, xy)) if xy and len(xy) == 2 else None)
        except (ValueError, TypeError):
            return "", None
    return s, None


def nearest(items, hint, key=lambda it: it, radius: float = HINT_RADIUS_M):
    """The item whose key(item) (an (x, y)) is nearest the hint within radius; all items without a hint."""
    if hint is None:
        return list(items)
    best = min(items, key=lambda it: math.dist(key(it), hint), default=None)
    return [best] if best is not None and math.dist(key(best), hint) <= radius else []
