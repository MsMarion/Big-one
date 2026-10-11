"""Which resting faces can a robot turn an object to, and in how many moves? (gripper geometry only; the arm's reach
is tools/grasp_study.py)

    python3 /repo/tools/reorient_graph.py                         every object of every robot's scenes
    python3 /repo/tools/reorient_graph.py --robot fr3 --object cracker_box

docs/learning/19_reorient.md
"""
import argparse
import sys

sys.path.insert(0, "/repo")
from pnp_core.catalog import load_robot, load_yaml  # noqa: E402
from pnp_core.reorient import plan_reorientation, rest_poses, transitions  # noqa: E402

DEFAULT = {"fr3": ["tomato_soup_can", "cracker_box"], "so101": ["lego_2x4", "lego_2x2", "lego_1x2"]}


def report(robot: str, name: str):
    obj = load_yaml("config/objects.yaml")[name]
    g = load_robot(robot)["gripper"]
    rests, moves = rest_poses(obj), transitions(obj, g)
    print(f"\n{robot} / {name}  (gripper opening {g['max_width_m'] * 1000:.0f} mm)")
    for r in rests.values():
        print(f"   {r.name:12s} {r.label:40s} center {r.height_m * 1000:5.1f} mm up, "
              f"footprint {r.footprint_m[0] * 1000:.0f} x {r.footprint_m[1] * 1000:.0f} mm")
    print("   one move: " + (", ".join(f"{m.frm} -> {m.to} (hand {m.hand})" for m in moves) or "none"))
    for t in rests:
        if t == "upright":
            continue
        p = plan_reorientation(obj, g, "upright", t, moves)
        print(f"   upright -> {t:12s} " + ("not possible with this gripper" if p is None else
                                           " -> ".join(["upright"] + [m.to for m in p]) + f"   ({len(p)} moves)"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot")
    ap.add_argument("--object")
    a = ap.parse_args()
    for robot, objs in DEFAULT.items():
        if a.robot and robot != a.robot:
            continue
        for name in ([a.object] if a.object else objs):
            report(robot, name)


if __name__ == "__main__":
    main()
