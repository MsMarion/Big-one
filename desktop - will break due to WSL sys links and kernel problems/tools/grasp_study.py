"""Which grasp of an object works best on THIS robot? Gripper checks + arm reach over the work area, ranked.
(docs/learning/20_grasp_study.md)

    python3 /repo/tools/grasp_study.py --robot fr3 --object cracker_box            (+ --rest upright | all)
    python3 /repo/tools/grasp_study.py --all                                        every object of both cells
    -> data/grasp_study/<robot>_<object>.json + data/grasp_study/index.html (also served by the workcell page at
       http://localhost:8088/grasp_study)

For each candidate grasp (pnp_core/grasp_study.py: closing side x approach tilt x depth x off-center) the object is
put on a grid of positions and yaws over the cell's pick + place areas; at each, numeric IK (the robot's URDF,
pnp_core/urdf_kin.py) must reach the grasp AND the pre-grasp (the robot's grasp_motion.approach_height_m back along
the approach), with every arm joint above the table. Reach = the fraction of those placements that work. Ranked by reach, then by how little gravity
twists the object in the fingers, then by how vertical the approach is. Not checked here: collisions of the arm with
itself / obstacles and the motion between poses (MoveIt does that at run time) -- this is a quick map, not a plan.
"""
import argparse
import json
import math
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, "/repo")
from pnp_core.catalog import load_robot, load_yaml, repo_root  # noqa: E402
from pnp_core.geometry import rot_z  # noqa: E402
from pnp_core.grasp_study import candidates, static_rank  # noqa: E402
from pnp_core.reorient import rest_poses  # noqa: E402
from pnp_core.urdf_kin import Chain, _axis_angle  # noqa: E402

CELLS = {"fr3": ("config/scenes/fr3_workcell.yaml", ["tomato_soup_can", "cracker_box"]),
         "so101": ("config/scenes/so101_workcell.yaml", ["lego_2x4", "lego_2x2", "lego_1x2"])}
OUT = repo_root() / "data" / "grasp_study"


def robot_urdf(cfg) -> str:
    if cfg.get("description"):
        from pnp_core.robot_description import urdf
        return urdf(cfg, repo_root())
    xacro = "/ws/install/pnp_bringup/share/pnp_bringup/urdf/fr3_sim.urdf.xacro"        # the FR3 (bringup's)
    home = " ".join(str(v) for v in cfg["arm"]["home"])
    return subprocess.run(["bash", "-c", f"source /opt/ros/jazzy/setup.bash && source /ws/install/setup.bash && xacro {xacro} home:='{home}' "
                                         f"hardware:=isaac"], capture_output=True, text=True, check=True).stdout


def joint_points(chain: Chain, q):
    """World positions of every joint origin along the chain (to check the arm stays above the table)."""
    T, i, pts = np.eye(4), 0, []
    for j in chain.joints:
        T = T @ j.T_origin
        pts.append(T[:3, 3].copy())
        if j.type in ("revolute", "continuous"):
            M = np.eye(4)
            M[:3, :3] = _axis_angle(j.axis, q[i])
            T, i = T @ M, i + 1
        elif j.type == "prismatic":
            M = np.eye(4)
            M[:3, 3] = j.axis / np.linalg.norm(j.axis) * q[i]
            T, i = T @ M, i + 1
    return np.array(pts)


def study(robot, obj_name, rests=("upright",), grid=4, n_yaws=3, top=4, max_cands=30, log=print):
    cfg = load_robot(robot)
    scene = load_yaml(CELLS[robot][0])
    obj = load_yaml("config/objects.yaml")[obj_name]
    chain = Chain(robot_urdf(cfg), cfg["frames"]["base"], cfg["frames"]["tcp"])
    if cfg["arm"].get("position_limits"):
        chain.clamp_limits(cfg["arm"]["position_limits"], 0.02)
    t = scene["table"]
    table_z = t["center"][2] + t["size"][2] / 2
    regs = [scene["target"]["spawn_region"], scene["goal"]["region"]]
    xs = np.linspace(min(r["x"][0] for r in regs), max(r["x"][1] for r in regs), grid)
    ys = np.linspace(min(r["y"][0] for r in regs), max(r["y"][1] for r in regs), grid)
    home = np.array(cfg["arm"]["home"], float)
    approach = float(cfg.get("grasp_motion", {}).get("approach_height_m", 0.10))    # the robot's own pre-grasp
    results = []
    t0 = time.time()
    for rest in rests:
        r = rest_poses(obj)[rest]
        cs = static_rank(candidates(obj, cfg["gripper"], rest, table_z=0.0))[:max_cands]
        period = r.period_deg or 360.0
        yaws = [math.radians(period * k / n_yaws) for k in range(n_yaws)]
        for c in cs:
            ok, n, margins = 0, 0, []
            ok_map = np.zeros((len(ys), len(xs)), int)
            q_prev = home
            d = c.T_obj_tcp[:3, 2]
            for iy, y in enumerate(ys):
                for ix, x in enumerate(xs):
                    for yaw in yaws:
                        n += 1
                        T_obj = np.eye(4)
                        T_obj[:3, :3] = rot_z(yaw) @ r.R
                        T_obj[:3, 3] = [x, y, table_z + r.height_m]
                        T = T_obj @ c.T_obj_tcp
                        good = True
                        for back in (approach, 0.0):      # pre-grasp first (it seeds the grasp's IK)
                            p = T[:3, 3] - back * (T_obj[:3, :3] @ d)
                            q, pe, ae = chain.ik(p, target_R=T[:3, :3], q0=q_prev, seeds=2)
                            if q is None or pe > 0.002 or ae > 1.0:
                                good = False
                                break
                            jp = joint_points(chain, q)
                            far = jp[np.linalg.norm(jp - p, axis=1) > 0.12]        # the arm, not the hand (the hand's
                            if len(far) and far[:, 2].min() < table_z + 0.02:      # table clearance: hand_boxes)
                                good = False
                                break
                            q_prev = q
                        if good:
                            ok += 1
                            ok_map[iy, ix] += 1
                            span = chain.upper - chain.lower
                            margins.append(float(np.min(np.minimum(q - chain.lower, chain.upper - q) / span)))
                        else:
                            q_prev = home
            c.scores = {"reach": ok / n, "placements": n, "joint_margin": float(np.mean(margins)) if margins else 0.0,
                        "map": (ok_map / len(yaws)).round(2).tolist()}
            results.append(c)
    # best first: reach, then little gravity twist, then straight-down, then joint-limit margin, then deeper
    results.sort(key=lambda k: (-round(k.scores["reach"], 2), round(k.twist_nm, 3), k.tilt_deg,
                                -round(k.scores["joint_margin"], 2), -k.depth_m))
    log(f"{robot}/{obj_name}: {len(results)} grasps x {len(xs) * len(ys) * n_yaws} placements in "
        f"{time.time() - t0:.0f} s")
    best = [c for rest in rests for c in [k for k in results if k.rest == rest][:top]]     # the best few per face
    return best, {"xs": xs.round(3).tolist(), "ys": ys.round(3).tolist(), "yaws_deg":
                           [round(math.degrees(v), 1) for v in yaws]}


def to_json(robot, obj_name, best, grid):
    return {"robot": robot, "object": obj_name, "grid": grid,
            "grasps": [{"name": c.name, "rest": c.rest, "width_m": c.width_m, "tilt_deg": c.tilt_deg,
                        "toward": c.toward, "depth_m": c.depth_m, "offset_m": c.offset_m, "twist_nm": c.twist_nm,
                        "margin_m": c.margin_m, "T_obj_tcp": np.round(c.T_obj_tcp, 5).tolist(), **c.scores}
                       for c in best]}


def write_report(studies):
    """One HTML page: per robot / object the best grasps with their reach maps (green = reachable)."""
    OUT.mkdir(parents=True, exist_ok=True)
    blocks = []
    for s in studies:
        rows = []
        for i, g in enumerate(s["grasps"]):
            cells = "".join(
                "<tr>" + "".join(f'<td style="background:rgba(31,157,85,{v:.2f})"></td>' for v in row) + "</tr>"
                for row in reversed(g["map"]))
            rows.append(f"""<tr><td>{i + 1}</td><td>{g['name']}</td><td>{g['rest']}</td>
              <td><b>{g['reach'] * 100:.0f}%</b></td><td>{g['twist_nm']:.3f}</td><td>{g['margin_m'] * 1000:.0f}</td>
              <td>{g['joint_margin'] * 100:.0f}%</td><td><table class="map">{cells}</table></td></tr>""")
        blocks.append(f"""<section><h2>{s['robot']} &middot; {s['object'].replace('_', ' ')}</h2>
          <table class="t"><tr><th>#</th><th>grasp</th><th>face</th><th>reach</th><th>twist [N&middot;m]</th>
          <th>opening margin [mm]</th><th>joint margin</th><th>where (top view)</th></tr>{''.join(rows)}</table></section>""")
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Grasp study</title>
<style>
:root {{ --bg:#f4f5f7; --panel:#fff; --ink:#1d2330; --muted:#6b7385; --line:#dfe3ea; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#12151b; --panel:#1b2029; --ink:#e7ebf3; --muted:#9aa3b5; --line:#2c3340; }} }}
body {{ margin:0; padding:16px; background:var(--bg); color:var(--ink); font:14px/1.45 system-ui, sans-serif; }}
section {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:10px 14px; margin-bottom:14px; overflow-x:auto; }}
h1 {{ font-size:18px; margin:0 0 4px; }} h2 {{ font-size:15px; margin:4px 0 8px; }}
p {{ color:var(--muted); max-width:900px; }}
.t {{ border-collapse:collapse; width:100%; font-size:13px; }}
.t th, .t td {{ text-align:left; padding:4px 8px; border-bottom:1px solid var(--line); vertical-align:middle; }}
.t th {{ color:var(--muted); font-weight:600; }}
.map {{ border-collapse:collapse; }} .map td {{ width:9px; height:9px; padding:0; border:1px solid var(--line); }}
</style></head><body>
<h1>Grasp study</h1>
<p>For each object and robot: the best candidate grasps (closing side, approach tilt, depth, off-center) by how much
of the cell's work area the arm reaches with them (IK for grasp + pre-grasp, every joint above the table), then by how
little gravity twists the object in the fingers. The small map is the work area from above (robot at the bottom):
green = reachable at that spot (darker = more of the object yaws). tools/grasp_study.py,
docs/learning/20_grasp_study.md. Generated {time.strftime('%Y-%m-%d %H:%M')}.</p>
{''.join(blocks)}</body></html>"""
    (OUT / "index.html").write_text(html, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot")
    ap.add_argument("--object")
    ap.add_argument("--rest", default="upright", help="a face, or 'all'")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--grid", type=int, default=4)
    ap.add_argument("--yaws", type=int, default=3)
    a = ap.parse_args()
    jobs = [(r, o) for r, (_, objs) in CELLS.items() for o in objs] if a.all else [(a.robot, a.object)]
    studies = []
    OUT.mkdir(parents=True, exist_ok=True)
    for robot, name in jobs:
        obj = load_yaml("config/objects.yaml")[name]
        rests = tuple(rest_poses(obj)) if a.rest == "all" else (a.rest,)
        best, grid = study(robot, name, rests, a.grid, a.yaws)
        js = to_json(robot, name, best, grid)
        (OUT / f"{robot}_{name}.json").write_text(json.dumps(js, indent=1), encoding="utf-8")
        studies.append(js)
        for g in best:
            print(f"   {g.rest:12s} {g.name:58s} reach {g.scores['reach'] * 100:3.0f}%  twist {g.twist_nm:.3f} N m  "
                  f"joint margin {g.scores['joint_margin'] * 100:.0f}%")
    # keep the other studies already on disk in the report
    have = {(s["robot"], s["object"]) for s in studies}
    for f in sorted(OUT.glob("*.json")):
        s = json.loads(f.read_text(encoding="utf-8"))
        if (s["robot"], s["object"]) not in have:
            studies.append(s)
    write_report(sorted(studies, key=lambda s: (s["robot"], s["object"])))
    print(f"report: {OUT / 'index.html'}")


if __name__ == "__main__":
    main()
