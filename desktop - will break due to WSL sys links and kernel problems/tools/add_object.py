"""Add a new object to the catalog from its MESH (+ a name, a few words, optional photos): the first step of
"show the robot a new thing" (docs/learning/24_vision_plan.md, unseen-object pipeline).

    /opt/fp_venv/bin/python /repo/tools/add_object.py usb_flash_drive --mesh ~/meshes/flash.stl \\
        --text "a USB flash drive" --text "a thumb drive" [--photos DIR] [--units mm] [--mass 0.01] [--dry-run]

What it does:
1. Loads the mesh (STL / OBJ / PLY / GLB...), units: metres, or mm when it is obviously mm (or --units).
2. Finds how it RESTS on a table (the most stable pose, trimesh.poses.compute_stable_poses), turns it so its long side
   is x, and puts the origin at the bounding-box center (the convention the bricks' meshes and the grasp code use).
3. Writes models/objects/<name>/mesh.obj (+ mesh.json: the transform applied, dims, the rest poses found).
4. Proposes a catalog entry: a box of those dims, a top-down grasp across the narrow side IF it fits the SO-101's jaw
   (55 mm with 4 mm margin each side), the text prompts and photo folder for the open-vocabulary / DINO stages.
   Appended to config/objects.yaml (or printed with --dry-run). Review it: the grasp depth and force are first guesses.
Next steps (planned): render templates from the mesh, embed them with DINOv2 (tools/build_templates.py), and
FoundationPose uses models/objects/<name>/mesh.obj directly.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh
import yaml

REPO = Path(__file__).resolve().parents[1]
JAW_MAX_M, MARGIN_M = 0.055, 0.004                     # config/robot_so101.yaml gripper.max_width_m, finger margin


def load_mesh(path: Path, units: str | None) -> tuple:
    m = trimesh.load(str(path), force="mesh")
    ext = float(np.max(m.extents))
    if units is None:
        units = "mm" if ext > 2.0 else "m"             # a 2 m object would be odd on a desk; a 2 mm one too
    if units == "mm":
        m.apply_scale(0.001)
    elif units == "cm":
        m.apply_scale(0.01)
    return m, units


def rest_and_align(m: trimesh.Trimesh):
    """-> (the mesh resting z-up, long side along x, origin at its bbox center; 4x4 applied; [(prob, T)])."""
    poses, probs = trimesh.poses.compute_stable_poses(m, n_samples=1, threshold=0.0)
    order = np.argsort(probs)[::-1]
    T_rest = poses[order[0]] if len(poses) else np.eye(4)
    r = m.copy()
    r.apply_transform(T_rest)
    # the long side along x: the minimum-area rectangle of the footprint (2D convex hull)
    hull = trimesh.convex.convex_hull(r).vertices[:, :2]
    best = None
    h2 = hull
    for a in np.radians(np.arange(0.0, 180.0, 0.5)):
        c, s = np.cos(a), np.sin(a)
        p = h2 @ np.array([[c, -s], [s, c]])
        span = p.max(axis=0) - p.min(axis=0)
        area = span[0] * span[1]
        if best is None or area < best[0]:
            best = (area, a, span)
    _, a, span = best
    if span[1] > span[0]:
        a += np.pi / 2
    Rz = trimesh.transformations.rotation_matrix(-a, [0, 0, 1])
    r.apply_transform(Rz)
    T_center = trimesh.transformations.translation_matrix(-r.bounds.mean(axis=0))
    r.apply_transform(T_center)
    T = T_center @ Rz @ T_rest
    return r, T, [(float(probs[i]), poses[i].round(6).tolist()) for i in order[:4]]


def catalog_entry(name: str, dims, texts, photos, mass) -> dict:
    L, W, H = (round(float(v), 4) for v in dims)
    fits = W + 2 * MARGIN_M <= JAW_MAX_M
    e = {
        "shape": "box",
        "dims_m": [L, W, H],
        "mass_kg": round(float(mass), 4),
        "mesh": f"models/objects/{name}/mesh.obj",
        "vision": {"text": list(texts) or [name.replace("_", " ")], **({"photos": photos} if photos else {})},
        "grasp": {"type": "top_down", "depth_below_top_m": round(min(0.006, H / 3), 4), "width_m": W,
                  "force_n": 5.0, "symmetric_yaw": False, "yaw_period_deg": 180, "finger_margin_m": MARGIN_M},
        "place_clearance_m": 0.002,
    }
    if not fits:
        e["grasp"]["note"] = f"narrow side {W * 1000:.0f} mm + margins > the {JAW_MAX_M * 1000:.0f} mm jaw: not graspable"
    return e


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--text", action="append", default=[])
    ap.add_argument("--photos")
    ap.add_argument("--units", choices=["m", "cm", "mm"])
    ap.add_argument("--mass", type=float)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cat_path = REPO / "config" / "objects.yaml"
    catalog = yaml.safe_load(cat_path.read_text())
    if a.name in catalog and not a.dry_run:
        sys.exit(f"{a.name} is already in config/objects.yaml (remove it first to re-add)")
    m, units = load_mesh(Path(a.mesh).expanduser(), a.units)
    r, T, rests = rest_and_align(m)
    dims = r.extents
    mass = a.mass if a.mass is not None else (abs(r.volume) * 1000.0 if r.is_volume else 0.01)   # ~water density
    entry = catalog_entry(a.name, dims, a.text, a.photos, mass)
    print(f"{a.name}: mesh in {units}, {len(r.faces)} faces, rests {dims[0] * 1000:.1f} x {dims[1] * 1000:.1f} x "
          f"{dims[2] * 1000:.1f} mm (most stable pose p = {rests[0][0]:.2f}), watertight {r.is_watertight}")
    print(yaml.safe_dump({a.name: entry}, sort_keys=False, default_flow_style=None).rstrip())
    if a.dry_run:
        print("(dry run: nothing written)")
        return 0
    out = REPO / "models" / "objects" / a.name
    out.mkdir(parents=True, exist_ok=True)
    r.export(str(out / "mesh.obj"))
    (out / "mesh.json").write_text(json.dumps({"source": str(a.mesh), "units": units, "T_applied": T.round(8).tolist(),
                                              "dims_m": [float(v) for v in dims], "rest_poses": rests}, indent=1))
    if a.photos:
        shutil.copytree(Path(a.photos).expanduser(), out / "photos", dirs_exist_ok=True)
    with cat_path.open("a", encoding="utf-8") as f:
        f.write(f"\n# added by tools/add_object.py from {Path(a.mesh).name}: review grasp depth / force\n")
        f.write(yaml.safe_dump({a.name: entry}, sort_keys=False, default_flow_style=None))
    print(f"written: {out.relative_to(REPO)}/mesh.obj, config/objects.yaml entry")
    return 0


if __name__ == "__main__":
    sys.exit(main())
