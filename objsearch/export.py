"""Hand objects to the perception pipeline (DINOv3 training, FoundationPose, Isaac Sim / Replicator).

    python -m objsearch export <out_dir> [--ids a b ..] [--owned] [--real-only]

out_dir/
  classes.txt            one object id per line; line index = class id (same order as models_info.json)
  models_info.json       BOP-style per class: diameter + bbox (mm), mass, source, license, model quality
  objects/<id>/
    mesh/mesh.obj (+.mtl, textures)  FoundationPose mesh: metres, Z-up, base on z=0 - SAME frame as the USD,
                                      so a pose estimated on this mesh places the USD asset directly
    usd/sim.usda (+ geometry, textures)  self-contained sim-ready USD: physics + semantic label = object id
    model.glb  thumb.png  info.json
  README.txt
"""
import json
import os
import shutil

import numpy as np

from . import config, db, physical, sim, track

README = """objsearch export
- Units: metres, Z-up, object base on z=0, centred in x/y. mesh/mesh.obj and usd/sim.usda share this frame.
- classes.txt: class id = line index (0-based); semantic label in the USD = object id.
- models_info.json: BOP-style (mm): diameter, min_x/y/z, size_x/y/z; plus mass_g, quality, license, source.
- quality: real (CAD / library model) | exact (LDraw LEGO part) | placeholder (box at the item's size: fine for
  detection/segmentation training, poor for pose estimation - FoundationPose needs real geometry).
- Licenses differ per object; check info.json before redistributing.
"""


def _quality(conn, oid):
    c = db.row(conn.execute("SELECT source_url, license, scores FROM candidates WHERE object_id=? AND chosen=1 "
                            "ORDER BY id DESC LIMIT 1", (oid,)).fetchone())
    flags = ((c or {}).get("scores") or {}).get("flags") or []
    # license 'generated' survives rebuilds (flags added after a build don't)
    if (c or {}).get("license") == "generated" or any(str(f).startswith("placeholder model") for f in flags):
        return "placeholder", flags
    if c and "ldraw.org" in (c.get("source_url") or ""):
        return "exact", flags
    return "real", flags


def _diameter(v, cap=4000):
    """Largest vertex-to-vertex distance (BOP 'diameter'), exact for up to `cap` points; above that, on the points
    farthest from the centre along many directions (the extremes always lie among them)."""
    if len(v) < 2:
        return 0.0
    if len(v) > cap:
        dirs = np.random.default_rng(0).normal(size=(256, 3))
        proj = (v - v.mean(axis=0)) @ dirs.T
        v = v[np.unique(np.concatenate([proj.argmax(axis=0), proj.argmin(axis=0)]))]
    best = 0.0
    for i in range(0, len(v), 512):  # chunked so 4000 points never need a 4000x4000x3 array
        best = max(best, float(np.sqrt(((v[i:i + 512, None, :] - v[None, :, :]) ** 2).sum(-1)).max()))
    return best


def _mesh_info(obj_path):
    import trimesh
    m = trimesh.load(obj_path, force="mesh", process=False)
    v = np.unique(np.round(np.asarray(m.vertices) * 1000.0, 3), axis=0)  # mm
    d = _diameter(v)
    lo, hi = v.min(axis=0), v.max(axis=0)
    return {"diameter": round(float(d), 3), "min_x": round(float(lo[0]), 3), "min_y": round(float(lo[1]), 3),
            "min_z": round(float(lo[2]), 3), "size_x": round(float(hi[0] - lo[0]), 3),
            "size_y": round(float(hi[1] - lo[1]), 3), "size_z": round(float(hi[2] - lo[2]), 3)}


def _mesh_dir(conn, oid):
    rel = conn.execute("SELECT path FROM assets WHERE object_id=? AND kind='obj' ORDER BY id DESC LIMIT 1",
                       (oid,)).fetchone()[0]
    return os.path.dirname(os.path.join(config.data_dir(), rel))


def _z_up(mesh_dir):
    p = os.path.join(os.path.dirname(mesh_dir), "result.json")
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f).get("obj_up") == "Z"
    except (OSError, ValueError):
        return False


def export(conn, out_dir, ids=None, owned=False, real_only=False):
    from .blender import interface
    q = ("SELECT DISTINCT o.id FROM objects o JOIN assets a ON a.object_id=o.id AND a.kind='obj' "
         "WHERE o.status != 'rejected'" + (" AND o.owned=1" if owned else "") + " ORDER BY o.id")
    oids = [r[0] for r in conn.execute(q)]
    if ids:
        oids = [i for i in oids if i in set(ids)]
    out_dir = os.path.abspath(out_dir)
    os.makedirs(os.path.join(out_dir, "objects"), exist_ok=True)
    classes, info_all, skipped = [], {}, {}
    with track.run(conn, "export", provider="pipeline", notes=f"export {len(oids)} -> {out_dir}"[:200]):
        for oid in oids:
            with track.step("export_object", object_id=oid) as s:
                o = db.get_object(conn, oid)
                quality, flags = _quality(conn, oid)
                if real_only and quality == "placeholder":
                    skipped[oid] = "placeholder model"
                    s.skip("placeholder model (--real-only)")
                src_mesh = _mesh_dir(conn, oid)
                if not _z_up(src_mesh):  # built before OBJs shared the USD frame: rebuild once from its source
                    from . import models
                    models.rebuild(conn, oid)
                    src_mesh = _mesh_dir(conn, oid)
                    if not _z_up(src_mesh):
                        raise RuntimeError(f"{oid}: rebuilt mesh still lacks the Z-up OBJ marker")
                    s.set(rebuilt="old Y-up OBJ")
                dst = os.path.join(out_dir, "objects", oid)
                shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(src_mesh, os.path.join(dst, "mesh"))
                cache = os.path.dirname(src_mesh)
                for f in ("model.glb", "thumb.png"):
                    if os.path.exists(os.path.join(cache, f)):
                        shutil.copyfile(os.path.join(cache, f), os.path.join(dst, f))
                if not os.path.exists(sim.path(oid)):
                    sim.build(conn, oid)
                loc = interface.run_module("objsearch.blender.usd_layer",
                                           ["--localize", sim.path(oid), os.path.join(dst, "usd")],
                                           os.path.join(dst, "usd_localize.log"))
                os.remove(os.path.join(dst, "usd_localize.log"))
                phys = physical.physics_for(o)
                src = db.row(conn.execute("SELECT source_url, license FROM assets WHERE object_id=? AND kind='usd' "
                                          "ORDER BY id DESC LIMIT 1", (oid,)).fetchone()) or {}
                info = {"id": oid, "class_id": len(classes), "name": o["name"], "category": o.get("category"),
                        **_mesh_info(os.path.join(dst, "mesh", "mesh.obj")), "quality": quality, "flags": flags,
                        "dims_mm_catalog": o.get("dims_mm"), "dims_source": o.get("dims_source") or "catalog",
                        "mass_g": phys["mass_g"], "mass_source": phys["mass_source"],
                        "static_friction": phys["static_friction"], "materials": phys["materials"],
                        "license": src.get("license") or "unknown", "source_url": src.get("source_url"),
                        "usd": "usd/sim.usda", "mesh": "mesh/mesh.obj", "usd_files": len(loc["files"])}
                with open(os.path.join(dst, "info.json"), "w", encoding="utf-8") as f:
                    json.dump(info, f, indent=1)
                info_all[str(len(classes))] = info
                classes.append(oid)
                s.set(quality=quality, diameter_mm=info["diameter"])
    with open(os.path.join(out_dir, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(classes) + "\n")
    with open(os.path.join(out_dir, "models_info.json"), "w", encoding="utf-8") as f:
        json.dump(info_all, f, indent=1)
    with open(os.path.join(out_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write(README)
    return {"out_dir": out_dir, "classes": classes, "skipped": skipped,
            "quality": {q: sum(1 for i in info_all.values() if i["quality"] == q) for q in ("real", "exact", "placeholder")}}
