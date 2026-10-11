"""Sim-ready USD per object: assets/<id>/sim.usda references the shared geometry and adds physics
(mass, convex-hull colliders, friction, restitution) and semantic labels. See objsearch/blender/usd_layer.py.

Rebuilt automatically after every model build and whenever the object's physics change (datasheet research),
or by hand:  python -m objsearch sim <ids>
"""
import json
import os
import re

from . import config, db, physical, track


def enabled():
    return os.environ.get("OBJSEARCH_SIM", "1") not in ("0", "false", "no", "")


def path(oid):
    return os.path.join(config.data_dir(), "assets", oid, "sim.usda")


def _geometry(conn, oid):
    r = conn.execute("SELECT path, source_url, license FROM assets WHERE object_id=? AND kind='usd' "
                     "ORDER BY id DESC LIMIT 1", (oid,)).fetchone()
    return (os.path.join(config.data_dir(), r[0]), r[1], r[2]) if r else (None, None, None)


def prim_name(oid):
    name = re.sub(r"[^A-Za-z0-9_]", "_", oid)
    return name if not name[0].isdigit() else "_" + name


def build(conn, oid):
    """(Re)write the object's sim.usda. -> {"path", "mass_g", ...} or None when there's no model yet."""
    from .blender import interface
    o = db.get_object(conn, oid)
    geo, source_url, license = _geometry(conn, oid)
    if not geo or not os.path.exists(geo):
        return None
    phys = physical.physics_for(o)
    out = path(oid)
    with track.step("sim_usd", object_id=oid, inputs={"geometry": track.rel(geo)}) as s:
        cfg = {
            "out": out, "prim": prim_name(oid), "label": oid, "category": o.get("category"),
            "asset_rel": os.path.relpath(geo, os.path.dirname(out)).replace("\\", "/"),
            "mass_kg": phys["mass_g"] / 1000, "static_friction": phys["static_friction"],
            "dynamic_friction": phys["dynamic_friction"], "restitution": phys["restitution"],
            "custom": {"id": oid, "name": o["name"], "category": o.get("category") or "",
                       "source_url": source_url or "", "license": license or "unknown",
                       "dims_mm": json.dumps(o.get("dims_mm")), "dims_source": o.get("dims_source") or "catalog",
                       "mass_g": phys["mass_g"], "mass_source": phys["mass_source"],
                       "materials": json.dumps(phys["materials"]), "physics_source": phys["source"]},
        }
        os.makedirs(os.path.dirname(out), exist_ok=True)
        cfg_path = os.path.join(os.path.dirname(out), "sim_config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=1)
        res = interface.run_module("objsearch.blender.usd_layer", [cfg_path],
                                   os.path.join(os.path.dirname(out), "sim_usd.log"))
        res.update(path=track.rel(out), mass_g=phys["mass_g"], mass_source=phys["mass_source"])
        s.set(**res)
    return res


def safe_build(conn, oid):
    """build() for automatic hooks: a failure is logged, never fatal to the model that triggered it."""
    try:
        return build(conn, oid)
    except Exception as e:
        import logging
        logging.getLogger("objsearch.sim").warning(f"{oid}: sim.usda not built: {e}")
        return None
