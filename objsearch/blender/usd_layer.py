"""Runs INSIDE .venv-bpy (needs pxr):  python -m objsearch.blender.usd_layer CONFIG.json

Writes a small sim-ready USD layer for one object that REFERENCES the shared geometry (model_cache/<key>/asset.usd)
and adds, as overrides, everything a simulator / synthetic-data pipeline needs:
  physics    RigidBodyAPI + MassAPI (kg) on the root; CollisionAPI + MeshCollisionAPI (convex hull) on every
             mesh; a physics material (static/dynamic friction, restitution) bound with the 'physics' purpose
  semantics  UsdSemantics LabelsAPI ('class' = object id, 'category') plus the legacy Isaac Sim
             SemanticsAPI attributes (semantic:Semantics:params:*), so Replicator segmentation picks it up
  metadata   customData objsearch:{id, name, source, license, mass source, dims source, ...}
Geometry is never copied, so rebuilding the layer (new datasheet mass, new label) takes well under a second.
Prints one line:  OBJSEARCH_RESULT {...json...}
"""
import json
import os
import sys

from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdSemantics, UsdShade


MAX_HULLS = 32  # more meshes than this -> a single box collider (simulators want few, simple shapes)


def main(cfg_path):
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    out = cfg["out"]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):
        os.remove(out)
    stage = Usd.Stage.CreateNew(out)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/" + cfg["prim"])
    prim = root.GetPrim()
    stage.SetDefaultPrim(prim)
    prim.GetReferences().AddReference(cfg["asset_rel"])  # the geometry's default prim (/root)
    Usd.ModelAPI(prim).SetKind("component")

    # physics: one rigid body with the researched / estimated mass
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(cfg["mass_kg"]))
    mat = UsdShade.Material.Define(stage, f"/{cfg['prim']}/PhysicsMaterial")
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(cfg["static_friction"]))
    pm.CreateDynamicFrictionAttr(float(cfg["dynamic_friction"]))
    pm.CreateRestitutionAttr(float(cfg["restitution"]))
    meshes = [p for p in Usd.PrimRange(prim) if p.IsA(UsdGeom.Mesh)]
    if len(meshes) <= MAX_HULLS:  # few parts: one convex hull each
        for p in meshes:
            UsdPhysics.CollisionAPI.Apply(p)
            UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
            UsdShade.MaterialBindingAPI.Apply(p).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")
    else:  # many small meshes (e.g. KiCad face meshes): one invisible box collider around the whole object
        rng = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeLocalBound(prim).ComputeAlignedRange()
        lo, hi = rng.GetMin(), rng.GetMax()
        cube = UsdGeom.Cube.Define(stage, f"/{cfg['prim']}/collision_box")
        cube.CreateSizeAttr(1.0)
        cube.CreatePurposeAttr(UsdGeom.Tokens.guide)  # not rendered
        xf = UsdGeom.Xformable(cube)
        xf.AddTranslateOp().Set(tuple((lo[i] + hi[i]) / 2 for i in range(3)))
        xf.AddScaleOp().Set(tuple(max(hi[i] - lo[i], 1e-6) for i in range(3)))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")

    # semantics: current UsdSemantics schema + the attributes older Isaac Sim / Replicator versions read
    UsdSemantics.LabelsAPI.Apply(prim, "class").CreateLabelsAttr([cfg["label"]])
    if cfg.get("category"):
        UsdSemantics.LabelsAPI.Apply(prim, "category").CreateLabelsAttr([cfg["category"]])
    prim.AddAppliedSchema("SemanticsAPI:Semantics")
    prim.CreateAttribute("semantic:Semantics:params:semanticType", Sdf.ValueTypeNames.String).Set("class")
    prim.CreateAttribute("semantic:Semantics:params:semanticData", Sdf.ValueTypeNames.String).Set(cfg["label"])

    prim.SetCustomDataByKey("objsearch", cfg.get("custom") or {})
    stage.GetRootLayer().documentation = f"objsearch sim layer for {cfg['label']}: physics + semantics over shared geometry"
    stage.GetRootLayer().Save()

    # read back through composition to prove the overrides land on the referenced meshes
    st = Usd.Stage.Open(out)
    p0 = st.GetDefaultPrim()
    coll = [p for p in Usd.PrimRange(p0) if p.HasAPI(UsdPhysics.CollisionAPI)]
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(p0).ComputeAlignedRange()
    size = [round((bbox.GetMax()[i] - bbox.GetMin()[i]) * 1000, 2) for i in range(3)]
    print("OBJSEARCH_RESULT " + json.dumps({"meshes": len(meshes), "colliders": len(coll), "size_mm": size,
                                           "mass_kg": UsdPhysics.MassAPI(p0).GetMassAttr().Get()}), flush=True)


def localize(src, out_dir):
    """Copy a layer and everything it pulls in (references, textures) into out_dir with paths rewritten, so the
    folder is self-contained (for handing objects to another machine / pipeline)."""
    from pxr import UsdUtils
    os.makedirs(out_dir, exist_ok=True)
    ok = UsdUtils.LocalizeAsset(src, out_dir)
    files = sorted(os.path.relpath(os.path.join(d, f), out_dir).replace("\\", "/")
                   for d, _, fs in os.walk(out_dir) for f in fs)
    st = Usd.Stage.Open(os.path.join(out_dir, os.path.basename(src)))
    print("OBJSEARCH_RESULT " + json.dumps({"ok": bool(ok) and st is not None, "files": files}), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--localize":
        localize(sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1])
