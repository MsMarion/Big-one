"""Print the structure of a USD asset: prims, types, physics APIs, joints, drives, bounding box.

What this teaches: a robot/prop USD is a tree of prims. Physics behaviour comes from *API schemas*
applied to prims (RigidBodyAPI, CollisionAPI, MassAPI, ArticulationRootAPI, DriveAPI...).
Before scripting against an asset, look at what it actually contains.

Run:  .\\run_isaac.ps1 sim\\inspect_usd.py Isaac/Robots/FrankaRobotics/FrankaFR3/fr3.usd [--depth 4]
"""

import argparse

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import omni.usd
from common import log, shutdown
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from pxr import Usd, UsdGeom, UsdPhysics

parser = argparse.ArgumentParser()
parser.add_argument("asset", help="path relative to the Isaac assets root, or a full URL")
parser.add_argument("--depth", type=int, default=6)
parser.add_argument("--all", action="store_true", help="list every prim (default: physics-relevant only)")
args = parser.parse_args()


def main() -> int:
    url = args.asset if "://" in args.asset else f"{get_assets_root_path()}/{args.asset}"
    add_reference_to_stage(usd_path=url, prim_path="/World/asset")
    app.update()
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath("/World/asset")
    log(f"[usd] {url}")
    base_depth = len(root.GetPath().pathString.split("/"))

    for prim in Usd.PrimRange(root):
        depth = len(prim.GetPath().pathString.split("/")) - base_depth
        if depth > args.depth:
            continue
        apis = [a for a in prim.GetAppliedSchemas() if "Physics" in a or "Physx" in a or "Mass" in a]
        is_joint = prim.IsA(UsdPhysics.Joint)
        if not (args.all or apis or is_joint or depth <= 1):
            continue
        line = f"{'  ' * depth}{prim.GetName()} <{prim.GetTypeName()}>"
        if apis:
            line += f" apis={apis}"
        if is_joint:
            lo, hi = prim.GetAttribute("physics:lowerLimit"), prim.GetAttribute("physics:upperLimit")
            if lo and hi and lo.Get() is not None:     # revolute: DEGREES (USD convention), prismatic: meters
                line += f" limits=[{lo.Get():.4g}, {hi.Get():.4g}]"
            for kind in ("angular", "linear"):
                drive = UsdPhysics.DriveAPI.Get(prim, kind)
                if drive and drive.GetStiffnessAttr().HasAuthoredValue():
                    line += (f" drive[{kind}] k={drive.GetStiffnessAttr().Get()} d={drive.GetDampingAttr().Get()}"
                             f" maxF={drive.GetMaxForceAttr().Get()}")
        log(line)

    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(root)
    rng = bbox.ComputeAlignedRange()
    log(f"[usd] world AABB min={tuple(round(v, 4) for v in rng.GetMin())} max={tuple(round(v, 4) for v in rng.GetMax())}")
    log(f"[usd] metersPerUnit={UsdGeom.GetStageMetersPerUnit(stage)} upAxis={UsdGeom.GetStageUpAxis(stage)}")
    return 0


if __name__ == "__main__":
    shutdown(app, main())
