"""Dump the collision setup of chosen links in a robot USD: approximation, offsets, decomposition parameters."""
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
import sys  # noqa: E402
from pxr import Usd, UsdPhysics  # noqa: E402
out = open(sys.argv[2], "w")
st = Usd.Stage.Open(sys.argv[1])
for p in Usd.PrimRange.Stage(st, Usd.TraverseInstanceProxies()):
    path = str(p.GetPath())
    if not any(k in path for k in ("gripper_link", "moving_jaw")) or not (p.HasAPI(UsdPhysics.CollisionAPI) or "collision" in path.lower()):
        continue
    attrs = {a.GetName(): a.Get() for a in p.GetAttributes()
             if any(s in a.GetName() for s in ("physics:approximation", "physxCollision", "physxConvex", "physxSDF"))}
    print(path, p.GetTypeName(), [s for s in p.GetAppliedSchemas()], attrs, file=out)
out.close()
app.close()
