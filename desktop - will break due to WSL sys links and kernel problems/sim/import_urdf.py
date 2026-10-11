"""Turn a robot profile's URDF into the USD the sim uses (once per robot; output not in git, regenerable).

    C:\\isaac-sim\\python.bat sim\\import_urdf.py --robot so101        # -> config/robot_so101.yaml sim.usd

What this teaches: Isaac Sim simulates USD, robots ship as URDF. Isaac's URDF importer (isaacsim.asset.importer.urdf)
converts links -> rigid bodies with their visual + collision meshes (collision as convex hulls), joints -> USD physics
joints with position DRIVES (what our ROS graph commands), and puts the articulation root on the fixed base.
The drive gains set here are placeholders: sim/scene.py sets the robot profile's sim.drive values when it loads it.
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--robot", default="so101")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import omni.kit.commands  # noqa: E402
import omni.usd  # noqa: E402
from common import REPO, log, shutdown  # noqa: E402
from pxr import Usd, UsdPhysics  # noqa: E402

from pnp_core.catalog import load_robot  # noqa: E402


def main() -> int:
    cfg = load_robot(args.robot)
    urdf = REPO / cfg["description"]["urdf"]
    out = REPO / cfg["sim"]["usd"]
    if not urdf.exists():
        log(f"[import] {urdf} missing: run robots/{args.robot}/fetch_description.sh first")
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    _, ic = omni.kit.commands.execute("URDFCreateImportConfig")
    ic.merge_fixed_joints = False         # keep gripper_frame_link etc. (named frames are useful)
    ic.fix_base = True                    # the arm is bolted to the table
    ic.import_inertia_tensor = True
    ic.self_collision = False
    ic.make_default_prim = True
    ic.create_physics_scene = False       # our scene has its own
    ic.collision_from_visuals = False     # the URDF has collision meshes (convex hulls of them)
    # convex DECOMPOSITION, not one hull per link: the SO-101's fixed jaw is L-shaped, its single hull filled the gap
    # the moving jaw closes into (the jaw stopped at 34 mm on a 16 mm brick)
    ic.convex_decomp = True
    ic.distance_scale = 1.0
    ic.set_default_drive_type(1)          # position drives
    ic.set_default_drive_strength(float(cfg["sim"]["drive"]["stiffness"]))
    ic.set_default_position_drive_damping(float(cfg["sim"]["drive"]["damping"]))
    ok, prim = omni.kit.commands.execute("URDFParseAndImportFile", urdf_path=str(urdf), import_config=ic,
                                         dest_path=str(out).replace("\\", "/"), get_articulation_root=True)
    log(f"[import] {urdf.name} -> {out} (ok={ok}, articulation root {prim})")
    stage = Usd.Stage.Open(str(out))
    for p in stage.Traverse():
        kind = ("joint" if p.IsA(UsdPhysics.Joint) else "body" if p.HasAPI(UsdPhysics.RigidBodyAPI) else "")
        if kind or p.GetPath().pathElementCount <= 2:
            log(f"[import]   {p.GetPath()}  {p.GetTypeName()} {kind}"
                + ("  ARTICULATION ROOT" if p.HasAPI(UsdPhysics.ArticulationRootAPI) else ""))
    return 0


if __name__ == "__main__":
    shutdown(app, main())
