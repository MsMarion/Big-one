import sys
import numpy as np
sys.path.insert(0, "/repo")
from pnp_core import robot_description as rdesc
from pnp_core.catalog import load_robot, repo_root
from pnp_core.servo_error import GravityModel
cfg = load_robot("so101"); J = cfg["arm"]["joints"]
g = GravityModel(rdesc.urdf(cfg, repo_root(), "mock"), "base_link", J)
for name, q in (("home", cfg["arm"]["home"]), ("zero", [0, 0, 0, 0, 0])):
    qd = dict(zip(J, q))
    ch = g.joint_chains["shoulder_lift"]
    T = ch.fk([qd[n] for n in ch.names])
    print(name, "shoulder_lift axis", np.round(T[:3, :3] @ ch.joints[-1].axis, 3), "at", np.round(T[:3, 3], 3))
    tot, mc = 0, np.zeros(3)
    for lch, m, com in g.links:
        Tl = lch.fk([qd.get(n, 0) for n in lch.names]); c = Tl[:3, :3] @ com + Tl[:3, 3]
        if "shoulder_lift" in lch.names:
            tot += m; mc += m * c
            print(f"   {lch.joints[-1].child:28s} m {m:.3f} com {np.round(c, 3)}")
    print("   carried mass", round(tot, 3), "COM", np.round(mc / tot, 3), "torques", {k: round(v, 3) for k, v in g.torques(qd).items()})
