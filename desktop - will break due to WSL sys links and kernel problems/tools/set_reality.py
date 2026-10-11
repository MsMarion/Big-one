"""Switch how realistic the simulated sensors are, live (config/sim_reality.yaml, docs/learning/15_sim_realism.md).

    python3 /repo/tools/set_reality.py realistic        # ideal | realistic | stress
    python3 /repo/tools/set_reality.py --truth          # where the sim cameras REALLY are vs where TF says

Two places change together: the sensor model's noise (/pnp_sensor_model profile) and the simulated cameras'
placement errors / drift (sim command set_reality: Isaac moves the cameras, ROS keeps believing its TF: the nominal
mounts, or the last calibration in config/calibration/sim_*.yaml, which then no longer fits -> tools/calibrate.py).
"""

import argparse
import sys

import numpy as np
import rclpy
import yaml

sys.path.insert(0, "/repo/tools")
from run_benchmark import Bench  # noqa: E402

from pnp_core.geometry import make_transform, quat_to_matrix  # noqa: E402
from rclpy.time import Time  # noqa: E402

from pnp_core.catalog import load_robot as _load_robot  # noqa: E402
BASE = _load_robot()["frames"]["base"]     # the active cell's robot base frame


def set_param(b, node, name, value):
    from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
    from rcl_interfaces.srv import SetParameters
    cli = b.create_client(SetParameters, f"/{node}/set_parameters")
    if not cli.wait_for_service(timeout_sec=5.0):
        return False
    p = Parameter(name=name, value=ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=value))
    res = b.wait_future(cli.call_async(SetParameters.Request(parameters=[p])), 5.0)
    return bool(res and res.results and res.results[0].successful)


def truth(b):
    """True camera poses (sim) vs believed (TF): the calibration error a calibration should find."""
    t = b.sim.call("camera_truth")
    frames = {"camera": "camera_color_optical_frame", "wrist_camera": "wrist_camera_color_optical_frame"}
    print(f"profile {t['profile']}:")
    for key, T_true in t["cameras"].items():
        T_true = np.array(T_true)
        frame = frames[key]
        for _ in range(50):                      # wait for the TF tree (a fresh process has to receive it first)
            if b.tf.can_transform(BASE, frame, Time()):
                break
            b.spin_for(0.1)
        tf = b.tf.lookup_transform(BASE, frame, Time()).transform
        T_bel = make_transform(quat_to_matrix([tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w]),
                               [tf.translation.x, tf.translation.y, tf.translation.z])
        d = np.linalg.norm(T_true[:3, 3] - T_bel[:3, 3]) * 1000
        dR = T_bel[:3, :3].T @ T_true[:3, :3]
        ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
        hint = f"  -> tools/calibrate.py {'fixed' if key == 'camera' else 'wrist'}" if d > 2.0 or ang > 0.2 else ""
        print(f"  {key:13s} true vs believed: {d:5.1f} mm, {ang:4.2f} deg{hint}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile", nargs="?",
                    choices=list(yaml.safe_load(open("/repo/config/sim_reality.yaml"))["profiles"]))
    ap.add_argument("--truth", action="store_true")
    a = ap.parse_args()
    rclpy.init()
    b = Bench(name="pnp_set_reality")
    b.spin_for(1.0)
    if a.profile:
        ok = set_param(b, "pnp_sensor_model", "profile", a.profile)
        r = b.sim.call("set_reality", profile=a.profile)
        print(f"sensor noise -> {a.profile}: {'ok' if ok else 'FAILED (sensor model running?)'}; "
              f"camera placement errors: {r.get('errors')}, drift: {r.get('drift')}")
        # each profile = the cameras somewhere else = its own calibration (config/calibration/sim_<profile>_*.yaml)
        ok = set_param(b, "pnp_camera_tf", "calibration_set", f"sim_{a.profile}")
        print(f"camera calibration set -> sim_{a.profile}: {'ok' if ok else 'FAILED (camera_tf_node running?)'}")
    if a.truth or a.profile:
        b.spin_for(0.5)
        truth(b)
    rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
