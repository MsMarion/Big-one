"""Put the simulated cameras back at their nominal poses (+ the active reality profile's errors) and check them."""
import sys
import rclpy
sys.path.insert(0, "/repo/tools")
from run_benchmark import Bench  # noqa: E402
import yaml  # noqa: E402

rclpy.init()
b = Bench(name="pnp_reset_cameras")
b.sim.wait_ready(timeout=5.0)
active = yaml.safe_load(open("/repo/config/sim_reality.yaml")).get("active", "ideal")
print("set_reality:", b.sim.call("set_reality", profile=active))
print("camera_truth:", b.sim.call("camera_truth"))
rclpy.try_shutdown()
