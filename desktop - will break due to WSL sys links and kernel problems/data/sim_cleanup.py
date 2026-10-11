import sys
import rclpy
sys.path.insert(0, "/repo/tools")
from run_benchmark import Bench
rclpy.init()
b = Bench(name="pnp_sim_cleanup")
b.sim.wait_ready(timeout=5.0)
print(b.sim.call("calibration_targets", table_board=False, hand_board=False))
print(b.sim.call("reset_target"))
rclpy.try_shutdown()
