"""The REAL SO-101: its Feetech servo bus <-> the same joint topics Isaac Sim uses (docs/learning/17_so101.md).

    ros2 launch pnp_bringup bringup.launch.py sim:=false scene:=so101_lego hardware:=real camera:=local
    ros2 service call /so101_bridge/enable std_srvs/srv/SetBool "{data: true}"      # torque on (arm holds / moves)

Architecture: ros2_control runs topic_based_ros2_control exactly as in the sim; only the topics differ
(/so101/hw_joint_states, /so101/hw_joint_command: pnp_core/robot_description.py hardware 'topics'). This node is the
other end: it reads the servos' positions -> joint states, and writes the commanded joint positions -> servo goals.
Everything above (controllers, MoveIt, pick & place, perception) is unchanged between sim and real.

Safety (a real arm can hurt fingers and itself):
  - starts with torque OFF: the arm is limp and its pose is published (move it by hand: RViz must follow = the
    mirror check). Torque only via the ~/enable service (or param auto_enable), and then the arm first HOLDS where
    it is: no jump to whatever was commanded before.
  - every command is clamped to the joint limits of the robot profile and the calibrated servo range;
  - the goal moves toward the command at most max_speed_rad_s (arm) / gripper_speed_rad_s: a sudden command (e.g. a
    trajectory starting elsewhere) becomes a slow move;
  - read errors are counted; after `max_read_errors` in a row the torque is switched off and the node reports it;
  - torque off when the node exits (param torque_off_on_exit).
Calibration: LeRobot's calibration file (lerobot-calibrate; same "zero = middle of range" as the URDF) or the same
fields in YAML (tools/so101_servo_setup.py). pnp_core/feetech.py has the protocol and the tick <-> radian mapping.
"""

import json
import time
from pathlib import Path

import numpy as np

from pnp_core.servo_hold import HoldIntegrator
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from controller_manager_msgs.srv import SwitchController
from std_srvs.srv import SetBool

from pnp_core import robot_description as rdesc
from pnp_core.catalog import load_robot, repo_root
from pnp_core.feetech import Calibration, FeetechBus, MockBus, SerialPort


class So101Bridge(Node):
    def __init__(self):
        super().__init__("so101_bridge")
        self.declare_parameter("robot", "so101")
        self.declare_parameter("mock", False)               # simulated servos (no hardware): tests / dry runs
        cfg = load_robot(self.get_parameter("robot").value)
        real = cfg["real"]
        self.declare_parameter("port", real["port"])
        self.declare_parameter("baud", int(real["baud"]))
        self.declare_parameter("calibration", real["calibration"])
        self.declare_parameter("rate_hz", float(real.get("rate_hz", 50.0)))
        self.declare_parameter("max_speed_rad_s", float(real.get("max_speed_rad_s", 1.0)))
        self.declare_parameter("gripper_speed_rad_s", float(real.get("gripper_speed_rad_s", 2.0)))
        self.declare_parameter("gripper_torque_limit", int(real.get("gripper_torque_limit", 400)))
        self.declare_parameter("auto_enable", False)
        self.declare_parameter("arm_gate_rad", 0.09)
        # the servos' position gain for the ARM joints, written at every torque-on (0 = leave it). LeRobot sets 16
        # (half the factory 32) at every connect "to avoid shakiness" in teleop: the stretched-out arm then sagged ~5 deg
        # at the shoulder and moves ended CONTROL_FAILED. config/robot_so101.yaml real.p_gain
        self.declare_parameter("p_gain", int(real.get("p_gain", 0)))
        # the outer integral loop that makes a loaded servo REACH its goal (pnp_core/servo_hold.py); 0 = off
        self.declare_parameter("hold_ki", float(real.get("hold_ki", 0.0)))
        self.declare_parameter("hold_max_deg", float(real.get("hold_max_deg", 5.0)))       # after torque-on: first command must be within ~5 deg
        self.declare_parameter("torque_off_on_exit", True)
        # 50 ticks = 1 s: a short bus glitch (2026-10-11 on the plate: corrupt bytes, ~0.2 s gap) cut torque at 10 and the arm
        # fell; the servos HOLD their last goal on their own while we can't read them, limp is the worse outcome
        self.declare_parameter("max_read_errors", 50)
        self.cfg = cfg
        self.ids = {n: int(i) for n, i in real["motor_ids"].items()}
        self.names = list(cfg["arm"]["joints"]) + [cfg["gripper"]["driven_joint"]]
        self.gripper = cfg["gripper"]["driven_joint"]
        cal_path = self.get_parameter("calibration").value
        cal_path = cal_path if cal_path.startswith("/") else str(repo_root() / cal_path)
        if self.get_parameter("mock").value and not Path(cal_path).exists():
            example = str(repo_root() / "config/calibration/so101_desk/so101_follower.example.json")
            self.get_logger().warn(f"no calibration {cal_path}: MOCK run uses the example {example}")
            cal_path = example
        lim = cfg["arm"]["position_limits"]
        margin = cfg["arm"].get("limit_margin", 0.02)
        self.lo = {n: lim[n][0] + margin for n in cfg["arm"]["joints"]}
        self.hi = {n: lim[n][1] - margin for n in cfg["arm"]["joints"]}
        g_lim = cfg["gripper"].get("joint_limits", [-0.17453, 1.74533])
        self.lo[self.gripper], self.hi[self.gripper] = g_lim
        self.cal = Calibration.load(cal_path, gripper=self.gripper, gripper_limits=tuple(g_lim))
        missing = [n for n in self.names if n not in self.cal.motors]
        if missing:
            raise RuntimeError(f"calibration {cal_path} has no entry for {missing}")
        if self.get_parameter("mock").value:
            self.bus = MockBus({self.ids[n]: self.cal.to_ticks(n, 0.0) for n in self.names})
            self.get_logger().warn("MOCK servo bus (no hardware)")
        else:
            port = SerialPort(self.get_parameter("port").value, self.get_parameter("baud").value)
            self.bus = FeetechBus(port)
        absent = [n for n in self.names if not self.bus.ping(self.ids[n])]
        if absent:
            raise RuntimeError(f"servos not answering: {[(n, self.ids[n]) for n in absent]} "
                               f"(power on? IDs set? port {self.get_parameter('port').value})")
        states_topic, cmd_topic = rdesc.hardware_topics(cfg, "topics")
        self.pub = self.create_publisher(JointState, states_topic, 10)
        self.pub_status = self.create_publisher(String, "~/status", 1)
        self.create_subscription(JointState, cmd_topic, self._cmd, 10)
        self.create_service(SetBool, "~/enable", self._enable)
        self.switch = self.create_client(SwitchController, "/controller_manager/switch_controller")
        self.ignore_until = 0.0
        self.enabled, self.cmd, self.goal, self.q, self.q_t = False, {}, {}, {}, None
        self.hold = HoldIntegrator([n for n in self.names if n != self.gripper],
                                   ki=float(self.get_parameter("hold_ki").value),
                                   max_rad=float(np.radians(self.get_parameter("hold_max_deg").value)))
        self.read_errors, self.last_status, self.health = 0, 0.0, {}
        self.bus.sync_write("torque_enable", {self.ids[n]: 0 for n in self.names})
        self._read()
        self.get_logger().info(f"SO-101 bridge: {len(self.names)} servos on {self.get_parameter('port').value}, "
                               f"torque OFF (enable: ros2 service call /so101_bridge/enable std_srvs/srv/SetBool "
                               f"'{{data: true}}'); joint states -> {states_topic}, commands <- {cmd_topic}")
        if self.get_parameter("auto_enable").value:
            self._set_torque(True)
        self.create_timer(1.0 / self.get_parameter("rate_hz").value, self._tick)

    # --- servo I/O ----------------------------------------------------------------------------------------------
    def _read_health(self):
        """Supply voltage (V), load (% of max, signed) and temperature (C) per servo, once a second: a supply that
        sags under load shows here before the bus drops out (2026-10-11: 10 failed reads mid-calibration)."""
        ids = [self.ids[n] for n in self.names]
        v, ld, t = (self.bus.sync_read(r, ids) for r in ("present_voltage", "present_load", "present_temperature"))
        def load(x):
            return round((-1 if x & 0x400 else 1) * (x & 0x3FF) / 10.0, 1)
        self.health = {n: {"V": v[i] / 10.0 if i in v else None, "load": load(ld[i]) if i in ld else None,
                           "C": t.get(i)} for n, i in self.ids.items() if n in self.names}

    def _read(self) -> bool:
        raw = self.bus.sync_read("present_position", [self.ids[n] for n in self.names])
        if len(raw) < len(self.names):
            self.read_errors += 1
            self.missing = [n for n in self.names if self.ids[n] not in raw]
            return False
        self.read_errors = 0
        now = time.time()
        q = {n: self.cal.to_rad(n, raw[self.ids[n]]) for n in self.names}
        self.dq = ({n: (q[n] - self.q[n]) / max(1e-3, now - self.q_t) for n in self.names}
                   if self.q and self.q_t else {n: 0.0 for n in self.names})
        self.q, self.q_t = q, now
        return True

    def _set_torque(self, on: bool):
        if on:
            self._read()
            self.hold.reset()
            self.goal = dict(self.q)                 # hold where it IS: no jump to an old command
            self.cmd = dict(self.q)
            # the trajectory controller still holds the pose of when IT started; the arm may have been moved by hand
            # since (torque off): restart the controllers so they hold the CURRENT pose, ignore commands meanwhile
            self.ignore_until = time.time() + 1.5
            # and follow commands only once one arrives NEAR the measured pose: a controller still holding a stale
            # pose (2026-10-11: the home pose -> the arm swung there on torque-on) can then never move the arm
            self.armed = False
            self._restart_controllers()
            self.bus.sync_write("goal_position", {self.ids[n]: self.cal.to_ticks(n, self.q[n]) for n in self.names})
        pg = int(self.get_parameter("p_gain").value)
        if on and pg > 0:
            for nm in self.names:
                if nm != self.gripper:
                    self.bus.write(self.ids[nm], "p_coefficient", pg)
        if on:      # the jaw squeezes until the servo stalls: cap its torque (0-1000 = 0-100 % of the servo's)
            self.bus.write(self.ids[self.gripper], "torque_limit", int(self.get_parameter("gripper_torque_limit").value))
        self.bus.sync_write("torque_enable", {self.ids[n]: int(on) for n in self.names})
        self.enabled = on
        self.get_logger().warn(f"torque {'ON: the arm holds its pose and follows commands' if on else 'OFF: arm limp'}")

    def _enable(self, req, resp):
        self._set_torque(bool(req.data))
        resp.success, resp.message = True, "torque on" if req.data else "torque off"
        return resp

    def _restart_controllers(self):
        if not self.switch.service_is_ready():
            self.get_logger().warn("controller_manager not reachable: controllers NOT restarted (stale hold pose?)")
            return
        req = SwitchController.Request()
        ctrls = [self.cfg["arm"]["controller"], self.cfg["gripper"]["controller"]]
        req.deactivate_controllers, req.activate_controllers = ctrls, ctrls
        req.strictness = SwitchController.Request.BEST_EFFORT
        self.switch.call_async(req)                  # fire and forget: commands are ignored until it has settled

    def _cmd(self, msg: JointState):
        if time.time() < self.ignore_until:
            return
        if not getattr(self, "armed", True):
            far = {n: abs(p - self.q[n]) for n, p in zip(msg.name, msg.position) if n in self.q and n != self.gripper}
            worst = max(far, key=far.get) if far else None
            if worst is not None and far[worst] > float(self.get_parameter("arm_gate_rad").value):
                self.get_logger().warn(f"ignoring commands until one is near the arm's pose ({worst} commanded "
                                       f"{np.degrees(far[worst]):.0f} deg away): holding", throttle_duration_sec=2.0)
                return
            self.armed = True
            self.get_logger().info("commands match the arm's pose: following them")
        for n, p in zip(msg.name, msg.position):
            if n in self.lo:
                self.cmd[n] = float(np.clip(p, self.lo[n], self.hi[n]))

    def _tick(self):
        if not self._read():
            if self.read_errors >= self.get_parameter("max_read_errors").value and self.enabled:
                self.get_logger().error(f"{self.read_errors} failed reads in a row: torque OFF for safety (not "
                                        f"answering: {getattr(self, 'missing', '?')}; last health {self.health})")
                self._set_torque(False)
            return
        m = JointState()
        m.header.stamp = self.get_clock().now().to_msg()
        m.name = list(self.names)
        m.position = [float(self.q[n]) for n in self.names]
        m.velocity = [float(self.dq[n]) for n in self.names]
        self.pub.publish(m)
        if self.enabled and self.cmd:
            dt = 1.0 / self.get_parameter("rate_hz").value
            out, moving = {}, {}
            for n in self.names:
                if n not in self.cmd:
                    continue
                vmax = self.get_parameter("gripper_speed_rad_s" if n == self.gripper else "max_speed_rad_s").value
                g = self.goal.get(n, self.q[n])
                self.goal[n] = g + float(np.clip(self.cmd[n] - g, -vmax * dt, vmax * dt))
                moving[n] = abs(self.goal[n] - g) > 1e-5
            off = self.hold.update(self.goal, self.q, self.dq, moving, dt)
            for n in self.goal:
                if n in self.cmd:
                    out[self.ids[n]] = self.cal.to_ticks(n, self.goal[n] + off.get(n, 0.0))
            self.bus.sync_write("goal_position", out)
        if time.time() - self.last_status > 1.0:
            self.last_status = time.time()
            self._read_health()
            self.pub_status.publish(String(data=json.dumps({"enabled": self.enabled, "read_errors": self.read_errors,
                                                            "q": {n: round(v, 4) for n, v in self.q.items()},
                                                            "health": self.health})))

    def shutdown(self):
        if self.get_parameter("torque_off_on_exit").value:
            try:
                self.bus.sync_write("torque_enable", {self.ids[n]: 0 for n in self.names})
            except Exception:  # noqa: BLE001
                pass


def main():
    rclpy.init()
    node = So101Bridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    node.shutdown()
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
