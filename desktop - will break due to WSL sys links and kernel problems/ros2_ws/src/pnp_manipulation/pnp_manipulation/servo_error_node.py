"""Sim only: the robot's joints behave like real hobby servos (pnp_core/servo_error.py, config/servo_error.yaml).

    bringup.launch.py servo_error:=typical      (sim; ros2_control then talks to this node instead of Isaac)

    ros2_control --/servo_model/joint_command--> [this node: where the joints REALLY go] --/isaac_joint_command--> Isaac
    ros2_control <--/servo_model/joint_states--- [this node: what the encoders REPORT] <--/isaac_joint_states--- Isaac

Like the camera sensor model (pnp_perception/sensor_model_node.py), Isaac stays perfect and the imperfection lives in
one node in the container. /sim/servo_error (std_msgs/String JSON, latched) has the truth: each joint's offset and
current gravity flex, for checks that measure the arm from outside (tools/arm_accuracy_check.py).
"""

import json

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from pnp_core import robot_description as rdesc
from pnp_core.catalog import load_robot, load_yaml, repo_root
from pnp_core.servo_error import GravityModel, ServoErrorModel


class ServoErrorNode(Node):
    def __init__(self):
        super().__init__("servo_error_node")
        self.declare_parameter("profile", "typical")
        self.declare_parameter("seed", -1)                   # -1: the profile's seed
        robot = load_robot()
        name = self.get_parameter("profile").value
        prof = load_yaml("config/servo_error.yaml")["profiles"][name] or {}
        joints = list(robot["arm"]["joints"])
        grav = GravityModel(rdesc.urdf(robot, repo_root(), "mock", offsets={}), robot["frames"]["base"], joints) \
            if prof.get("compliance_deg_per_nm") else None
        seed = self.get_parameter("seed").value
        self.model = ServoErrorModel.from_profile(prof, joints, grav, None if seed < 0 else seed)
        self.q_true = {}
        self.pub_cmd = self.create_publisher(JointState, "/isaac_joint_command", 10)
        self.pub_js = self.create_publisher(JointState, "/servo_model/joint_states", 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.pub_truth = self.create_publisher(String, "/sim/servo_error", latched)
        self.create_subscription(JointState, "/servo_model/joint_command", self._on_cmd, 10)
        self.create_subscription(JointState, "/isaac_joint_states", self._on_js, 10)
        self.create_timer(1.0, self._publish_truth)
        offs = {j: round(v * 57.2958, 2) for j, v in self.model.offset.items()}
        self.get_logger().info(f"servo errors '{name}': offsets [deg] {offs}, deadband "
                               f"{prof.get('deadband_deg', 0)} deg, flex {prof.get('compliance_deg_per_nm', 0)} deg/N m")

    def _on_cmd(self, m: JointState):
        out = self.model.command(dict(zip(m.name, m.position)))
        o = JointState(header=m.header, name=list(m.name), position=[float(out[n]) for n in m.name],
                       velocity=list(m.velocity), effort=list(m.effort))
        self.pub_cmd.publish(o)

    def _on_js(self, m: JointState):
        self.q_true = dict(zip(m.name, m.position))
        rep = self.model.report(self.q_true)
        o = JointState(header=m.header, name=list(m.name), position=[float(rep[n]) for n in m.name],
                       velocity=list(m.velocity), effort=list(m.effort))
        self.pub_js.publish(o)

    def _publish_truth(self):
        if self.q_true:
            self.pub_truth.publish(String(data=json.dumps({"profile": self.get_parameter("profile").value,
                                                           "joints": self.model.truth(self.q_true)})))


def main():
    rclpy.init()
    n = ServoErrorNode()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    rclpy.try_shutdown()
