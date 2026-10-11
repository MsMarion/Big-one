"""Robot description + ros2_control / MoveIt configs GENERATED from a robot profile (config/robot_<name>.yaml).

Robot profile #1 (the FR3) has hand-written xacro + yaml files in pnp_bringup (they came with franka_ros2).
A profile with a `description:` section (the SO-101, and future robots) needs none: this module turns the vendor's
URDF + the profile into everything bringup needs, so adding a robot = a profile + its URDF/SRDF
(docs/plans/platform_design.md):
  urdf()                 vendor URDF, mesh paths made absolute, + our TCP link, + the <ros2_control> block
  ros2_controllers()     joint_state_broadcaster + a JointTrajectoryController (arm) + a GripperActionController
  moveit_controllers()   which controller MoveIt executes through
  joint_limits()         velocity / acceleration limits for planning
  kinematics()           IK plugin settings for the planning group
  rviz_config()          pnp_bringup's RViz configs (written for the FR3) with this robot's frames + planning group
  srdf_for()             our SRDF renamed to the URDF's robot (one SRDF serves variants, e.g. with a wrist camera)
  joint_offsets()        the active joint offset calibration (tools/fit_joint_offsets.py): urdf() folds each offset
                         into its joint's origin, so FK, IK, MoveIt and TF all use the corrected arm
Hardware (the <ros2_control> plugin) is the only sim / real difference:
  isaac   topic_based_ros2_control on /isaac_joint_states + /isaac_joint_command (Isaac Sim)
  topics  the same plugin on the robot's own topics (a servo bridge node, e.g. pnp_hardware so101_bridge)
  mock    mock_components/GenericSystem (states follow commands; no robot)
  servo_model  isaac, through the sim's servo error model (bringup servo_error:=<profile>, pnp_core/servo_error.py)
Pure Python (xml + dicts), unit-tested in tests/test_robot_description.py.
"""

from __future__ import annotations

import math
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml


def _origin(xyz, rpy) -> ET.Element:
    return ET.Element("origin", xyz=" ".join(f"{v:.6g}" for v in xyz), rpy=" ".join(f"{v:.6g}" for v in rpy))


def hardware_topics(robot_cfg: dict, hardware: str) -> tuple[str, str]:
    """(joint states topic, joint command topic) the topic_based plugin uses for this hardware."""
    if hardware == "isaac":
        return "/isaac_joint_states", "/isaac_joint_command"
    if hardware == "servo_model":     # Isaac behind pnp_manipulation/servo_error_node.py (sim servo errors)
        return "/servo_model/joint_states", "/servo_model/joint_command"
    name = robot_cfg["name"]
    return f"/{name}/hw_joint_states", f"/{name}/hw_joint_command"


def joint_offsets() -> dict:
    """{joint: offset [rad]} of the active joint offset calibration, or {}. Bringup sets PNP_JOINT_OFFSETS (for every
    node) to the cell's config/calibration/<cell>/joint_offsets_<source>.yaml when it exists (source: real, or
    sim_<servo_error profile>). Offset = true joint angle - encoder angle."""
    path = os.environ.get("PNP_JOINT_OFFSETS")
    if path is None:
        # a tool started outside the launch (docker exec ... python3 tools/grab.py) doesn't inherit the variable: use
        # what the running bringup chose, or its IK solves on the nominal arm while TF shows the calibrated one
        # (2026-10-11, real SO-101: ~13 mm apart at 20 cm reach -> hover 1-2 cm off, grabs missed)
        active = Path(os.environ.get("PNP_DATA", "/data")) / "pnp_active_joint_offsets"
        path = active.read_text().strip() if active.exists() else ""
    if not path or not Path(path).exists():
        return {}
    c = yaml.safe_load(Path(path).read_text()) or {}
    return {j: math.radians(float(v)) for j, v in (c.get("offsets_deg") or {}).items()}


def _rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])


def apply_joint_offsets(root: ET.Element, offsets: dict) -> None:
    """Fold joint offsets into the joints' origins: a joint whose true angle is (encoder + offset) is the same as a
    joint at the encoder angle whose origin is turned by the offset about its own axis."""
    for j in root.findall("joint"):
        d = offsets.get(j.get("name"))
        if not d:
            continue
        ax = j.find("axis")
        a = np.array([float(v) for v in (ax.get("xyz") if ax is not None else "1 0 0").split()])
        a /= np.linalg.norm(a)
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        Ra = np.eye(3) + math.sin(d) * K + (1 - math.cos(d)) * K @ K
        o = j.find("origin")
        if o is None:
            o = ET.SubElement(j, "origin", xyz="0 0 0", rpy="0 0 0")
        R = _rpy_matrix(*map(float, o.get("rpy", "0 0 0").split())) @ Ra
        rpy = (math.atan2(R[2, 1], R[2, 2]), math.asin(-max(-1.0, min(1.0, R[2, 0]))), math.atan2(R[1, 0], R[0, 0]))
        o.set("rpy", " ".join(f"{v:.9g}" for v in rpy))


def urdf(robot_cfg: dict, repo_root, hardware: str = "isaac", gripper_open_command: float | None = None,
         offsets: dict | None = None) -> str:
    """offsets: {joint: rad} folded into the joint origins; None = the active calibration (joint_offsets()),
    {} = the nominal arm (what the sim's servo model and the offset fit need)."""
    d = robot_cfg["description"]
    path = Path(repo_root) / d["urdf"]
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    apply_joint_offsets(root, joint_offsets() if offsets is None else offsets)
    base_dir = path.parent
    for m in root.iter("mesh"):                         # relative mesh paths -> file:// (RViz, MoveIt collision)
        fn = m.get("filename", "")
        if not fn.startswith(("package://", "file://")):
            m.set("filename", "file://" + (base_dir / fn).as_posix())
    tcp = d.get("tcp")
    if tcp:
        name = robot_cfg["frames"]["tcp"]
        root.append(ET.Element("link", name=name))
        j = ET.Element("joint", name=f"{name}_joint", type="fixed")
        j.append(ET.Element("parent", link=tcp["parent"]))
        j.append(ET.Element("child", link=name))
        j.append(_origin(tcp["xyz"], tcp["rpy"]))
        root.append(j)
    root.append(ros2_control_element(robot_cfg, hardware, gripper_open_command))
    return ET.tostring(root, encoding="unicode")


def srdf_for(srdf: str, urdf_text: str) -> str:
    """The SRDF with its <robot name> set to the URDF's. MoveIt reports an SRDF written for another robot name
    ("Semantic description is not specified for the same robot as the URDF"); our SO-101 SRDF serves the stock arm
    and the D405 variant (robot names so101_new_calib / so101_d405_wrist)."""
    name = ET.fromstring(urdf_text).get("name")
    return re.sub(r'(<robot\s[^>]*?name=")[^"]*', lambda m: m.group(1) + name, srdf, count=1)


def ros2_control_element(robot_cfg: dict, hardware: str, gripper_open_command: float | None = None) -> ET.Element:
    rc = ET.Element("ros2_control", name=f"{robot_cfg['name']}_system", type="system")
    hw = ET.SubElement(rc, "hardware")
    if hardware == "mock":
        ET.SubElement(hw, "plugin").text = "mock_components/GenericSystem"
    elif hardware in ("isaac", "topics", "servo_model"):
        states, commands = hardware_topics(robot_cfg, hardware)
        ET.SubElement(hw, "plugin").text = "topic_based_ros2_control/TopicBasedSystem"
        ET.SubElement(hw, "param", name="joint_commands_topic").text = commands
        ET.SubElement(hw, "param", name="joint_states_topic").text = states
        ET.SubElement(hw, "param", name="sum_wrapped_joint_states").text = "true"
    else:
        raise ValueError(f"unknown hardware {hardware!r} (isaac | topics | mock)")
    init = dict(zip(robot_cfg["arm"]["joints"], robot_cfg["arm"]["home"]))
    init[robot_cfg["gripper"]["driven_joint"]] = 0.0 if gripper_open_command is None else gripper_open_command
    for jn in list(robot_cfg["arm"]["joints"]) + [robot_cfg["gripper"]["driven_joint"]]:
        j = ET.SubElement(rc, "joint", name=jn)
        ET.SubElement(j, "command_interface", name="position")
        s = ET.SubElement(j, "state_interface", name="position")
        ET.SubElement(s, "param", name="initial_value").text = f"{init[jn]:.6g}"
        ET.SubElement(j, "state_interface", name="velocity")
    return rc


def ros2_controllers(robot_cfg: dict, update_rate: int = 100) -> dict:
    arm, g = robot_cfg["arm"], robot_cfg["gripper"]
    revolute_jaw = g.get("type") == "jaw"
    return {
        "controller_manager": {"ros__parameters": {
            "update_rate": update_rate,
            "joint_state_broadcaster": {"type": "joint_state_broadcaster/JointStateBroadcaster"},
            arm["controller"]: {"type": "joint_trajectory_controller/JointTrajectoryController"},
            g["controller"]: {"type": "position_controllers/GripperActionController"}}},
        arm["controller"]: {"ros__parameters": {
            "joints": list(arm["joints"]), "command_interfaces": ["position"],
            "state_interfaces": ["position", "velocity"], "allow_partial_joints_goal": False,
            "open_loop_control": False,
            # on (re)activation hold the MEASURED pose. Jazzy's default (true) holds the last COMMANDED value: on the
            # real SO-101 the torque-on controller restart then drove the arm to the startup (home) pose
            "set_last_command_interface_value_as_state_on_activation": False,
            # PNP_GOAL_TOLERANCE_RAD (the real SO-101): its joints sag under gravity (the elbow ~3 deg stretched out,
            # 12 V 2 A); with 0.02 every such move ended "CONTROL_FAILED" although the arm was where it should be
            "constraints": {"stopped_velocity_tolerance": 0.05, "goal_time": 1.0,
                            **{j: {"goal": float(os.environ.get("PNP_GOAL_TOLERANCE_RAD", "0.02"))}
                               for j in arm["joints"]}}}},
        g["controller"]: {"ros__parameters": {
            "joint": g["driven_joint"], "goal_tolerance": 0.02 if revolute_jaw else 0.002,
            "allow_stalling": True, "stall_velocity_threshold": 0.05 if revolute_jaw else 0.01,
            "stall_timeout": 0.5}},
    }


def moveit_controllers(robot_cfg: dict) -> dict:
    arm, g = robot_cfg["arm"], robot_cfg["gripper"]
    return {
        "moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager",
        "moveit_simple_controller_manager": {
            "controller_names": [arm["controller"], g["controller"]],
            arm["controller"]: {"type": "FollowJointTrajectory", "action_ns": "follow_joint_trajectory",
                                "default": True, "joints": list(arm["joints"])},
            g["controller"]: {"type": "GripperCommand", "action_ns": "gripper_cmd", "default": True,
                              "joints": [g["driven_joint"]]}},
        "trajectory_execution": {"allowed_execution_duration_scaling": 1.5, "allowed_goal_duration_margin": 1.0,
                                 "allowed_start_tolerance": 0.02},
    }


def joint_limits(robot_cfg: dict) -> dict:
    arm = robot_cfg["arm"]
    lim = {j: {"has_velocity_limits": True, "max_velocity": float(arm["max_velocity"]),
               "has_acceleration_limits": True, "max_acceleration": float(arm["max_acceleration"])}
           for j in arm["joints"]}
    lim[robot_cfg["gripper"]["driven_joint"]] = {"has_velocity_limits": True, "max_velocity": float(arm["max_velocity"]),
                                                 "has_acceleration_limits": True,
                                                 "max_acceleration": float(arm["max_acceleration"])}
    return {"default_velocity_scaling_factor": float(arm["velocity_scaling"]),
            "default_acceleration_scaling_factor": float(arm["acceleration_scaling"]), "joint_limits": lim}


def kinematics(robot_cfg: dict) -> dict:
    """KDL for the planning group (RViz's interactive marker, start-state checks). With fewer than 6 joints KDL
    can't hit full 6-D goals: our code solves pose goals itself (arm.ik: numeric) and gives MoveIt joint goals."""
    return {robot_cfg["arm"]["planning_group"]: {
        "kinematics_solver": "kdl_kinematics_plugin/KDLKinematicsPlugin",
        "kinematics_solver_search_resolution": 0.005, "kinematics_solver_timeout": 0.05,
        "position_only_ik": len(robot_cfg["arm"]["joints"]) < 6}}


# RViz keeps per-link / per-frame display settings under the link names; only these keys are robot-independent
_LINK_OPTIONS = {"All Links Enabled", "Expand Joint Details", "Expand Link Details", "Expand Tree", "Link Tree Style"}


def rviz_config(template: str, robot_cfg: dict) -> str:
    """One of pnp_bringup's RViz configs (rviz/*.rviz, written for the FR3) adapted to a robot.

    RViz draws everything in the FIXED FRAME: when that frame doesn't exist (fr3_link0 on an SO-101), every display
    reports a transform error and shows nothing. So: fixed frame = the robot's base, the MotionPlanning panel's
    planning group, the TF display's base / tool frames, no FR3 link names, no wrist-camera displays for a robot
    without a wrist camera, and the view from the profile's optional `rviz_view` (a small arm needs a closer look)."""
    cfg = yaml.safe_load(template)
    frames, wrist = robot_cfg["frames"], bool(robot_cfg.get("wrist_camera"))
    rename = {"fr3_link0": frames["base"], "fr3_hand_tcp": frames["tcp"]}

    def fix(node):
        if isinstance(node, list):
            for item in node:
                fix(item)
        if not isinstance(node, dict):
            return
        for key, val in list(node.items()):
            if key == "Links" and isinstance(val, dict):
                node[key] = {k: v for k, v in val.items() if k in _LINK_OPTIONS}
            elif key == "Frames" and isinstance(val, dict):          # TF display: which frames it draws
                node[key] = {rename.get(k, k): v for k, v in val.items()
                             if wrist or not k.startswith("wrist_camera")}
            elif key == "Planning Group":
                node[key] = robot_cfg["arm"]["planning_group"]
            elif key == "Displays" and isinstance(val, list):
                node[key] = [d for d in val if wrist or "wrist" not in str(d.get("Name", "")).lower()]
                fix(node[key])
            else:
                fix(val)

    fix(cfg)
    vm = cfg["Visualization Manager"]
    # the workcell's clickable objects + draggable goal (pnp_manipulation/workcell_node.py)
    vm.setdefault("Displays", []).append({
        "Class": "rviz_default_plugins/InteractiveMarkers", "Enabled": True, "Name": "Workcell (click / drag / right-click)",
        "Interactive Markers Namespace": "/workcell", "Show Axes": False, "Show Descriptions": True,
        "Show Visual Aids": False, "Value": True})
    vm["Global Options"]["Fixed Frame"] = frames["base"]
    view = robot_cfg.get("rviz_view")
    if view:
        cur = vm["Views"]["Current"]
        cur["Distance"] = float(view["distance"])
        cur["Focal Point"] = dict(zip("XYZ", (float(v) for v in view["focal_point"])))
    return yaml.safe_dump(cfg, sort_keys=False, width=100000)
