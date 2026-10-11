"""Bring up the ROS 2 side of the pick & place system.

    ros2 launch pnp_bringup bringup.launch.py                 # sim (Isaac Sim + zenoh link running)
    ros2 launch pnp_bringup bringup.launch.py use_rviz:=false
    ros2 launch pnp_bringup bringup.launch.py obstacle_source:=perceived   # Phase 5B: obstacles from depth
    ros2 launch pnp_bringup bringup.launch.py obstacle_source:=perceived sensor_view:=true   # every sensor stage in RViz
    ros2 launch pnp_bringup bringup.launch.py sim:=false camera:=local     # REAL D435i on this PC (usbipd)
    ros2 launch pnp_bringup bringup.launch.py sim:=false camera:=jetson    # REAL D435i on a Jetson (network)
    ros2 launch pnp_bringup bringup.launch.py scene:=config/scenes/cracker_box.yaml   # another target object
    ros2 launch pnp_bringup bringup.launch.py map_mode:=freeze_per_task change_policy:=stop   # frozen map per task

scene: the scene file (config/scenes/) decides the TARGET OBJECT for every node (detector class + shape, pose
mesh, obstacle filter size, grasp rules) and its YOLO model (target.detector_model). The sim must run the same
scene (run_sim.py --scene ...; setup/stack.ps1 -Scene).

sim:=false (no Isaac): no sim time; the arm is a MOCK by default (hardware:=mock: ros2_control's
GenericSystem, states follow commands) so MoveIt + RViz work. Pick & place, the workcell page and the wrist map
run as in the sim, except on the mock arm (it can't really grasp: manipulation:=true to try anyway). Camera pose from config/calibration/d435i_real.yaml.

obstacle_source:  known      boxes from /scene/known_obstacles -> MoveIt collision objects (Layer A)
                  perceived  the depth camera -> Octomap; known boxes are ignored (Layer B)

Starts:
  robot_state_publisher   URDF -> TF of the robot (from /joint_states)
  ros2_control_node       hardware plugin (sim: topic_based <-> Isaac) + controllers:
                            joint_state_broadcaster -> /joint_states
                            fr3_arm_controller      -> FollowJointTrajectory action
                            fr3_gripper             -> GripperCommand action
  static TF               fr3_link0 -> camera_link -> camera_color_optical_frame (from config/camera_d435i.yaml)
  move_group              MoveIt 2 (OMPL + Pilz LIN), executing through the controllers above
  detector_node           YOLO-seg + depth -> /perception/detections, /perception/detect (if the model exists)
  obstacle_cloud_node     (obstacle_source:=perceived) depth -> /perception/obstacle_cloud -> MoveIt Octomap
  pick_place_node         /pick_place action (pnp_interfaces/PickPlace): the full pick & place task
  rviz2                   MotionPlanning panel + detection markers + perception debug image

All nodes use simulation time (/clock from Isaac) when sim:=true.
Configuration comes from the repo's config/ (robot_fr3.yaml, camera_d435i.yaml) via $PNP_REPO.
"""

import os
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

from pnp_core import robot_description as rdesc
from pnp_core.catalog import robot_name
from pnp_core.gripper import GripperModel
from pnp_core.urdf_tools import clamp_joint_limits
from pnp_core.wrist import housing_in_description, housing_link, urdf_snippet, wrist_camera

REPO = Path(os.environ.get("PNP_REPO", "/repo"))


def perception_nodes(context, use_sim_time, scene_cfg):
    """YOLO-seg + depth detector (pnp_perception). Skipped with a warning if the model isn't trained yet."""
    if LaunchConfiguration("perception").perform(context).lower() != "true":
        return []
    model = LaunchConfiguration("model").perform(context) or scene_cfg["target"].get("detector_model",
                                                                                     "models/soup_can_seg_v1.pt")
    model = model if Path(model).is_absolute() else REPO / model
    model = Path(model)
    if not model.is_file():
        print(f"[bringup] WARNING: perception model {model} not found -> perception NOT started "
              f"(train it: tools/train_yolo.py)")
        return []
    # prefix: run the node with the image's venv Python (PyTorch/Ultralytics live there). colcon writes the
    # entry-point script with the *system* python shebang, which can't import ultralytics.
    return [Node(package="pnp_perception", executable="detector_node", output="screen",
                 prefix=os.environ.get("PNP_VENV_PYTHON", "/opt/pnp_venv/bin/python3"),
                 parameters=[{"model": str(model), "object": scene_cfg["target"]["object"]}, use_sim_time])]


def camera_nodes(camera, wrist=None):
    """Real camera sources. Both end with the SAME raw topics the sim publishes (config/interfaces.yaml)."""
    if camera == "wrist":
        # ONLY the wrist D405 on this computer's USB (the Jetson in the robot role, before the D435i exists)
        if not wrist:
            raise RuntimeError("camera:=wrist but this robot has no wrist camera")
        params = yaml.safe_load((REPO / "config/realsense_d405.yaml").read_text())
        # PNP_RS_INITIAL_RESET=false (the Jetson compose file): no power-cycle at start. On the Jetson the reset
        # sometimes didn't come back (USB 3 link power "U1/U2 failed", then gone until replugged).
        if os.environ.get("PNP_RS_INITIAL_RESET"):
            params["initial_reset"] = os.environ["PNP_RS_INITIAL_RESET"].lower() == "true"
        return [Node(package="realsense2_camera", executable="realsense2_camera_node", name="camera",
                     namespace="wrist_camera", output="screen", parameters=[params])]
    if camera == "local":
        # The RealSense driver right here (camera on this PC's USB, passed into WSL with usbipd; see
        # docs/learning/08_real_sensors.md). The container must have USB access (run_dev.sh with PNP_USB=1).
        # + the wrist D405 when the robot has one (each driver picks its camera by device_type / serial_no).
        nodes = []
        for cfg, ns in (("config/realsense_d435i.yaml", "camera"),) + ((("config/realsense_d405.yaml", "wrist_camera"),)
                                                                      if wrist else ()):
            params = yaml.safe_load((REPO / cfg).read_text())
            if os.environ.get("PNP_RS_INITIAL_RESET"):           # (as for camera:=wrist: the Jetson's resets hung)
                params["initial_reset"] = os.environ["PNP_RS_INITIAL_RESET"].lower() == "true"
            nodes.append(Node(package="realsense2_camera", executable="realsense2_camera_node", name="camera",
                              namespace=ns, output="screen", parameters=[params]))
        return nodes
    if camera == "jetson":
        # The Jetson sends JPEG color + lossless PNG depth (image_transport compressed / compressedDepth) through
        # zenoh; decompress back into the raw topics everything else uses. Both cameras: the fixed D435i and, when the
        # robot has one, the wrist D405 (the calibration tools on the PC read its raw image).
        nodes = []
        for base, tag in (("/camera/camera", ""),) + ((("/wrist_camera/camera", "wrist_"),) if wrist else ()):
            nodes += [
                Node(package="image_transport", executable="republish", name=f"decompress_{tag}color", output="log",
                     parameters=[{"in_transport": "compressed", "out_transport": "raw"}],
                     remappings=[("in/compressed", f"{base}/color/image_raw/compressed"),
                                 ("out", f"{base}/color/image_raw")]),
                Node(package="image_transport", executable="republish", name=f"decompress_{tag}depth", output="log",
                     parameters=[{"in_transport": "compressedDepth", "out_transport": "raw"}],
                     remappings=[("in/compressedDepth", f"{base}/aligned_depth_to_color/image_raw/compressedDepth"),
                                 ("out", f"{base}/aligned_depth_to_color/image_raw")]),
            ]
        return nodes
    return []          # sim: Isaac publishes the camera


def _setup(context):
    sim = LaunchConfiguration("sim").perform(context).lower() == "true"
    camera = LaunchConfiguration("camera").perform(context) or ("sim" if sim else "local")
    hardware = LaunchConfiguration("hardware").perform(context) or ("isaac" if sim else "mock")
    # the same stack as the sim on the REAL arm too (2026-10-11: the real cell came up without pick & place, the
    # workcell page and RViz's clickable objects); only the mock arm, which can't grasp, leaves it off
    manipulation = LaunchConfiguration("manipulation").perform(context) or ("false" if hardware == "mock" else "true")
    if sim and camera != "sim":
        raise RuntimeError("camera:=local/jetson needs sim:=false (Isaac publishes the same camera topics)")
    use_sim_time = {"use_sim_time": sim}
    scene = LaunchConfiguration("scene").perform(context)
    if not scene.startswith("config/") and not scene.endswith(".yaml"):
        scene = f"config/scenes/{scene}.yaml"                     # scene:=so101_lego works too
    scene_cfg = yaml.safe_load((REPO / scene).read_text())
    rname = robot_name(scene_cfg)
    robot = yaml.safe_load((REPO / f"config/robot_{rname}.yaml").read_text())
    print(f"[bringup] scene {scene}: robot {rname}, target object '{scene_cfg['target']['object']}', hardware {hardware}")
    # every node (and tools started later in the container) works on THIS cell (pnp_core.catalog.active_scene_path)
    os.environ["PNP_SCENE"] = scene
    try:
        Path("/data").mkdir(exist_ok=True)
        Path("/data/pnp_active_scene").write_text(scene)
    except OSError:
        pass
    env_actions = [SetEnvironmentVariable("PNP_SCENE", scene)]
    if hardware == "real" and rname == "fr3":
        raise RuntimeError("hardware:=real for the FR3 (franka_hardware) is not set up; the SO-101 has its servo bridge")
    pkg = Path(get_package_share_directory("pnp_bringup"))
    generated = "description" in robot          # profile robots: description + configs generated, not hand-written
    gen_dir = Path("/tmp/pnp_generated")
    if generated:
        gen_dir.mkdir(parents=True, exist_ok=True)
        hw = {"real": "topics"}.get(hardware, hardware)
        # joint offset calibration (tools/fit_joint_offsets.py): one per hardware source; every node gets it via
        # PNP_JOINT_OFFSETS (pnp_core.robot_description.urdf folds it into the joints)
        se = LaunchConfiguration("servo_error").perform(context)
        source = f"sim_{se if se and se != 'none' else 'none'}" if sim else "real"
        jo = REPO / "config" / "calibration" / scene_cfg.get("cell", "fr3_table") / f"joint_offsets_{source}.yaml"
        if jo.exists():
            os.environ["PNP_JOINT_OFFSETS"] = str(jo)
            env_actions.append(SetEnvironmentVariable("PNP_JOINT_OFFSETS", str(jo)))
            print(f"[bringup] joint offsets: {jo.relative_to(REPO)} {rdesc.joint_offsets()}")
        else:
            os.environ.pop("PNP_JOINT_OFFSETS", None)
            env_actions.append(SetEnvironmentVariable("PNP_JOINT_OFFSETS", ""))
        try:        # for tools started OUTSIDE this launch (docker exec): same arm model as TF (rdesc.joint_offsets)
            Path("/data/pnp_active_joint_offsets").write_text(str(jo) if jo.exists() else "")
        except OSError:
            pass
        # sim servo errors (config/servo_error.yaml): ros2_control talks to servo_error_node, which talks to Isaac
        servo_error = LaunchConfiguration("servo_error").perform(context)
        if servo_error and servo_error != "none" and hw == "isaac":
            hw = "servo_model"
            print(f"[bringup] sim servo errors: profile '{servo_error}' (pnp_manipulation/servo_error_node.py)")
        open_cmd = GripperModel(robot["gripper"]).command(robot["gripper"]["max_width_m"])
        (gen_dir / f"{rname}.urdf").write_text(rdesc.urdf(robot, REPO, hw, open_cmd))
        for name, fn in (("ros2_controllers", rdesc.ros2_controllers), ("moveit_controllers", rdesc.moveit_controllers),
                         ("joint_limits", rdesc.joint_limits), ("kinematics", rdesc.kinematics)):
            (gen_dir / f"{rname}_{name}.yaml").write_text(yaml.safe_dump(fn(robot), sort_keys=False))
        srdf = gen_dir / f"{rname}.srdf"
        srdf.write_text(rdesc.srdf_for((REPO / robot["description"]["srdf"]).read_text(),
                                       (gen_dir / f"{rname}.urdf").read_text()))
        controllers_yaml = gen_dir / f"{rname}_ros2_controllers.yaml"
    else:
        srdf = Path(get_package_share_directory("franka_description")) / "robots/fr3/fr3.srdf.xacro"
        controllers_yaml = pkg / "config/ros2_controllers.yaml"

    perceived = LaunchConfiguration("obstacle_source").perform(context) == "perceived"
    sensor_view = LaunchConfiguration("sensor_view").perform(context).lower() == "true"
    # pose_estimation: true / false, or auto (default) = the scene's perception.pose_estimation (the SO-101 cells:
    # on -- FoundationPose tracks the clicked / picked brick), else off
    pe = LaunchConfiguration("pose_estimation").perform(context).lower()
    pose_estimation = (pe == "true" if pe in ("true", "false")
                       else bool(scene_cfg.get("perception", {}).get("pose_estimation", False)))
    fp_python = "/opt/fp_venv/bin/python3"
    if pose_estimation and not os.path.exists(fp_python):
        print("[bringup] WARNING pose_estimation:=true needs the pnp:pose image (PNP_IMAGE=pnp:pose run_dev.sh): skipped")
        pose_estimation = False
    if generated:
        builder = (
            MoveItConfigsBuilder(rname, package_name="pnp_bringup")
            .robot_description(file_path=str(gen_dir / f"{rname}.urdf"))
            .robot_description_semantic(file_path=str(srdf))
            .robot_description_kinematics(file_path=str(gen_dir / f"{rname}_kinematics.yaml"))
            .joint_limits(file_path=str(gen_dir / f"{rname}_joint_limits.yaml"))
            .trajectory_execution(file_path=str(gen_dir / f"{rname}_moveit_controllers.yaml"))
            .planning_pipelines(pipelines=["ompl", "pilz_industrial_motion_planner"], default_planning_pipeline="ompl")
            .pilz_cartesian_limits(file_path="config/pilz_cartesian_limits.yaml")
            .planning_scene_monitor(publish_robot_description=True, publish_robot_description_semantic=True)
        )
    else:
        builder = (
            MoveItConfigsBuilder("fr3", package_name="pnp_bringup")
            .robot_description(file_path="urdf/fr3_sim.urdf.xacro",
                               mappings={"home": " ".join(str(v) for v in robot["arm"]["home"]), "hardware": hardware})
            .robot_description_semantic(file_path=str(srdf), mappings={"hand": "true", "ee_id": "franka_hand"})
            .robot_description_kinematics(file_path="config/kinematics.yaml")
            .joint_limits(file_path="config/joint_limits.yaml")
            .trajectory_execution(file_path="config/moveit_controllers.yaml")
            .planning_pipelines(pipelines=["ompl", "pilz_industrial_motion_planner"], default_planning_pipeline="ompl")
            .pilz_cartesian_limits(file_path="config/pilz_cartesian_limits.yaml")
            .planning_scene_monitor(publish_robot_description=True, publish_robot_description_semantic=True)
        )
    if perceived:
        builder = builder.sensors_3d(file_path="config/octomap_sensors.yaml")
    moveit_config = builder.to_moveit_configs()
    # MoveIt grows the robot and the TABLE by padding_offset before it drops their points from the Octomap: the
    # default 2 cm also wipes 1 cm bricks off the desk. A cell with small objects sets its own (perception.octomap_padding_m)
    pad = scene_cfg.get("perception", {}).get("octomap_padding_m")
    if perceived and pad is not None and isinstance(getattr(moveit_config, "sensors_3d", None), dict):
        for name in moveit_config.sensors_3d.get("sensors", []):
            moveit_config.sensors_3d[name]["padding_offset"] = float(pad)
        print(f"[bringup] octomap padding {float(pad) * 1000:.0f} mm (scene perception.octomap_padding_m)")
    # The real FR3 (and Isaac's fr3.usd) has narrower joint limits than franka_description's URDF: clamp, so
    # MoveIt, robot_state_publisher and ros2_control all use limits the robot can actually reach.
    urdf, changes = clamp_joint_limits(str(moveit_config.robot_description["robot_description"]),
                                       robot["arm"]["position_limits"], robot["arm"]["limit_margin"])
    # Wrist camera (config/robot_<name>.yaml wrist_camera + wrist_camera_mount): a fixed camera link on the hand, from
    # the same mount the sim uses (pnp_core/wrist.py) -> MoveIt avoids the housing. (The FR3: a box we add here; the
    # SO-101: the camera is in its URDF already.) The camera frames' TF comes from camera_tf_node (calibration).
    wcfg = wrist_camera(robot)
    if wcfg:
        if not housing_in_description(wcfg):
            urdf = urdf.replace("</robot>", urdf_snippet(wcfg) + "</robot>")
        srdf = moveit_config.robot_description_semantic["robot_description_semantic"]
        h = housing_link(wcfg)
        near = dict.fromkeys([wcfg["mount"]["parent"]] + [n for n in robot.get("hand_links", []) if n != h])
        allow = "".join(f'<disable_collisions link1="{h}" link2="{n}" reason="Adjacent"/>' for n in near)
        moveit_config.robot_description_semantic["robot_description_semantic"] = srdf.replace("</robot>", allow + "</robot>")
        print(f"[bringup] wrist camera {wcfg['name']} ({wcfg['profile']}) on {wcfg['mount']['parent']}")
    moveit_config.robot_description["robot_description"] = urdf
    for c in changes:
        print(f"[bringup] joint limit clamped (config/robot_{rname}.yaml position_limits): {c}")
    voxel = float(scene_cfg.get("perception", {}).get("obstacle_cloud", {}).get("voxel_m", 0.02))
    octomap = {"octomap_frame": robot["frames"]["base"], "octomap_resolution": voxel, "max_range": 2.0}

    # Both cameras' TF (fixed: base -> camera; wrist: hand -> camera) from the calibration files, else the nominal
    # mounts (real fixed camera: also config/calibration/d435i_real.yaml); reloadable live after a calibration
    # (pnp_perception/camera_tf_node.py, docs/learning/16_calibration.md)
    # sim: one calibration set per reality profile (each puts the cameras elsewhere); tools/set_reality.py switches it
    reality = (LaunchConfiguration("reality").perform(context)
               or yaml.safe_load((REPO / "config/sim_reality.yaml").read_text()).get("active", "ideal"))
    cal_set = {"calibration_set": f"sim_{reality}" if sim else "real"}
    tfs = [Node(package="pnp_perception", executable="camera_tf_node", output="log", parameters=[cal_set, use_sim_time]),
           # watches the AprilTag with the fixed camera: has the camera moved since its calibration?
           # (venv Python: the ArUco / AprilTag detector needs OpenCV >= 4.7; the system's 4.6 segfaulted)
           Node(package="pnp_perception", executable="calibration_monitor_node", output="log",
                prefix=os.environ.get("PNP_VENV_PYTHON", "/opt/pnp_venv/bin/python3"),
                parameters=[use_sim_time])]
    # a hand in the workspace pauses a running arm move (pnp_perception/hand_detector_node.py; sim: the magenta
    # hand_prop stand-in, real: MediaPipe in the venv). hand_detector:=false: nothing pauses
    if LaunchConfiguration("hand_detector").perform(context).lower() != "false":
        tfs.append(Node(package="pnp_perception", executable="hand_detector_node", output="log",
                        prefix=os.environ.get("PNP_VENV_PYTHON", "/opt/pnp_venv/bin/python3"),
                        parameters=[{"sim": sim}, use_sim_time]))

    servo_error = LaunchConfiguration("servo_error").perform(context)
    nodes = [
        *([Node(package="pnp_manipulation", executable="servo_error_node", output="screen",
                parameters=[{"profile": servo_error}, use_sim_time])]
          if generated and sim and hardware == "isaac" and servo_error and servo_error != "none" else []),
        Node(package="robot_state_publisher", executable="robot_state_publisher", output="log",
             parameters=[moveit_config.robot_description, use_sim_time]),
        Node(package="controller_manager", executable="ros2_control_node", output="screen",
             parameters=[str(controllers_yaml), use_sim_time],
             remappings=[("~/robot_description", "/robot_description")]),
        Node(package="controller_manager", executable="spawner", output="screen",
             arguments=["joint_state_broadcaster", robot["arm"]["controller"], robot["gripper"]["controller"],
                        "--controller-manager-timeout", "60"]),
        Node(package="moveit_ros_move_group", executable="move_group", output="screen",
             # OMPL checks a path for collisions only every `longest_valid_segment_fraction` of the state space
             # extent (default 0.005: several cm of fingertip motion on the FR3), so it can step THROUGH a thin
             # obstacle; MoveIt's final check then rejects the plan (INVALID_MOTION_PLAN). Phase 5's 3 cm wall
             # hit exactly that -> check ~5x finer.
             # fix_start_state: a joint that settled a hair past its (clamped) limit, e.g. 0.5 mrad after a move to the
             # limit, made EVERY later plan fail with START_STATE_INVALID ("Start state out of bounds"); the robot was
             # stuck for good. With it, MoveIt clamps such a start state into the bounds and plans.
             parameters=[moveit_config.to_dict(),
                         {"ompl": {robot["arm"]["planning_group"]: {"longest_valid_segment_fraction": 0.001},
                                   "fix_start_state": True},
                          "pilz_industrial_motion_planner": {"fix_start_state": True}},
                         *([octomap] if perceived else []), use_sim_time]),
        *camera_nodes(camera, wcfg),
        # sim: perfect Isaac images (/sim_raw/...) -> realistic standard camera topics (config/sim_reality.yaml)
        *([Node(package="pnp_perception", executable="sensor_model_node", output="log",
                parameters=[use_sim_time, *([{"profile": LaunchConfiguration("reality").perform(context)}]
                                            if LaunchConfiguration("reality").perform(context) else [])])]
          if camera == "sim" and yaml.safe_load((REPO / "config/interfaces.yaml").read_text()).get("sim_raw_prefix")
          else []),
        *perception_nodes(context, use_sim_time, scene_cfg),
        *([Node(package="pnp_perception", executable="obstacle_cloud_node", output="screen",
                parameters=[{"restamp": not sim, "scene": scene}, use_sim_time])] if perceived or sensor_view else []),
        # sensor view: the colored point cloud + the measured scene model (pnp_perception), shown in sensors.rviz
        *([Node(package="pnp_perception", executable="camera_cloud_node", output="log", parameters=[use_sim_time]),
           Node(package="pnp_perception", executable="scene_model_node", output="log",
                parameters=[{"scene": scene}, use_sim_time])]
          if sensor_view else []),
        # 6-D pose (FoundationPose): own Python env (NumPy 2) in the pnp:pose image, docs/learning/10_6d_pose.md
        *([Node(package="pnp_perception", executable="pose_estimator_node", output="screen", prefix=fp_python,
                parameters=[{"object": scene_cfg["target"]["object"]}, use_sim_time])] if pose_estimation else []),
        # close look from wrist DEPTH instead (no FoundationPose mesh for the object, e.g. LEGO bricks): scene key
        # perception.wrist_refine: depth (pnp_core/wrist_refine.py)
        *([Node(package="pnp_perception", executable="wrist_refine_node", output="screen", parameters=[use_sim_time])]
          if wcfg and not pose_estimation and scene_cfg.get("perception", {}).get("wrist_refine") == "depth"
          # PNP_WRIST_REFINE=false (the Jetson, until it starts on demand): it decoded every wrist image (67 % CPU)
          and os.environ.get("PNP_WRIST_REFINE", "true").lower() != "false" else []),
        Node(package="pnp_manipulation", executable="pick_place_node", output="screen",
             parameters=[{"use_known_obstacles": not perceived, "scene": scene,
                          "map_mode": LaunchConfiguration("map_mode").perform(context),
                          "change_policy": LaunchConfiguration("change_policy").perform(context)}, use_sim_time],
             condition=IfCondition(manipulation)),
        # the workcell page (http://localhost:8088) + RViz's clickable objects / goal: scan, choose, pick
        Node(package="pnp_manipulation", executable="workcell_node", output="log", parameters=[use_sim_time],
             condition=IfCondition(LaunchConfiguration("workcell").perform(context) or manipulation)),
        # the desk flat map from the wrist camera (service /wrist_map/scan): robots with a wrist camera
        Node(package="pnp_manipulation", executable="wrist_map_node", output="log", parameters=[use_sim_time],
             condition=IfCondition(manipulation if robot.get("wrist_camera") else "false")),
        Node(package="rviz2", executable="rviz2", output="log",
             arguments=["-d", str(rviz_file(pkg, gen_dir, robot, sensor_view))],
             parameters=[moveit_config.robot_description, moveit_config.robot_description_semantic,
                         moveit_config.robot_description_kinematics, moveit_config.planning_pipelines,
                         moveit_config.joint_limits, use_sim_time],
             condition=IfCondition(LaunchConfiguration("use_rviz"))),
    ]
    hw_mock = LaunchConfiguration("hw_mock").perform(context).lower() == "true"
    hw_port = LaunchConfiguration("hw_port").perform(context)
    hw_calibration = LaunchConfiguration("hw_calibration").perform(context)
    out = env_actions + tfs + hardware_nodes(robot, hardware, sim, hw_mock, hw_port, hw_calibration) + nodes
    # role: the whole stack on one computer (all), or split across two (docs/learning/22_jetson_d405.md):
    #   robot  next to the arm (the Jetson): servo bridge, ros2_control, robot description / TF, MoveIt, the wrist
    #          camera + its TF, the wrist flat map -- everything that touches hardware or needs the camera's raw images
    #   brain  the PC: perception on the fixed camera, pick & place, the workcell page, RViz
    role = LaunchConfiguration("role").perform(context) or "all"
    if role != "robot":                         # the PC side keeps the failure ring buffer (big images)
        out += ring_buffer(LaunchConfiguration("bag_buffer").perform(context))
    if role == "all":
        return out
    robot_side = {"so101_bridge", "robot_state_publisher", "ros2_control_node", "spawner", "move_group",
                  "realsense2_camera_node", "camera_tf_node", "wrist_map_node", "wrist_refine_node",
                  "servo_error_node"}

    def side(a):
        exe = getattr(a, "node_executable", None) or getattr(a, "_Node__node_executable", None)
        if exe is None:                         # SetEnvironmentVariable etc.: both sides need them
            return None
        name = exe if isinstance(exe, str) else "".join(getattr(t, "text", "") for t in exe)
        return "robot" if name in robot_side else "brain"
    kept = [a for a in out if side(a) in (None, role)]
    print(f"[bringup] role {role}: {sum(1 for a in kept if side(a))} of {sum(1 for a in out if side(a))} nodes")
    return kept


def ring_buffer(flag):
    """bag_buffer:=true: ros2 bag record in SNAPSHOT mode -- the last seconds of every session topic
    (config/recording.yaml) kept in memory, written to /data/bags/ring_<time> only when pick_place_node calls
    /rosbag2_recorder/snapshot after a failed task (pnp_core/trials.py artifacts links it to the attempt)."""
    if str(flag).lower() != "true":
        return []
    import time as _t
    rec = yaml.safe_load(open("/repo/config/recording.yaml"))
    out = f"/data/bags/ring_{_t.strftime('%Y%m%d_%H%M%S')}"
    print(f"[bringup] failure ring buffer -> {out} ({rec['ring']['max_cache_bytes'] / 1e6:.0f} MB in memory)")
    return [ExecuteProcess(cmd=["ros2", "bag", "record", "--snapshot-mode", "--max-cache-size",
                                str(int(rec["ring"]["max_cache_bytes"])), "-s", "mcap", "--include-hidden-topics",
                                "--qos-profile-overrides-path", "/repo/config/recording_qos.yaml",
                                "-o", out, "--topics", *rec["session_topics"]], output="log")]


def rviz_file(pkg, gen_dir, robot, sensor_view):
    """The RViz config for THIS robot: rviz/*.rviz (written for the FR3) with the robot's fixed frame, planning group
    and view (pnp_core.robot_description.rviz_config); with the FR3's fixed frame an SO-101 shows only TF errors."""
    name = "sensors.rviz" if sensor_view else "moveit.rviz"
    gen_dir.mkdir(parents=True, exist_ok=True)
    out = gen_dir / name
    out.write_text(rdesc.rviz_config((pkg / "rviz" / name).read_text(), robot))
    return out


def hardware_nodes(robot, hardware, sim, hw_mock=False, hw_port="", hw_calibration=""):
    """hardware:=real for a profile robot: its servo bridge (the topic_based plugin talks to it, like to Isaac)."""
    if sim or hardware != "real":
        return []
    bridge = robot.get("real", {}).get("bridge")
    if not bridge:
        raise RuntimeError(f"robot {robot['name']} has no real.bridge in its profile")
    return [Node(package=bridge["package"], executable=bridge["executable"], output="screen",
                 prefix=os.environ.get("PNP_VENV_PYTHON", "/opt/pnp_venv/bin/python3"),
                 parameters=[{"robot": robot["name"], "mock": hw_mock,
                              **({"port": hw_port} if hw_port else {}),
                              **({"calibration": hw_calibration} if hw_calibration else {})}])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("sim", default_value="true", description="true: Isaac Sim, false: real sensors/robot"),
        DeclareLaunchArgument("camera", default_value="",
                              description="sim | local (D435i + wrist D405 here) | wrist (only the wrist D405 here) | "
                                          "jetson | none (default: sim if sim:=true, else local)"),
        DeclareLaunchArgument("hardware", default_value="",
                              description="isaac | mock | real (default: isaac if sim:=true, else mock). real = the "
                                          "robot's own driver (SO-101: the servo bridge, robot profile real.bridge)"),
        DeclareLaunchArgument("use_rviz", default_value="true"),
        DeclareLaunchArgument("hw_port", default_value="", description="hardware:=real: the servo bus device "
                              "(default: the robot profile's real.port, e.g. /dev/ttyACM0)"),
        DeclareLaunchArgument("hw_calibration", default_value="", description="hardware:=real: the servo calibration "
                              "file (default: the robot profile's real.calibration)"),
        DeclareLaunchArgument("hw_mock", default_value="false",
                              description="hardware:=real with SIMULATED servos (the bridge's mock bus): a dry run"),
        DeclareLaunchArgument("perception", default_value="true", description="start the YOLO-seg detector"),
        DeclareLaunchArgument("manipulation", default_value="", description="start the /pick_place action server (default: on, except hardware:=mock)"),
        DeclareLaunchArgument("scene", default_value="config/scenes/basic.yaml",
                              description="scene file (repo-relative): table, target object, goal, obstacles"),
        DeclareLaunchArgument("model", default_value="",
                              description="YOLO-seg weights (default: the scene's target.detector_model)"),
        DeclareLaunchArgument("obstacle_source", default_value="known", choices=["known", "perceived"],
                              description="known: /scene/known_obstacles boxes (5A); perceived: depth -> Octomap (5B)"),
        DeclareLaunchArgument("map_mode", default_value="continuous", choices=["continuous", "freeze_per_task"],
                              description="obstacles update continuously, or per task from a snapshot + change monitor "
                                          "(also live: ros2 param set /pnp_pick_place map_mode ...)"),
        DeclareLaunchArgument("change_policy", default_value="refresh", choices=["warn", "refresh", "stop"],
                              description="freeze_per_task: what to do when the scene changed during a task"),
        DeclareLaunchArgument("role", default_value="all", choices=["all", "robot", "brain"],
                              description="all: one computer; robot: the arm's side (Jetson); brain: the PC's side"),
        DeclareLaunchArgument("workcell", default_value="",
                              description="the workcell page + RViz picking (default: with manipulation)"),
        DeclareLaunchArgument("hand_detector", default_value="true",
                              description="pause a running arm move while a hand is in the workspace (fixed camera)"),
        DeclareLaunchArgument("bag_buffer", default_value="false",
                              description="true: keep the last seconds of every sensor in memory (rosbag2 snapshot "
                                          "mode) and save them when a pick fails (config/recording.yaml)"),
        DeclareLaunchArgument("servo_error", default_value="",
                              description="sim: joints behave like real servos, a config/servo_error.yaml profile "
                                          "(typical | worn); empty / none = perfect joints"),
        DeclareLaunchArgument("reality", default_value="",
                              description="sim sensor noise profile (config/sim_reality.yaml): ideal | realistic | stress; "
                                          "default: its 'active'. Camera placement errors: tools/set_reality.py"),
        DeclareLaunchArgument("pose_estimation", default_value="auto",
                              description="true: 6-D pose of the target (FoundationPose, needs the pnp:pose image); "
                                          "auto: the scene's perception.pose_estimation"),
        DeclareLaunchArgument("sensor_view", default_value="false",
                              description="true: colored point cloud + measured scene model, RViz 'sensor view' layout"),
        OpaqueFunction(function=_setup),
    ])
