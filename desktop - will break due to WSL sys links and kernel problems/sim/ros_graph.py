"""OmniGraph action graphs that connect the Isaac Sim scene to ROS 2 (plan section 5a: interface contract).

What this teaches:
  - OmniGraph is Isaac's node-based dataflow. An "execution" graph runs its nodes when triggered;
    OnPlaybackTick fires once per rendered frame while the simulation is playing.
  - The ROS 2 bridge nodes (isaacsim.ros2.bridge.*) publish/subscribe real ROS 2 topics.
  - Topic names, frame ids and encodings are chosen to MATCH THE REAL DRIVERS (realsense2_camera,
    ros2_control), so the ROS side cannot tell sim from real. Names come from config/interfaces.yaml.

Graphs built here:
  /ROS_Robot   (every PHYSICS step, 120 Hz) clock, /joint_states, /joint_command -> drives, ground-truth TF
  /ROS_Camera  (every RENDERED frame, 30 Hz) color image, camera_info, aligned depth

The point cloud is NOT published from Isaac: at ~5 MB/frame it would swamp the Windows->container link.
The ROS side computes it from depth + camera_info when needed (depth_image_proc), like many real setups.
"""

from __future__ import annotations

import json

import omni.graph.core as og
import usdrt.Sdf
from isaacsim.core.utils.extensions import enable_extension

from common import load_yaml

enable_extension("isaacsim.ros2.bridge")


def _split(topic: str) -> tuple[str, str]:
    """'/camera/camera/color/image_raw' -> namespace '/camera/camera', name 'color/image_raw'."""
    parts = topic.strip("/").split("/")
    return "/" + "/".join(parts[:2]), "/".join(parts[2:])


def build_robot_graph(robot_prim: str, base_link_prim: str, gt_prims: list[str]) -> None:
    t = load_yaml("config/interfaces.yaml")["topics"]
    keys = og.Controller.Keys
    og.Controller.edit(
        # OnPhysicsStep-driven graphs must use the ON_DEMAND pipeline stage (they are ticked by the
        # physics callback, not by the per-frame graph evaluation) - otherwise they never run.
        {"graph_path": "/ROS_Robot", "evaluator_name": "execution",
         "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND},
        {
            keys.CREATE_NODES: [
                ("OnStep", "isaacsim.core.nodes.OnPhysicsStep"),
                ("SimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("PubClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ("PubJoints", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                ("SubJoints", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                ("ArtCtrl", "isaacsim.core.nodes.IsaacArticulationController"),
                ("PubGroundTruthTF", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
            ],
            keys.CONNECT: [
                ("OnStep.outputs:step", "PubClock.inputs:execIn"),
                ("OnStep.outputs:step", "PubJoints.inputs:execIn"),
                ("OnStep.outputs:step", "SubJoints.inputs:execIn"),
                ("OnStep.outputs:step", "ArtCtrl.inputs:execIn"),
                ("OnStep.outputs:step", "PubGroundTruthTF.inputs:execIn"),
                ("Context.outputs:context", "PubClock.inputs:context"),
                ("Context.outputs:context", "PubJoints.inputs:context"),
                ("Context.outputs:context", "SubJoints.inputs:context"),
                ("Context.outputs:context", "PubGroundTruthTF.inputs:context"),
                ("SimTime.outputs:simulationTime", "PubClock.inputs:timeStamp"),
                ("SimTime.outputs:simulationTime", "PubJoints.inputs:timeStamp"),
                ("SimTime.outputs:simulationTime", "PubGroundTruthTF.inputs:timeStamp"),
                # commands received on /joint_command drive the articulation's joint targets
                ("SubJoints.outputs:jointNames", "ArtCtrl.inputs:jointNames"),
                ("SubJoints.outputs:positionCommand", "ArtCtrl.inputs:positionCommand"),
                ("SubJoints.outputs:velocityCommand", "ArtCtrl.inputs:velocityCommand"),
                ("SubJoints.outputs:effortCommand", "ArtCtrl.inputs:effortCommand"),
            ],
            keys.SET_VALUES: [
                ("PubClock.inputs:topicName", t["clock"]["name"]),
                # Raw sim joint topics; ros2_control (topic_based_ros2_control) is the only consumer/producer.
                ("PubJoints.inputs:topicName", t["isaac_joint_states"]["name"]),
                ("PubJoints.inputs:targetPrim", [usdrt.Sdf.Path(robot_prim)]),
                ("SubJoints.inputs:topicName", t["isaac_joint_command"]["name"]),
                ("ArtCtrl.inputs:targetPrim", [usdrt.Sdf.Path(robot_prim)]),
                # Ground truth (sim only): TF from the robot base to each object, child frame = prim name.
                ("PubGroundTruthTF.inputs:topicName", "/tf"),
                ("PubGroundTruthTF.inputs:parentPrim", [usdrt.Sdf.Path(base_link_prim)]),
                ("PubGroundTruthTF.inputs:targetPrims", [usdrt.Sdf.Path(p) for p in gt_prims]),
            ],
        },
    )


# ROS "sensor data" QoS (best effort, keep last 5): a late image is worthless, so drop it rather than
# block. This also lets the zenoh bridge route images as droppable (see config/zenoh_bridge.json5).
SENSOR_QOS = json.dumps({"history": "keepLast", "depth": 5, "reliability": "bestEffort", "durability": "volatile",
                         "deadline": 0.0, "lifespan": 0.0, "liveliness": "systemDefault", "leaseDuration": 0.0})


def build_camera_graph(camera_prim: str, cam_cfg_path: str = "config/camera_d435i.yaml",
                       graph_path: str = "/ROS_Camera") -> None:
    """Color + aligned depth + camera_info of one camera. The D435i's topics come from config/interfaces.yaml, other
    cameras (the wrist D405) carry theirs in their own config; profile 'rgb' = no depth (an ordinary RGB camera)."""
    cam = load_yaml(cam_cfg_path)
    t = load_yaml("config/interfaces.yaml")["topics"]
    if "topics" in cam:
        t = {k: {"name": v} for k, v in cam["topics"].items()}
    # perfect images go to /sim_raw/...; the sensor model (container) publishes the standard topics with realistic noise
    raw = load_yaml("config/interfaces.yaml").get("sim_raw_prefix", "")
    t = {k: {"name": raw + v["name"]} for k, v in t.items()}
    frame = cam["frames"]["color_optical"]
    helpers = {
        # node name: (helper type, topic)
        "Color": ("rgb", t["color_image"]["name"]),
        "Depth": ("depth", t["aligned_depth"]["name"]),       # 32FC1 meters (real D435i: 16UC1 mm)
    }
    if cam.get("profile") == "rgb":
        helpers.pop("Depth")
    keys = og.Controller.Keys
    nodes = [
        ("OnTick", "omni.graph.action.OnPlaybackTick"),
        ("Context", "isaacsim.ros2.bridge.ROS2Context"),
        ("RenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
        ("Info", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
    ] + [(n, "isaacsim.ros2.bridge.ROS2CameraHelper") for n in helpers]
    connect = [
        ("OnTick.outputs:tick", "RenderProduct.inputs:execIn"),
        ("RenderProduct.outputs:execOut", "Info.inputs:execIn"),
        ("RenderProduct.outputs:renderProductPath", "Info.inputs:renderProductPath"),
        ("Context.outputs:context", "Info.inputs:context"),
    ]
    ns, info_name = _split(t["color_info"]["name"])
    values = [
        ("RenderProduct.inputs:cameraPrim", [usdrt.Sdf.Path(camera_prim)]),
        ("RenderProduct.inputs:width", cam["width"]),
        ("RenderProduct.inputs:height", cam["height"]),
        ("Info.inputs:nodeNamespace", ns),
        ("Info.inputs:topicName", info_name),
        ("Info.inputs:frameId", frame),
        ("Info.inputs:qosProfile", SENSOR_QOS),
    ]
    for node, (kind, topic) in helpers.items():
        ns, name = _split(topic)
        connect += [
            ("RenderProduct.outputs:execOut", f"{node}.inputs:execIn"),
            ("RenderProduct.outputs:renderProductPath", f"{node}.inputs:renderProductPath"),
            ("Context.outputs:context", f"{node}.inputs:context"),
        ]
        values += [
            (f"{node}.inputs:type", kind),
            (f"{node}.inputs:nodeNamespace", ns),
            (f"{node}.inputs:topicName", name),
            (f"{node}.inputs:frameId", frame),
            (f"{node}.inputs:qosProfile", SENSOR_QOS),
        ]
    og.Controller.edit({"graph_path": graph_path, "evaluator_name": "execution"},
                       {keys.CREATE_NODES: nodes, keys.CONNECT: connect, keys.SET_VALUES: values})
