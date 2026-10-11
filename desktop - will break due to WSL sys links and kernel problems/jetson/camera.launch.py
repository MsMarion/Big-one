"""RealSense D435i driver on the Jetson, with the SAME parameters the PC uses (config/realsense_d435i.yaml).

Started by jetson/docker-compose.yml. Publishes /camera/camera/... raw topics locally (DDS on the Jetson) plus,
through image_transport, the .../compressed (JPEG) and .../compressedDepth (PNG) versions that the zenoh bridge
forwards to the PC (jetson/zenoh_jetson.json5).
"""

import yaml
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    params = yaml.safe_load(open("/config/realsense_d435i.yaml"))
    return LaunchDescription([
        Node(package="realsense2_camera", executable="realsense2_camera_node", name="camera", namespace="camera",
             output="screen", parameters=[params]),
    ])
