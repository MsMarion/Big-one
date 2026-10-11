#!/usr/bin/env bash
# Dual-camera health: both cameras' streams, both TF chains, the close-look service.
source /opt/ros/jazzy/setup.bash
source /ws/install/setup.bash
hz() { printf "  %-58s " "$1"; timeout 6 ros2 topic hz "$1" --window 15 2>/dev/null | grep -m1 -o "average rate: [0-9.]*" || echo "NO DATA"; }
echo "[overhead D435i]"
hz /camera/camera/color/image_raw
hz /camera/camera/aligned_depth_to_color/image_raw
echo "[wrist D405]"
hz /wrist_camera/camera/color/image_raw
hz /wrist_camera/camera/aligned_depth_to_color/image_raw
BASE=$(cd /repo && python3 -c "from pnp_core.catalog import load_robot; print(load_robot()['frames']['base'])")
echo "[TF] (base: $BASE)"
for f in camera_color_optical_frame wrist_camera_color_optical_frame; do
  printf "  %s -> %-34s " "$BASE" "$f"
  timeout 6 ros2 run tf2_ros tf2_echo $BASE $f 2>/dev/null | grep -m1 "Translation" || echo "MISSING"
done
echo "[services / nodes]"
ros2 service list 2>/dev/null | grep -E "refine_pose|calibration/reload" | sed 's/^/  /'
ros2 node list 2>/dev/null | grep -E "wrist_refine|camera_tf|pick_place|detector|obstacle_cloud" | sed 's/^/  /'
