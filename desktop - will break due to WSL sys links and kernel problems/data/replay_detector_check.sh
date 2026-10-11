#!/usr/bin/env bash
# Replay check: does a recorded session drive perception offline? Plays a bag on its OWN ROS domain (77: nothing of
# the live stack on domain 42 can see it or interfere) with ONLY the detector running, and counts what it detects.
#   bash /repo/data/replay_detector_check.sh [/data/bags/<name>]      (in the pnp container)
set -o pipefail          # (not -u: the ROS setup scripts read unset variables)
BAG=${1:-/data/bags/test_session_20s}
export ROS_DOMAIN_ID=77
source /opt/ros/jazzy/setup.bash
source /ws/install/setup.bash
/opt/pnp_venv/bin/python3 /ws/install/pnp_perception/lib/pnp_perception/detector_node --ros-args \
  -p model:=/repo/models/lego_seg_v1.pt -p object:=lego_2x4 -p use_sim_time:=true > /tmp/replay_detector.log 2>&1 &
DET=$!
for _ in $(seq 1 60); do grep -q "detector ready" /tmp/replay_detector.log && break; sleep 0.5; done
echo "detector: $(grep -o 'detector ready.*' /tmp/replay_detector.log | cut -c1-110)"
# play only the sensor inputs: the detector's own output must come from the detector, not from the bag
ros2 bag play "$BAG" --clock --topics /camera/camera/color/image_raw /camera/camera/color/camera_info \
  /camera/camera/aligned_depth_to_color/image_raw /tf /tf_static > /tmp/replay_play.log 2>&1 &
PLAY=$!
python3 - <<'EOF'
import collections, time
import rclpy
from vision_msgs.msg import Detection3DArray
rclpy.init()
n = rclpy.create_node("replay_count")
msgs, labels = [0], collections.Counter()
def cb(m):
    msgs[0] += 1
    for d in m.detections:
        labels[d.results[0].hypothesis.class_id] += 1
n.create_subscription(Detection3DArray, "/perception/detections", cb, 10)
end = time.time() + 22
while time.time() < end:
    rclpy.spin_once(n, timeout_sec=0.1)
print(f"replayed: {msgs[0]} detection messages, objects seen: {dict(labels)}")
EOF
kill $PLAY $DET 2>/dev/null; wait 2>/dev/null
