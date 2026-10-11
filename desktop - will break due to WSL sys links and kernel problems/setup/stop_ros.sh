#!/usr/bin/env bash
# Stop every ROS process of the pnp stack inside the container (launch files, controllers, MoveIt,
# RViz, test tools). The zenoh bridge is left running.
#   bash ~/pnp_repo/docker/run_dev.sh "bash /repo/setup/stop_ros.sh"
# Patterns are anchored to the executables, so this script never matches (and kills) its own shell.
# Includes our own nodes from /ws/install, also when started with a venv python (detector: pnp_venv; 6-D pose:
# fp_venv; missing fp_venv here once left orphaned pose nodes holding ~7 GB GPU memory each).
RE='^(/usr/bin/python3 /opt/ros/jazzy/bin/ros2 (launch|run|topic|action|service)|/opt/ros/jazzy/lib/[a-z0-9_]+/(ros2_control_node|move_group|robot_state_publisher|static_transform_publisher|rviz2|spawner)|python3 /repo/tools/|(/usr/bin/python3 |/opt/pnp_venv/bin/python3 |/opt/fp_venv/bin/python3 )?/ws/install/|/opt/fp_venv/bin/python3? /repo/tools/)'
# + the failure ring buffer (bringup bag_buffer:=true): a stale one kept running after a restart and answered the
#   next snapshot too (2026-10-11). A session recording (tools/record_session.sh) is NOT stopped here.
RE="$RE|^/usr/bin/python3 /opt/ros/jazzy/bin/ros2 bag record --snapshot-mode"
# + the Jetson link's image decompressors (bringup camera:=jetson): they outlived each launch, 3 sets piled up
#   subscribed to the same stream and the PC got no images on Cyclone (2026-10-11); + our data/ helper scripts
RE="$RE|^/opt/ros/jazzy/lib/image_transport/republish|^python3 (-u )?/repo/data/"
count() { pgrep -fc "$RE" || true; }
echo "running before: $(count)"
pkill -INT -f "$RE"; sleep 3
pkill -KILL -f "$RE"; sleep 1
echo "running after:  $(count)"
# Fast DDS shared-memory segments of the killed processes (/dev/shm/fastrtps_*): a process killed while holding one's
# lock left the NEXT launch's first node (camera_tf_node) waiting on it forever, all threads in futex_wait, nothing
# published -- no camera TF (2026-10-11, 163 stale segments). The cleaner removes only segments nobody uses.
fastdds shm clean > /dev/null 2>&1 || true
