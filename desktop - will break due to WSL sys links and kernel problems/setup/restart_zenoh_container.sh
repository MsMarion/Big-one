#!/usr/bin/env bash
# Restart the container-side zenoh router + its supervisor (ROS nodes keep running).
#   bash ~/pnp_repo/docker/run_dev.sh "bash /repo/setup/restart_zenoh_container.sh"
# Anchored patterns: the container's MAIN process is "bash -c bash /repo/setup/zenoh_container.sh & exec sleep
# infinity"; an unanchored match on "setup/zenoh_container.sh" kills it too, and with it the whole container.
pkill -f "^bash /repo/setup/zenoh_container.sh"; pkill -f "^zenoh-bridge-ros2dds router"; sleep 2
nohup bash /repo/setup/zenoh_container.sh > /dev/null 2>&1 &
sleep 1
echo "container zenoh router restarted (pid $(pgrep -f '^zenoh-bridge-ros2dds router'))"
