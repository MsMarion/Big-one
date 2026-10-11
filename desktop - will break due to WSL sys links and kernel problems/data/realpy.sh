#!/usr/bin/env bash
# python3 for the REAL arm: ROS + the real limits (setup/real_env.sh). data/rospy.sh is the sim's.
source /repo/setup/real_env.sh
exec python3 "$@"
