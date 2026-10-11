#!/usr/bin/env bash
# python3 with ROS sourced, via setup/ros_env.sh: PNP_RMW=rmw_cyclonedds_cpp for the real Jetson link (services across
# it need Cyclone on both ends); the default (Fast DDS) for the sim.
source /repo/setup/ros_env.sh
exec python3 "$@"
