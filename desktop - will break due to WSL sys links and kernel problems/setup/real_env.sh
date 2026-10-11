# The REAL SO-101 (the Jetson robot role) driven from the PC: source this in EVERY PC process that commands it
# (the brain launch, the calibration tools; data/realpy.sh does). The same limits the Jetson's robot container sets
# (jetson/docker-compose.robot.yml) -- RobotInterface reads them in the process that SENDS the goals, i.e. here.
# Without it a PC tool moved the real arm with the sim's defaults (2026-10-11: no 15 % cap, no rest-pose nudge).
export PNP_RMW=rmw_cyclonedds_cpp          # services across the PC <-> Jetson link need Cyclone on both ends
export PNP_MAX_VELOCITY_SCALING=0.15       # every MoveIt move capped at 15 %
export PNP_UNSTICK_INSIDE=0.05             # a joint past its limit is nudged ~3 deg inside (gravity sag)
export PNP_UNSTICK_MAX=0.30                # ... from up to 0.30 rad past: the folded rest reads 0.18 rad past shoulder_lift
export PNP_GOAL_TOLERANCE_RAD=0.07
export PNP_SCENE="${PNP_SCENE:-config/scenes/so101_plate_tags.yaml}"
source /repo/setup/ros_env.sh
