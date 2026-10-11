#!/usr/bin/env bash
# Re-check / redo the ARM-side calibrations on the Jetson, in order (docs/learning/23_mounted_workcell_plan.md section 3).
# The arm MOVES: work area clear (only the board on it), your hand on the servo power plug.
#   bash ~/pnp/jetson/recalibrate.sh [--board hand_board|table_board] [--write-offsets]
# 1. torque on, arm to the look pose (Pilz PTP, <= 15 %)
# 2. arm accuracy check: ~6-14 views of the board -> how consistent the arm + wrist camera are (the spread, mm)
# 3. joint-offset fit on those views: REPORTED; written only with --write-offsets (then the stack restarts: the arm is
#    folded to rest first, torque off during the restart)
# 4. D405 hand-eye (calibrate.py wrist --real), written with --force: the real arm's repeatability (~1 mm backlash,
#    gravity sag) keeps it above the sim-tuned 1 px gate
# 5. one look: the desk offset / tilt it reports should be small (a few mm, < 2 deg)
# Results stay on the Jetson; commit them with:  bash jetson/sync_to_jetson.sh commit "recalibration" (on the PC)
set -euo pipefail
BOARD=table_board WRITE=0
while [ $# -gt 0 ]; do
  case "$1" in --board) BOARD="$2"; shift ;; --write-offsets) WRITE=1 ;; *) echo "unknown $1"; exit 2 ;; esac
  shift
done
C="docker exec jetson-robot-1 bash -c"
ENV="source /repo/setup/ros_env.sh; source /ws/install/setup.bash; cd /repo"
step() { echo; echo "=== $*"; }

step "1. torque on, look pose"
$C "$ENV; python3 tools/move_to.py 0 0 0 80 0 --enable" | grep -E "torque|nudg|move:"

step "2. arm accuracy check ($BOARD)"
$C "$ENV; printf '\n\n\n' | python3 -u tools/arm_accuracy_check.py --real --poses 10 --board $BOARD" \
  | grep -v -E "RTPS|nudged" | tail -8 || true     # its checks FAIL often (exit 1): keep going, the fit decides
SAMPLES=$($C "ls -t /repo/data/calibration/arm_accuracy_*.json | head -1")

step "3. joint-offset fit on $SAMPLES"
$C "$ENV; python3 tools/fit_joint_offsets.py $SAMPLES --reference free" | grep -v RTPS | tail -8
if [ "$WRITE" = 1 ]; then
  $C "$ENV; python3 tools/fit_joint_offsets.py $SAMPLES --reference free --write" | tail -1
  $C "$ENV; python3 tools/move_to.py -5 -95 92 45 0 --off" | grep -E "move:|torque"
  (cd ~/pnp && docker compose -f jetson/docker-compose.robot.yml up -d --force-recreate robot >/dev/null 2>&1)
  sleep 60
  $C "$ENV; python3 tools/move_to.py 0 0 0 80 0 --enable" | grep -E "torque|nudg|move:"
fi

step "4. D405 hand-eye ($BOARD)"
$C "$ENV; printf '\n\n\n' | python3 -u tools/calibrate.py wrist --real --board $BOARD" | grep -v -E "RTPS|nudged" | tail -10
RUN=$($C "ls -td /repo/data/calibration/d405_* | head -1")
$C "$ENV; python3 -u tools/calibrate.py wrist --real --resolve ${RUN#/repo/} --force" | grep -v RTPS | tail -3

step "5. one look"
$C "$ENV; python3 tools/move_to.py 0 0 0 80 0" | grep -E "move:"
$C "$ENV; ros2 service call /wrist_map/look std_srvs/srv/Trigger" | grep -o "message=.*"
echo; echo "done: commit the results from the PC (bash jetson/sync_to_jetson.sh commit \"recalibration\" && ... pull)"
