# Rebuild our ROS packages and (re)start the stack on a workcell scene (fr3_workcell | so101_workcell).
param([string]$Scene = "fr3_workcell")
. (Join-Path (Split-Path $PSScriptRoot -Parent) "setup\stack.ps1")
Build-Stack
Ensure-Stack -Rviz -ForceRestart -Perceived -SensorView -PoseEstimation -Scene $Scene
