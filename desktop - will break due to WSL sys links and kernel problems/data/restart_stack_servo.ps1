. (Join-Path (Split-Path $PSScriptRoot -Parent) "setup\stack.ps1")
Build-Stack
Ensure-Stack -Rviz -ForceRestart -Perceived -SensorView -PoseEstimation -Scene so101_lego -ServoError typical
