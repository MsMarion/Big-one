. (Join-Path (Split-Path $PSScriptRoot -Parent) "setup\stack.ps1")
Ensure-Stack -Rviz -Perceived -SensorView -PoseEstimation -Scene basic
