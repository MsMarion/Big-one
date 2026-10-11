# Visual playground / handoff: Isaac Sim with viewports + RViz + an interactive menu window.
#
#   .\demo.ps1            # reuse a running GUI sim + stack if there is one, else start everything
#   .\demo.ps1 -Fresh     # rebuild the ROS packages and restart sim + stack (after code changes)
#   .\demo.ps1 -NoRviz    # skip RViz (the planning scene with the boxes is nice to see there, though)
#   .\demo.ps1 -Perceived # Phase 5B: the robot only SEES the boxes (depth camera -> Octomap; voxels in RViz)
#   .\demo.ps1 -Sensors   # every sensor stage in RViz (raw images, 3D clouds, measured scene); implies -Perceived
#   .\demo.ps1 -Sensors -Pose  # + 6-D pose of the can (FoundationPose: register once, then track; needs the pnp:pose image)
#   .\demo.ps1 -Sensors -Pose -Scene cracker_box   # another target object (config/scenes/cracker_box.yaml)
#   .\demo.ps1 -Sensors -Freeze  # start with the map frozen per task + change monitor (menu keys f / g / x switch live)
#   .\demo.ps1 -Scene so101_lego -ServoError typical   # sim joints err like real servos (config/servo_error.yaml)
#
# What you get:
#   * Isaac Sim window: main viewport + "D435i camera (sim)" viewport (what the detector sees)
#   * RViz (from the container, via WSLg): robot, planned trajectories, the MoveIt planning scene (table +
#     known obstacle boxes, inflated by the safety padding), YOLO detections + debug image
#   * "pnp playground" window: tools\demo_menu.py  (pick & place, randomize, obstacles on/off, loop, speed)
# Close the menu with q. The sim and stack keep running (for test.ps1 or another demo); stop the sim by
# closing its window, the stack with:  wsl -d Ubuntu-24.04 -u root --exec bash /root/pnp_repo/docker/run_dev.sh "bash /repo/setup/stop_ros.sh"
param([switch]$Fresh, [switch]$NoRviz, [switch]$Perceived, [switch]$Sensors, [switch]$Pose, [string]$Scene = "basic", [switch]$Freeze, [string]$Reality = "", [string]$ServoError = "")
. (Join-Path $PSScriptRoot "setup\stack.ps1")

Start-Bridge
$sim = Get-SimProcess
$headless = $sim -and ($sim.CommandLine -like "*--headless*")
if ($Fresh -or $headless) { Stop-Sim; $sim = $null }
if ($Fresh) { Build-Stack }
Ensure-Stack -Rviz:(-not $NoRviz) -ForceRestart:$Fresh -Perceived:($Perceived -or $Sensors) -SensorView:$Sensors -PoseEstimation:$Pose -Scene $Scene -Freeze:$Freeze -Reality $Reality -ServoError $ServoError      # GUI sim (a headless one was stopped above)

Write-Host "[demo] opening the playground menu in a new window"
$menu = "wsl -d Ubuntu-24.04 -u root --exec bash /root/pnp_repo/docker/run_dev.sh 'python3 /repo/tools/demo_menu.py'"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$host.UI.RawUI.WindowTitle = 'pnp playground'; $menu"
