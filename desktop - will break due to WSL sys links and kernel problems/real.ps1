# The REAL robot cell (no Isaac Sim): ROS stack in the container, real camera + real arm. docs/learning/17_so101.md
#
#   .\real.ps1 -Scene so101_lego                  # real D435i + real SO-101 (servo bridge), torque OFF at start
#   .\real.ps1 -Scene so101_lego -DryRun          # the same configuration with SIMULATED servos (no arm needed)
#   .\real.ps1 -Scene so101_lego -NoCamera        # arm only (mirror check, motion_check.py); no perception
#   .\real.ps1 -Stop                              # stop the stack (servo torque goes OFF)
#
# Before the first run: the checklist in docs/learning/17_so101.md (USB into WSL with usbipd, servo calibration,
# mirror check). The arm starts LIMP: enable the torque yourself when you're ready (the script prints how), and keep
# the power plug in reach.
param([string]$Scene = "so101_lego", [switch]$DryRun, [switch]$NoCamera, [switch]$NoRviz, [switch]$Stop)
. (Join-Path $PSScriptRoot "setup\stack.ps1")

if ($Stop) { In-Container "bash /repo/setup/stop_ros.sh" | Out-Null; Write-Host "[real] stack stopped"; return }
if (Get-SimProcess) { Write-Host "[real] an Isaac Sim is running: stopping it (the real cell doesn't use it)"; Stop-Sim }
$scenePath = Resolve-Scene $Scene
$cam = if ($NoCamera) { "camera:=none perception:=false" } else { "camera:=local" }
$mock = if ($DryRun) { "hw_mock:=true" } else { "" }
$args_ = "sim:=false scene:=$scenePath hardware:=real $cam $mock obstacle_source:=perceived manipulation:=true"
Build-Stack
try { Start-Bringup -Rviz:(-not $NoRviz) -LaunchArgs $args_ -TimeoutS 120 } catch { Write-Host "[real] $_" }
In-Container "grep -E 'SO-101 bridge|servos not answering|cannot open|calibration|ERROR' /data/logs_bringup.log | head -8" |
    ForEach-Object { Write-Host "  $_" }
Write-Host ""
Write-Host "[real] next steps (in the container: wsl -d Ubuntu-24.04 -u root --exec bash /root/pnp_repo/docker/run_dev.sh):"
Write-Host "  1. mirror check: torque is OFF; move the arm by hand, RViz must follow the same way"
Write-Host "  2. torque on:    ros2 service call /so101_bridge/enable std_srvs/srv/SetBool '{data: true}'"
Write-Host "  3. first motion: python3 /repo/tools/motion_check.py --scaling 0.1      (hand on the power plug)"
Write-Host "  4. pick & place: python3 /repo/tools/demo_menu.py   (or the /pick_place action)"
