# Tiered end-to-end tests. Keeps the sim + ROS stack running between calls (startup costs ~2 min; a smoke
# test ~1-2 min), and only restarts what's needed.
#
#   .\test.ps1                         # smoke: 5 trials, fast profile          -> after any change
#   .\test.ps1 -Tier regression        # 20 trials                              -> before a commit
#   .\test.ps1 -Tier gate -Obstacles   # 50 trials with random boxes            -> phase gates
#   .\test.ps1 -Rebuild                # rebuild ROS packages + restart the stack, keep the sim
#                                      #   (after editing ROS nodes, a .action/.srv, launch files, config)
#   .\test.ps1 -Fresh                  # like -Rebuild, and restart the sim (headless) too (after editing sim\*.py)
#   .\test.ps1 -Profile dev            # at the watchable speed instead of "fast"
#   .\test.ps1 -Obstacles -Perceived   # Phase 5B: boxes only via the depth camera (Octomap), not as known boxes
#   .\test.ps1 -Pose                   # + the 6-D pose node (FoundationPose); grasps come from it (YOLO = fallback)
#   .\test.ps1 -Pose -GraspSource ab   # alternate pose / detector grasps trial by trial (A/B comparison)
#   .\test.ps1 -Pose -Scene cracker_box   # another target object (config/scenes/<name>.yaml; sim + stack follow)
#   .\test.ps1 -Obstacles -Perceived -Freeze   # map frozen per task + change monitor (summary: scene_change_alarms)
#   .\test.ps1 -Pose -Scene cracker_box -GraspSource detector -WristRefine ab   # close look with the wrist D405 off/on
#
# If a GUI sim is already running (e.g. from demo.ps1) it is reused; otherwise a HEADLESS one is started
# (no viewports to render -> more GPU for the camera + YOLO).
# Results (WSL side): ~/pnp_data/runs/<run_id>/{episodes.jsonl,summary.json}  (= /data/runs in the container)
param(
    [ValidateSet("smoke", "regression", "gate")][string]$Tier = "smoke",
    [string]$Profile = "fast",
    [switch]$Obstacles,
    [switch]$Perceived,
    [switch]$Pose,
    [string]$Scene = "basic",
    [switch]$Freeze,
    [ValidateSet("", "ideal", "realistic", "stress")][string]$Reality = "",
    [ValidateSet("", "warn", "refresh", "stop")][string]$ChangePolicy = "",
    [ValidateSet("", "on", "off", "ab")][string]$WristRefine = "",
    [ValidateSet("", "auto", "pose", "detector", "ab")][string]$GraspSource = "",
    [switch]$Fresh,
    [switch]$Rebuild,
    [int]$Trials = 0
)
# (no $ErrorActionPreference = "Stop": in Windows PowerShell 5.1 any stderr line of wsl/docker would abort)
. (Join-Path $PSScriptRoot "setup\stack.ps1")

Start-Bridge
if ($Fresh) { Stop-Sim }
if ($Fresh -or $Rebuild) { Build-Stack }
# a GUI sim that's already running is reused; a new one is headless
Ensure-Stack -Headless -ForceRestart:($Fresh -or $Rebuild) -Perceived:$Perceived -PoseEstimation:$Pose -Scene $Scene -Freeze:$Freeze -Reality $Reality

$benchArgs = "--tier $Tier --profile $Profile"
if ($Obstacles) { $benchArgs += " --obstacles" }
if ($Perceived) { $benchArgs += " --perceived" }
if ($GraspSource) { $benchArgs += " --grasp-source $GraspSource" }
if ($ChangePolicy) { $benchArgs += " --change-policy $ChangePolicy" }
if ($WristRefine) { $benchArgs += " --wrist-refine $WristRefine" }
if ($Trials -gt 0) { $benchArgs += " --trials $Trials" }
Write-Host "[test] run_benchmark.py $benchArgs"
In-Container "python3 /repo/tools/run_benchmark.py $benchArgs"
