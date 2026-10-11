# Shared helpers to start / check / stop the whole stack from Windows. Dot-source it:
#     . (Join-Path $PSScriptRoot "setup\stack.ps1")
# Used by test.ps1 (tiered headless tests) and demo.ps1 (visual handoff). Pieces:
#   zenoh bridge (Windows)  setup\zenoh_windows.ps1      single instance, auto-restarting
#   Isaac Sim               sim\run_sim.py via run_isaac.ps1 (process: kit.exe)
#   ROS 2 stack             bringup.launch.py in the pnp_dev container (WSL Docker engine)
# Rule learned the hard way: whenever the sim (re)starts, restart the bringup too (topic_based_ros2_control
# and the zenoh routes don't recover cleanly from a sim restart).

$Repo = Split-Path $PSScriptRoot -Parent
$LogDir = Join-Path $Repo "data\logs"
New-Item -ItemType Directory -Force $LogDir | Out-Null
$env:WSL_UTF8 = 1

function In-Container([string]$cmd) {
    # Runs `cmd` in the pnp_dev container (started if needed) and returns its output lines.
    # Output is converted to plain strings, minus the harmless "bash -i" job-control warnings (no TTY).
    wsl -d Ubuntu-24.04 -u root --exec bash /root/pnp_repo/docker/run_dev.sh $cmd 2>&1 | ForEach-Object { "$_" } |
        Where-Object { $_ -notmatch "cannot set terminal process group|no job control in this shell" }
}

function Get-SimProcess {
    Get-CimInstance Win32_Process -Filter "Name='kit.exe'" | Where-Object { $_.CommandLine -like "*run_sim.py*" }
}

function Start-Bridge {
    $running = Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
        Where-Object { $_.CommandLine -like "*zenoh_windows.ps1*" }
    if (-not $running) {
        Write-Host "[stack] starting the Windows zenoh bridge"
        # relative path + working directory: Start-Process does not quote arguments, and the repo path has spaces
        Start-Process powershell -ArgumentList "-NoProfile", "-File", "setup\zenoh_windows.ps1" -WorkingDirectory $Repo -WindowStyle Hidden
        Start-Sleep 5
    }
}

function Restart-Bridge {
    # A stale zenoh route (seen after ROS stack restarts) can silently stop ONE direction of one topic, e.g. Isaac's
    # joint feedback: ros2_control then echoes commands as joint states and nothing moves. Restarting the Windows
    # bridge rebuilds every route (the container side reconnects by itself).
    Write-Host "[stack] restarting the Windows zenoh bridge"
    Get-CimInstance Win32_Process | Where-Object { ($_.Name -eq "powershell.exe" -and $_.CommandLine -like "*-File setup\zenoh_windows.ps1*") -or $_.Name -like "zenoh-bridge*" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Start-Sleep 2
    Start-Bridge
    Start-Sleep 8
}

function Stop-Sim {
    Get-SimProcess | ForEach-Object { Write-Host "[stack] stopping sim (pid $($_.ProcessId))"; Stop-Process -Id $_.ProcessId -Force }
    Start-Sleep 2
}

function Resolve-Scene([string]$Scene) {
    # "cracker_box" -> "config/scenes/cracker_box.yaml"; a path is kept (repo-relative, forward slashes)
    if (-not $Scene) { return "config/scenes/basic.yaml" }
    if ($Scene -notmatch "\.yaml$") { return "config/scenes/$Scene.yaml" }
    return ($Scene -replace "\\", "/")
}

function Get-SimScene {
    # The scene the running sim was started with ("" = no sim; run_sim.py's default = basic)
    $sim = Get-SimProcess | Select-Object -First 1
    if (-not $sim) { return "" }
    if ($sim.CommandLine -match "--scene\s+(\S+)") { return ($Matches[1] -replace "\\", "/") }
    return "config/scenes/basic.yaml"
}

function Start-Sim([switch]$Headless, [string]$Scene = "config/scenes/basic.yaml", [int]$TimeoutS = 240) {
    $mode = if ($Headless) { "headless" } else { "GUI" }
    $log = Join-Path $LogDir ("run_sim_" + (Get-Date -Format yyyyMMdd_HHmmss) + ".log")
    Write-Host "[stack] starting Isaac Sim ($mode, $Scene), log: $log"
    $simArgs = @("-NoProfile", "-File", "run_isaac.ps1", "sim\run_sim.py", "--scene", $Scene)
    if ($Headless) { $simArgs += "--headless" }
    Start-Process powershell -ArgumentList $simArgs -WorkingDirectory $Repo -WindowStyle Minimized -RedirectStandardOutput $log -RedirectStandardError "$log.err"
    $deadline = (Get-Date).AddSeconds($TimeoutS)
    while ((Get-Date) -lt $deadline) {
        $hit = Select-String -Path $log -Pattern "\[sim\] running|Traceback" -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($hit) {
            if ($hit.Line -match "Traceback") { Get-Content $log -Tail 30; throw "sim failed to start (see $log)" }
            Write-Host "[stack] sim is running"; return
        }
        Start-Sleep 3
    }
    throw "sim did not report '[sim] running' within $TimeoutS s (see $log)"
}

function Test-Bringup {
    $n = In-Container "pgrep -fc 'pnp_manipulation/[p]ick_place_node' || true" | ForEach-Object { "$_" } |
        Where-Object { $_ -match '^\d+$' } | Select-Object -Last 1
    return ([int]"0$n" -gt 0)
}

function Get-BringupArgs {
    # The arguments the running bringup was launched with ("" if none is running)
    In-Container "pgrep -fa 'ros2 launch pnp_bringup [b]ringup.launch.py' || true" | ForEach-Object { "$_" } |
        Where-Object { $_ -match "bringup.launch.py" } | Select-Object -First 1
}

function Build-Stack {
    Write-Host "[stack] building our ROS packages"
    In-Container "pnp_build --packages-select pnp_interfaces pnp_manipulation pnp_bringup pnp_perception 2>&1 | tail -2" |
        Select-String "Summary|failed|Failed" | ForEach-Object { Write-Host "  $($_.Line)" }
}

function Start-Bringup([switch]$Rviz, [string]$LaunchArgs = "", [int]$TimeoutS = 120) {
    $r = if ($Rviz) { "true" } else { "false" }
    Write-Host "[stack] (re)starting the ROS 2 stack (rviz=$r $LaunchArgs)"
    # camera routes stay alive across stack restarts (setup/camera_keepalive.py: suspected cause of camera stalls)
    # (anchored pattern: an unanchored one matches this very command line and never starts it)
    In-Container "pgrep -f '^python3 /repo/setup/camera_keepalive.py' > /dev/null || nohup python3 /repo/setup/camera_keepalive.py > /data/camera_keepalive.log 2>&1 &" | Out-Null
    In-Container "bash /repo/setup/stop_ros.sh" | Out-Null
    In-Container "nohup ros2 launch pnp_bringup bringup.launch.py use_rviz:=$r $LaunchArgs > /data/logs_bringup.log 2>&1 &" | Out-Null
    $deadline = (Get-Date).AddSeconds($TimeoutS)
    while ((Get-Date) -lt $deadline) {
        Start-Sleep 5
        # RViz is only a viewer: a crash of it (seen once: a glibc mutex assertion in its startup) doesn't fail bringup
        $ready = In-Container "grep -cE 'detector ready|action server ready|perception NOT started' /data/logs_bringup.log; grep -E 'Traceback|process has died|Failed to load sensor' /data/logs_bringup.log | grep -v rviz2 | head -1"
        if ($ready -match "process has died|Traceback|Failed to load sensor") { $ready | ForEach-Object { Write-Host "  $_" }; throw "bringup failed (container: /data/logs_bringup.log)" }
        $n = $ready | ForEach-Object { "$_" } | Where-Object { $_ -match '^\d+$' } | Select-Object -First 1   # skip bash -i noise
        if ([int]"0$n" -ge 2) {
            Write-Host "[stack] ROS 2 stack ready"
            # the restart may have left a stale bridge route (no joint feedback): check, fix by restarting the bridge
            if ((Get-SimProcess) -and -not (Test-Link)) { Restart-Bridge; [void](Test-Link) }
            return
        }
    }
    throw "bringup not ready within $TimeoutS s (container: /data/logs_bringup.log)"
}

function Test-Link {
    # Command round trip container -> Isaac -> container (tools/link_check.py moves the gripper a little).
    $out = In-Container "python3 /repo/tools/link_check.py"
    $line = $out | Where-Object { $_ -match "link check:" } | Select-Object -Last 1
    Write-Host "[stack] $line"
    return ($line -match "OK")
}

function Test-Camera {
    # Camera frames reaching the container (color + depth + camera_info >= 5 Hz, tools/camera_check.py). Twice on
    # the sim kept running physics but stopped rendering (window dark, no camera) right after a ROS stack
    # restart; the gripper link check alone does not notice that.
    $out = In-Container "python3 /repo/tools/camera_check.py --seconds 3 > /dev/null 2>&1; echo CAMERA_EXIT=`$?"
    $ok = [bool]($out -match "CAMERA_EXIT=0")
    Write-Host "[stack] camera check: $(if ($ok) { 'OK (frames arriving)' } else { 'FAIL (no camera frames)' })"
    return $ok
}

function Set-Reality([string]$Name) {
    # Sensor realism (config/sim_reality.yaml): noise in the sensor model + camera placement errors in the sim, live
    if (-not $Name) { return }
    In-Container "python3 /repo/tools/set_reality.py $Name" | Where-Object { $_ -match "sensor noise|true vs believed" } |
        ForEach-Object { Write-Host "[stack] $_" }
}

function Ensure-Stack([switch]$Headless, [switch]$Rviz, [switch]$ForceRestart, [switch]$Perceived, [switch]$SensorView,
                      [switch]$PoseEstimation, [string]$Scene = "", [switch]$Freeze, [string]$Reality = "",
                      [string]$ServoError = "", [switch]$BagBuffer) {
    # Sim + stack up (in the right mode and SCENE) AND commands reaching the robot. One automatic full restart if
    # the link is broken. The sim and the ROS stack must run the same scene (it decides the target object).
    # -Freeze only sets the START value of map_mode; it can be switched live (menu key f, ros2 param set), so a
    # running stack is not restarted just for it.
    $Scene = Resolve-Scene $Scene
    # built item by item: in an @(...) literal an $(if ...) that yields nothing swallowed the items after it
    # (servo_error:= never reached the launch, 2026-10-11)
    $parts = New-Object System.Collections.Generic.List[string]
    $parts.Add("scene:=$Scene")
    if ($Perceived) { $parts.Add("obstacle_source:=perceived") }
    if ($SensorView) { $parts.Add("sensor_view:=true") }
    if ($PoseEstimation) { $parts.Add("pose_estimation:=true") }
    if ($Freeze) { $parts.Add("map_mode:=freeze_per_task") }
    if ($ServoError) { $parts.Add("servo_error:=$ServoError") }
    if ($BagBuffer) { $parts.Add("bag_buffer:=true") }      # failure ring buffer (config/recording.yaml)
    $launchArgs = $parts -join " "
    $simScene = Get-SimScene
    if ($simScene -and $simScene -ne $Scene) {
        Write-Host "[stack] the running sim has scene $simScene, need $Scene"
        Stop-Sim
    }
    $simStarted = $false
    if (-not (Get-SimProcess)) {
        Start-Sim -Headless:$Headless -Scene $Scene; $simStarted = $true
        # a NEW Isaac process: the bridge's old routes don't carry its /clock, and with sim time the controllers
        # can't even activate ("Switch controller timed out", seen on every scene switch) -> fresh routes first
        Restart-Bridge
    }
    $running = Get-BringupArgs
    $runningScene = if ($running -match "scene:=(\S+)") { $Matches[1] } else { "config/scenes/basic.yaml" }
    $runningServo = if ($running -match "servo_error:=(\S+)") { $Matches[1] } else { "" }
    $modeMismatch = $running -and ((($running -match "obstacle_source:=perceived") -ne [bool]$Perceived) -or
                                   (($running -match "sensor_view:=true") -ne [bool]$SensorView) -or
                                   (($running -match "pose_estimation:=true") -ne [bool]$PoseEstimation) -or
                                   ($runningScene -ne $Scene) -or ($runningServo -ne $ServoError) -or
                                   (($running -match "bag_buffer:=true") -ne [bool]$BagBuffer))
    if ($simStarted -or $ForceRestart -or $modeMismatch -or -not (Test-Bringup)) { Start-Bringup -Rviz:$Rviz -LaunchArgs $launchArgs }
    $mode = if ($Freeze) { "freeze_per_task" } else { "continuous" }
    In-Container "ros2 param set /pnp_pick_place map_mode $mode" | Out-Null      # running stack: switch live
    if ((Test-Link) -and (Test-Camera)) { Set-Reality $Reality; return }
    Write-Host "[stack] commands or camera frames don't get through: restarting sim + stack"
    Stop-Sim
    Start-Sim -Headless:$Headless -Scene $Scene
    Restart-Bridge
    Start-Bringup -Rviz:$Rviz -LaunchArgs $launchArgs
    if (-not ((Test-Link) -and (Test-Camera))) { throw "sim link still broken after a full restart (see data\logs\zenoh_windows.log)" }
    Set-Reality $Reality
}

function Invoke-Sim([string]$cmd) {
    # One sim command (randomize_scene, randomize_obstacles, clear_obstacles, state, ...), see sim/sim_services.py
    In-Container "python3 /repo/tools/sim_client.py $cmd" | Select-Object -Last 1
}
