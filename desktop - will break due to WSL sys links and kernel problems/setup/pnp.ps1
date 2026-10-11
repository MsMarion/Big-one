# Look inside the running project from a Windows terminal (PowerShell). docs/learning/28_how_it_runs.md
#
#   .\setup\pnp.ps1 status              what runs: Isaac, the bridges, the container, the ROS nodes
#   .\setup\pnp.ps1 shell               a bash shell INSIDE the container (ROS sourced: ros2 topic list, ...)
#   .\setup\pnp.ps1 run "<command>"     one command inside the container, e.g. "ros2 topic hz /joint_states"
#   .\setup\pnp.ps1 logs [ros|sim|zenoh|zenoh-win|camera]   follow a log live (Ctrl+C stops watching, not the program)
#   .\setup\pnp.ps1 py <script> [args]  a Python script inside the container with ROS sourced (data/rospy.sh)
#
# Starting / restarting things is setup\stack.ps1 (Ensure-Stack); this script only looks and runs commands.
param([Parameter(Position = 0)][string]$What = "status", [Parameter(Position = 1, ValueFromRemainingArguments = $true)][string[]]$Rest)
$Repo = Split-Path -Parent $PSScriptRoot
$Distro = "Ubuntu-24.04"

function In-Container([string]$cmd) {
    # docker/run_dev.sh: starts the container if it isn't running, then `docker exec pnp_dev bash -ic "<cmd>"`
    # (interactive bash: ROS 2 + our workspace are sourced). As root: the project's Docker engine socket is root's.
    wsl -d $Distro -u root --exec bash /root/pnp_repo/docker/run_dev.sh $cmd
}

switch ($What) {
    "status" {
        Write-Host "== Windows ==" -ForegroundColor Cyan
        $kit = Get-Process kit -ErrorAction SilentlyContinue
        Write-Host ("Isaac Sim (kit):        " + ($(if ($kit) { "running (pid $($kit.Id -join ', '))" } else { "NOT running" })))
        $zb = Get-Process zenoh-bridge-ros2dds -ErrorAction SilentlyContinue
        Write-Host ("zenoh bridge (Windows): " + ($(if ($zb) { "running (pid $($zb.Id -join ', '))" } else { "NOT running" })))
        $sim = Get-ChildItem (Join-Path $Repo "data\logs\run_sim_*.log") -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1
        if ($sim) { Write-Host "newest sim log:         $($sim.FullName)" }
        Write-Host "== WSL / container ==" -ForegroundColor Cyan
        wsl -d $Distro -u root --exec docker -H unix:///var/run/docker-pnp.sock ps --format "container {{.Names}}: {{.Status}} ({{.Image}})"
        In-Container "echo; echo 'ros2 launch:'; pgrep -fa 'ros2 launch pnp_bringup [b]ringup' | cut -c1-160 || echo '  NOT running'; echo; echo 'ROS nodes:'; ros2 node list 2>/dev/null | grep -v -E 'transform_listener|_private_|^/moveit_[0-9]' | sed 's/^/  /'"
    }
    "shell" { wsl -d $Distro -u root --exec bash /root/pnp_repo/docker/run_dev.sh }
    "run" { In-Container ($Rest -join " ") }
    "py" { In-Container ("bash /repo/data/rospy.sh /repo/" + ($Rest -join " ")) }
    "logs" {
        $which = if ($Rest) { $Rest[0] } else { "ros" }
        switch ($which) {
            "ros" { In-Container "tail -n 40 -F /data/logs_bringup.log" }         # every ROS node's screen output
            "camera" { In-Container "tail -n 40 -F /data/camera_keepalive.log" }
            "zenoh" { In-Container "tail -n 40 -F /tmp/zenoh.log" }                # the container's zenoh router
            "zenoh-win" { Get-Content (Join-Path $Repo "data\logs\zenoh_windows.log") -Tail 40 -Wait }
            "sim" {
                $sim = Get-ChildItem (Join-Path $Repo "data\logs\run_sim_*.log") | Sort-Object LastWriteTime | Select-Object -Last 1
                Write-Host "following $($sim.FullName)"
                Get-Content $sim.FullName -Tail 40 -Wait
            }
            default { Write-Host "logs: ros | sim | zenoh | zenoh-win | camera" }
        }
    }
    default { Get-Content $PSCommandPath -TotalCount 9 | ForEach-Object { $_ -replace "^# ?", "" } }
}
