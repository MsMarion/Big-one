# Windows side of the Windows <-> container ROS 2 link. Run it in its own terminal and leave it open
# while Isaac Sim runs:    .\setup\zenoh_windows.ps1
#
# How the link works:
#   Isaac Sim --(DDS, Windows loopback only)--> zenoh-bridge-ros2dds.exe (this script, zenoh CLIENT)
#     --(one TCP connection to 127.0.0.1:7447)--> zenoh-bridge in the pnp container (zenoh ROUTER)
#     --(DDS inside the container)--> ROS 2 nodes (MoveIt, perception, ...)
# Each side's DDS traffic stays local, so WSL firewall / multicast issues don't matter.
#
# Get the binary (one time): download zenoh-plugin-ros2dds-<ver>-x86_64-pc-windows-msvc-standalone.zip
# from https://github.com/eclipse-zenoh/zenoh-plugin-ros2dds/releases (same version as docker/Dockerfile)
# and unzip into tools_bin\zenoh-<ver>-win\.
param([string]$Version = "1.10.1", [string]$Router = "tcp/127.0.0.1:7447")

. (Join-Path $PSScriptRoot "windows_env.ps1")
$Repo = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $Repo "tools_bin\zenoh-$Version-win\zenoh-bridge-ros2dds.exe"
if (-not (Test-Path $exe)) { throw "zenoh bridge not found at $exe (see header of this script)" }
# Big socket buffers so ~1 MB camera images (many UDP fragments) are not dropped.
# (Same content as config/cyclonedds_bridge.xml, inlined: CycloneDDS file URIs break on paths with spaces.)
$env:CYCLONEDDS_URI = "<CycloneDDS><Domain id='any'><Internal><SocketReceiveBufferSize min='32MB'/><SocketSendBufferSize min='32MB'/></Internal></Domain></CycloneDDS>"

# Discovery range SUBNET (the ROS default) so the bridge's CycloneDDS and Isaac's Fast DDS find each
# other; LOCALHOST mode was tested and they don't. Windows may ask once to allow the exe through the
# firewall: local traffic works either way.
#
# --pub-max-frequency: camera images are ~1-1.2 MB each; at 30 Hz color+depth exceed what the bridge
# carries reliably, and frames get dropped at random. Downsampling to a steady 15 Hz is plenty for
# pick & place perception. camera_info is tiny and stays at full rate.
#
# Output goes to a LOG FILE, not this console: clicking/selecting text in a Windows console ("QuickEdit")
# pauses any program writing to it. A paused bridge misses its keep-alives, the container drops it, and
# every Isaac topic silently stops flowing (this happened - see docs/learning/00_setup.md).
#
# If the router connection drops (container restarted), the bridge reconnects by itself (tested). In
# client mode it exits if the router isn't reachable at startup, so keep relaunching it.
# Exactly one Windows bridge: two would forward every message twice (duplicate joint commands!).
# A named system-wide mutex guarantees only ONE copy of this script (and its relaunch loop) runs.
$mutex = New-Object System.Threading.Mutex($false, "Global\pnp_zenoh_windows_bridge")
if (-not $mutex.WaitOne(0)) {
    Write-Host "[zenoh] zenoh_windows.ps1 is already running in another window. Not starting a second one."
    return
}
Get-Process zenoh-bridge-ros2dds -ErrorAction SilentlyContinue | Stop-Process -Force   # stray exe from elsewhere
$logDir = Join-Path $Repo "data\logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$log = Join-Path $logDir "zenoh_windows.log"
Write-Host "[zenoh] Windows bridge -> $Router (domain $env:ROS_DOMAIN_ID). Log: $log"
Write-Host "[zenoh] Leave this window open; Ctrl+C to stop. Safe to click in this window."
while ($true) {
    # priorities + image downsampling: config/zenoh_bridge.json5 (shared with the container side)
    & $exe client --connect $Router --domain $env:ROS_DOMAIN_ID --no-multicast-scouting --ros-automatic-discovery-range SUBNET `
        -c (Join-Path $Repo "config\zenoh_bridge.json5") *>> $log
    Write-Host "[zenoh] $(Get-Date -Format HH:mm:ss) bridge exited (container not up?); retrying in 3 s"
    Start-Sleep 3
}
