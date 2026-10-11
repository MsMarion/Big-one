# Environment for running Isaac Sim with its internal ROS 2 Jazzy libraries on Windows.
# Dot-source it in a PowerShell session:   . .\setup\windows_env.ps1
# (run_isaac.ps1 and setup\zenoh_windows.ps1 do this for you.)
#
# Why each variable:
#   ROS_DISTRO          -> tells isaacsim.ros2.bridge which bundled ROS libs to load (jazzy/humble)
#   RMW_IMPLEMENTATION  -> DDS vendor used by Isaac Sim
#   ROS_DOMAIN_ID       -> isolates our ROS graph; must match the container (setup/ros_env.sh)
#   PATH                -> the bridge's bundled ROS 2 DLLs must be loadable
#   (ROS_AUTOMATIC_DISCOVERY_RANGE stays at the default SUBNET: in LOCALHOST mode Isaac's Fast DDS and
#    the zenoh bridge's CycloneDDS do not discover each other on Windows - tested.)

$IsaacSim = if ($env:ISAAC_SIM_PATH) { $env:ISAAC_SIM_PATH } else { "C:\isaac-sim" }

$env:ROS_DISTRO = "jazzy"
$env:RMW_IMPLEMENTATION = "rmw_fastrtps_cpp"
if (-not $env:ROS_DOMAIN_ID) { $env:ROS_DOMAIN_ID = "42" }
Remove-Item Env:ROS_AUTOMATIC_DISCOVERY_RANGE, Env:FASTDDS_DEFAULT_PROFILES_FILE, Env:FASTRTPS_DEFAULT_PROFILES_FILE -ErrorAction SilentlyContinue

$bridgeLib = Join-Path $IsaacSim "exts\isaacsim.ros2.bridge\jazzy\lib"
if (-not ($env:PATH -split ";" | Where-Object { $_ -eq $bridgeLib })) {
    $env:PATH = "$env:PATH;$bridgeLib"
}

Write-Host "[env] ISAAC_SIM=$IsaacSim ROS_DISTRO=$env:ROS_DISTRO ROS_DOMAIN_ID=$env:ROS_DOMAIN_ID"
