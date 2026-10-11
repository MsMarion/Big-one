# Run a script with Isaac Sim's Python, with the ROS 2 environment set up.
#   .\run_isaac.ps1 sim\run_sim.py --scene config\scenes\basic.yaml
#   .\run_isaac.ps1 sim\check_assets.py
. (Join-Path $PSScriptRoot "setup\windows_env.ps1")
$IsaacSim = if ($env:ISAAC_SIM_PATH) { $env:ISAAC_SIM_PATH } else { "C:\isaac-sim" }
Push-Location $PSScriptRoot
try {
    & (Join-Path $IsaacSim "python.bat") @args
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
