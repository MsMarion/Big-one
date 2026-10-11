# Repeat "turn the cracker box to a face" from a reset scene and count the outcomes (sim, FR3 workcell).
#   powershell -File data\flip_bench.ps1 -Rest side_x+ -N 4
param([string]$Rest = "side_x+", [int]$N = 4, [double]$X = 0.45, [double]$Y = -0.25, [switch]$Probe)
$repo = Split-Path $PSScriptRoot -Parent
$ok = 0
for ($i = 1; $i -le $N; $i++) {
    foreach ($try in 1..2) {      # the sim's answer can get lost on the bridge: once more (a repeated id isn't re-run)
        wsl -d Ubuntu-24.04 -u root --exec docker -H unix:///var/run/docker-pnp.sock exec pnp_dev bash -c "source /opt/ros/jazzy/setup.bash; source /ws/install/setup.bash; python3 /repo/tools/sim_client.py reset_target > /dev/null 2>&1"
        if ($LASTEXITCODE -eq 0) { break }
    }
    Start-Sleep 3
    $probe = if ($Probe) { Start-Job { wsl -d Ubuntu-24.04 -u root --exec docker -H unix:///var/run/docker-pnp.sock exec pnp_dev bash /repo/data/rospy.sh /repo/data/grip_slip_probe.py cracker_box 400 2>&1 } } else { $null }
    $out = & "$repo\data\workcell_face_pick.ps1" -Class cracker_box -X $X -Y $Y -YawDeg 0 -Rest $Rest
    $res = ($out | Select-String "^result:").Line
    $slip = $null
    if ($probe) { Wait-Job $probe -Timeout 30 | Out-Null; Stop-Job $probe -ErrorAction SilentlyContinue
                  $slip = (Receive-Job $probe | Select-String "^RELEASE" | Select-Object -First 1).Line }
    if ($slip) { "   in the hand at release (since the lift): $($slip.Substring(15))" }
    Start-Sleep 8      # the node rescans after the task
    $truth = wsl -d Ubuntu-24.04 -u root --exec docker -H unix:///var/run/docker-pnp.sock exec pnp_dev bash -c "source /opt/ros/jazzy/setup.bash; source /ws/install/setup.bash; python3 /repo/tools/sim_client.py state" | ConvertFrom-Json
    $w = curl.exe -s -m 10 http://localhost:8088/api/world | ConvertFrom-Json
    $b = $w.objects | Where-Object { $_.class -eq "cracker_box" } | Select-Object -First 1
    $face = if ($b) { "$($b.rest) (h $($b.measured_height))" } else { "not seen" }
    $xy = $truth.objects.cracker_box
    if ($res -match "SUCCESS") { $ok++ }
    "trial ${i}: $res | truth ($([math]::Round($xy[0],3)), $([math]::Round($xy[1],3))) | seen $face"
}
"$Rest : $ok / $N succeeded"
