# Pick an object through the workcell API and set it down on another FACE (turning it over), then report.
#   powershell -File data\workcell_face_pick.ps1 -Class cracker_box -X 0.45 -Y -0.28 -YawDeg 0 -Rest side_x+
param([string]$Class = "cracker_box", [double]$X = 0.45, [double]$Y = -0.28, [double]$YawDeg = 0, [string]$Rest = "side_x+")
$base = "http://localhost:8088"
function Post($path, $obj) { ($obj | ConvertTo-Json -Compress) | Out-File -Encoding ascii "$env:TEMP\wc_body.json"; curl.exe -s -m 10 -X POST -H "Content-Type: application/json" --data-binary "@$env:TEMP\wc_body.json" "$base$path" }
Post "/api/scan" @{} | Out-Null
do { Start-Sleep 1; $w = curl.exe -s -m 10 "$base/api/world" | ConvertFrom-Json } while ($w.scan.busy)
$o = $w.objects | Where-Object { $_.class -eq $Class } | Select-Object -First 1
if (-not $o) { "no $Class found"; exit 1 }
"scan: $Class at ($($o.xy -join ', ')), resting $($o.rest)"
"pick: " + (Post "/api/pick" @{id = $o.id; xy = @($X, $Y); yaw_deg = $YawDeg; rest = $Rest})
do { Start-Sleep 2; $w = curl.exe -s -m 10 "$base/api/world" | ConvertFrom-Json } while ($w.task.busy)
$w.task.log | Select-String "^pick|^move |DETECT\]|LOOK\] refined|VERIFY_GRASP|VERIFY_PLACE\]|^[A-Z_]+:" | ForEach-Object { "  " + $_.Line }
"result: $($w.task.result)"
$o2 = $w.objects | Where-Object { $_.class -eq $Class } | Select-Object -First 1
if ($o2) { "after: $Class at ($($o2.xy -join ', ')), resting $($o2.rest)" }
