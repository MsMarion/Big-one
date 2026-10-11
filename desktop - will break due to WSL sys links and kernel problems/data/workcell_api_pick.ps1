# Drive the workcell page's API like the page does: scan, choose an object by class, set a goal, pick, wait.
param([string]$Class = "cracker_box", [double]$X = 0.56, [double]$Y = 0.0, [double]$YawDeg = 0)
$base = "http://localhost:8088"
function Post($path, $obj) { ($obj | ConvertTo-Json -Compress) | Out-File -Encoding ascii "$env:TEMP\wc_body.json"; curl.exe -s -m 10 -X POST -H "Content-Type: application/json" --data-binary "@$env:TEMP\wc_body.json" "$base$path" }
Post "/api/scan" @{} | Out-Null
do { Start-Sleep 1; $w = curl.exe -s -m 10 "$base/api/world" | ConvertFrom-Json } while ($w.scan.busy)
$o = $w.objects | Where-Object { $_.class -eq $Class } | Select-Object -First 1
if (-not $o) { "no $Class found: $($w.objects | ForEach-Object { $_.class })"; exit 1 }
"scan: " + (($w.objects | ForEach-Object { "$($_.class) ($($_.xy -join ', '))" }) -join "; ")
Post "/api/select" @{id = $o.id} | Out-Null
Post "/api/goal" @{xy = @($X, $Y); yaw_deg = $YawDeg} | Out-Null
"pick: " + (Post "/api/pick" @{id = $o.id})
do { Start-Sleep 2; $w = curl.exe -s -m 10 "$base/api/world" | ConvertFrom-Json } while ($w.task.busy)
$w.task.log | Select-String "DETECT\]|LOOK\] refined|VERIFY_GRASP|VERIFY_PLACE\]" | ForEach-Object { "  " + $_.Line }
"result: $($w.task.result)"
do { Start-Sleep 1; $w = curl.exe -s -m 10 "$base/api/world" | ConvertFrom-Json } while ($w.scan.busy)
"after: " + (($w.objects | ForEach-Object { "$($_.class) ($($_.xy -join ', '))" }) -join "; ")
