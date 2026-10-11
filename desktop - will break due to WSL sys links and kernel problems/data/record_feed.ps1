# Save frames of a workcell camera feed while a task runs (to see what happened, e.g. when an object tips over).
#   powershell -File data\record_feed.ps1 -Feed overhead -Out data\rec_flip -Seconds 120
param([string]$Feed = "overhead", [string]$Out = "data\rec", [int]$Seconds = 120, [int]$Hz = 4)
New-Item -ItemType Directory -Force $Out | Out-Null
$end = (Get-Date).AddSeconds($Seconds); $k = 0
while ((Get-Date) -lt $end) {
    $state = (curl.exe -s -m 2 http://localhost:8088/api/world | ConvertFrom-Json).task.state
    curl.exe -s -m 2 -o ("{0}\{1:D4}_{2}.jpg" -f $Out, $k, $state) "http://localhost:8088/api/feed/$Feed.jpg"
    $k++; Start-Sleep -Milliseconds ([int](1000 / $Hz))
}
"$k frames in $Out"
