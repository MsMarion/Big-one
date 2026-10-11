# Pass a RealSense plugged into THIS PC through to WSL (and from there into the pnp container).
#
#   .\setup\usb_camera.ps1            # find the camera, attach it to WSL (bind once as admin, see below)
#   .\setup\usb_camera.ps1 -Detach    # give it back to Windows (e.g. for the RealSense Viewer)
#
# Why: the ROS driver (realsense2_camera) runs in the Linux container, but USB devices belong to Windows.
# usbipd-win shares a USB device with WSL over a virtual USB/IP link. The ROS-packaged librealsense uses the
# userspace RSUSB backend (libusb), so no special WSL kernel modules are needed.
#
# One-time setup (yours to do, needs admin):
#   winget install --exact dorssel.usbipd-win
#   usbipd bind --busid <BUSID>            # in an ADMIN terminal; this script prints the BUSID
# The container must be created with USB access (once):
#   wsl -d Ubuntu-24.04 -u root --exec bash -c "docker -H unix:///var/run/docker-pnp.sock rm -f pnp_dev; PNP_USB=1 bash /root/pnp_repo/docker/run_dev.sh 'echo ok'"
# Then: bringup.launch.py sim:=false camera:=local   (docs/learning/08_real_sensors.md)
#
# Expect USB/IP to be the weak spot: if frames drop, lower the profile in config/realsense_d435i.yaml
# (e.g. 424x240x15) or use the Jetson path instead.
param([switch]$Detach)

if (-not (Get-Command usbipd -ErrorAction SilentlyContinue)) {
    Write-Host "usbipd-win is not installed. Install it (admin):  winget install --exact dorssel.usbipd-win"
    exit 1
}
$line = usbipd list | Where-Object { $_ -match "8086:" -and $_ -match "RealSense|Depth|8086:0b3a|8086:0b07" } | Select-Object -First 1
if (-not $line) {
    Write-Host "No Intel RealSense found in 'usbipd list' (VID 8086). Is it plugged into a USB 3 port?"
    usbipd list
    exit 1
}
$busid = ($line -split "\s+")[0]
Write-Host "RealSense: $line"
if ($Detach) { usbipd detach --busid $busid; exit $LASTEXITCODE }
if ($line -match "Not shared") {
    Write-Host "Not shared with WSL yet. Once, in an ADMIN terminal:  usbipd bind --busid $busid"
    exit 1
}
usbipd attach --wsl --busid $busid
if ($LASTEXITCODE -eq 0) {
    Write-Host "Attached to WSL. Check from the container:  lsusb | grep -i intel   (needs PNP_USB=1 container)"
}
exit $LASTEXITCODE
