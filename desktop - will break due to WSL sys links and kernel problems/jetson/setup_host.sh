#!/usr/bin/env bash
# One-time host setup of the Jetson (Orin Nano Super 8 GB, JetPack 6.2) for the ROBOT role. READ IT FIRST, then run
# it on the Jetson yourself (docs/learning/22_jetson_d405.md):
#   bash ~/pnp/jetson/setup_host.sh                 # check only: prints what's set and what isn't, changes nothing
#   sudo bash ~/pnp/jetson/setup_host.sh --apply    # groups, USB power, chrony, firewall rule for zenoh
# Separate opt-ins (they're yours to decide):
#   --maxn                 power mode MAXN SUPER (looked up by name in /etc/nvpmodel.conf; may ask to reboot)
#   --docker-root DIR      move Docker's data (images, the ~6 GB robot image) to DIR, e.g. /mnt/ssd/docker
#   --pc-ip IP             with ufw active: allow TCP 7447 only from this IP (else from anywhere on the LAN)
set -uo pipefail
APPLY=0 MAXN=0 DROOT="" PCIP=""
while [ $# -gt 0 ]; do
  case "$1" in
    --apply) APPLY=1 ;;
    --maxn) MAXN=1 ;;
    --docker-root) DROOT="$2"; shift ;;
    --pc-ip) PCIP="$2"; shift ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
  shift
done
USR="${SUDO_USER:-$USER}"
if [ "$APPLY$MAXN" != "00" ] || [ -n "$DROOT" ]; then
  [ "$(id -u)" = 0 ] || { echo "changes need sudo: sudo bash $0 $*"; exit 1; }
fi
ok()   { echo "  [ok]   $*"; }
todo() { echo "  [todo] $*"; }

echo "== system"
cat /etc/nv_tegra_release 2>/dev/null | head -1 || echo "  (not a Jetson?)"
echo "  $(lsb_release -ds 2>/dev/null), kernel $(uname -r), $(nproc) cores, $(free -g | awk '/Mem/{print $2}') GB RAM"
df -h / | tail -1 | awk '{print "  root fs: " $4 " free of " $2}'
lsblk -dno NAME,SIZE,MODEL | grep -v -E "^(loop|zram|mtdblock)" | sed 's/^/  disk: /'

echo "== power mode"
nvpmodel -q 2>/dev/null | head -2 | sed 's/^/  /'
if [ "$MAXN" = 1 ]; then
  id=$(grep -oP 'POWER_MODEL ID=\K[0-9]+(?= NAME=MAXN_SUPER)' /etc/nvpmodel.conf | head -1)
  if [ -n "$id" ]; then nvpmodel -m "$id" && ok "MAXN SUPER (mode $id) set"; else todo "no MAXN_SUPER in /etc/nvpmodel.conf (JetPack 6.2 + the super config?)"; fi
fi

echo "== groups for $USR (docker: no sudo for compose; dialout: the servo board's serial port)"
for g in docker dialout plugdev; do
  if id -nG "$USR" | grep -qw "$g"; then ok "$g"
  elif [ "$APPLY" = 1 ]; then usermod -aG "$g" "$USR" && ok "$g added (log out + in)"
  else todo "$g (--apply adds it)"; fi
done

echo "== docker"
docker --version 2>/dev/null | sed 's/^/  /' || todo "docker missing: sudo apt install docker.io docker-compose-v2"
docker compose version >/dev/null 2>&1 && ok "compose plugin" || todo "docker compose missing: sudo apt install docker-compose-v2"
echo "  data root: $(docker info -f '{{.DockerRootDir}}' 2>/dev/null)"
if [ -n "$DROOT" ]; then
  mkdir -p "$DROOT" && python3 - "$DROOT" <<'EOF'
import json, os, sys
p = "/etc/docker/daemon.json"
d = json.load(open(p)) if os.path.exists(p) else {}
d["data-root"] = sys.argv[1]
json.dump(d, open(p, "w"), indent=2)
print("  daemon.json data-root ->", sys.argv[1], "(other settings, e.g. the nvidia runtime, kept)")
EOF
  systemctl restart docker && ok "docker restarted on $DROOT (images pulled before this stay in the old root)"
fi

echo "== USB (the D405 and the servo board must never be suspended)"
echo "  autosuspend: $(cat /sys/module/usbcore/parameters/autosuspend) s (-1 = off)"
RULE=/etc/udev/rules.d/99-pnp-usb-power.rules
if [ -f "$RULE" ]; then ok "$RULE"
elif [ "$APPLY" = 1 ]; then
  echo 'ACTION=="add", SUBSYSTEM=="usb", TEST=="power/control", ATTR{power/control}="on"' > "$RULE"
  echo -1 > /sys/module/usbcore/parameters/autosuspend
  udevadm control --reload && ok "USB power always on (rule + now)"
else todo "USB autosuspend rule (--apply)"; fi
lsusb | grep -i -E "8086:0b5b|intel.*realsense|1a86|ch34|feetech|wch" | sed 's/^/  usb: /' || true

echo "== DDS socket buffers (Cyclone DDS: ~1 MB camera images between the robot container's processes)"
SYS=/etc/sysctl.d/60-pnp-dds.conf
rmem=$(cat /proc/sys/net/core/rmem_max)
if [ "$rmem" -ge 33554432 ]; then ok "net.core.rmem_max $rmem"
elif [ "$APPLY" = 1 ]; then
  printf 'net.core.rmem_max=33554432\nnet.core.wmem_max=33554432\n' > "$SYS"
  sysctl -q -p "$SYS" && ok "socket buffers 32 MB ($SYS, now and at boot)"
else todo "net.core.rmem_max is $rmem: images over Cyclone DDS drop (--apply sets 32 MB)"; fi
ser=$(ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || true)
if [ -n "$ser" ]; then echo "$ser" | sed 's/^/  serial: /'; else todo "no /dev/ttyACM*: servo board not plugged in?"; fi
# who holds the devices? (only one program can own a camera or the servo bus: e.g. a RealSense viewer, LeRobot)
for d in /dev/ttyACM* /dev/video*; do
  [ -e "$d" ] || continue
  p=$(fuser "$d" 2>/dev/null | xargs)
  [ -n "$p" ] && echo "  in use: $d by $(ps -o comm=,args= -p ${p// /,} 2>/dev/null | cut -c1-90 | sort -u | tr '
' ';')"
done
pgrep -af "pyrealsense|rs_hub|realsense2_camera|realsense-viewer" 2>/dev/null | cut -c1-110 | sed 's/^/  realsense user: /' || true

echo "== clock (zenoh drops messages stamped > 0.5 s in the future; latencies need one time base)"
if systemctl is-active --quiet chrony; then ok "chrony: $(chronyc tracking 2>/dev/null | grep 'System time' | sed 's/ *System time *: //')"
elif timedatectl show -p NTPSynchronized --value 2>/dev/null | grep -q yes; then ok "NTP synchronized (systemd-timesyncd): enough"
elif [ "$APPLY" = 1 ]; then apt-get install -y chrony >/dev/null && systemctl enable --now chrony && ok "chrony installed"
else todo "chrony (--apply installs it)"; fi

echo "== firewall (the PC connects IN to TCP 7447)"
if command -v ufw >/dev/null && ufw status | grep -q "Status: active"; then
  if ufw status | grep -q 7447; then ok "7447 open"
  elif [ "$APPLY" = 1 ]; then
    if [ -n "$PCIP" ]; then ufw allow from "$PCIP" to any port 7447 proto tcp; else ufw allow 7447/tcp; fi
    ok "7447 opened"
  else todo "ufw active and 7447 closed (--apply [--pc-ip IP])"; fi
else ok "no active ufw"; fi
echo "  this Jetson's addresses: $(hostname -I)"
echo
[ "$APPLY" = 1 ] || echo "check only. To apply: sudo bash $0 --apply   (opt-ins: --maxn, --docker-root DIR, --pc-ip IP)"
