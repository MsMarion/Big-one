# Jetson: stream the real D435i to the PC

> **Two roles.** This page is the CAMERA role: the Jetson only streams a D435i. In the **ROBOT role** the SO-101 and
> its wrist D405 hang on the Jetson, which runs the servo bridge, MoveIt and the wrist flat map, and the PC is the
> brain. For that, see [docs/learning/22_jetson_d405.md](../docs/learning/22_jetson_d405.md)
> (`docker-compose.robot.yml`, `setup_host.sh`, `sync_to_jetson.sh`). Our unit is an Orin Nano Super 8 GB
> (JetPack 6.2).

The Jetson runs the RealSense driver and sends **compressed** color + depth over the network. The PC decompresses
them into exactly the topics the sim publishes, so perception (and later the whole pick & place) doesn't care
where the images come from. Concepts, PC side and troubleshooting: [docs/learning/08_real_sensors.md](../docs/learning/08_real_sensors.md).

```
 Jetson                                                       PC (WSL container)
 ┌────────────────────────────────────────┐                   ┌─────────────────────────────────────────────┐
 │ D435i ─USB3─ realsense2_camera         │                   │ zenoh router ─connects to─► Jetson :7447    │
 │   raw topics (local only)              │   TCP 7447        │ image_transport republish (decompress)      │
 │   + /compressed (JPEG), /compressedDepth│ ◄──────────────  │   -> /camera/camera/color/image_raw ...     │
 │ zenoh-bridge-ros2dds (listens :7447,   │   ~35 Mbit/s      │ detector, Octomap, RViz (unchanged)         │
 │   allow-list: compressed + camera_info)│                   │                                             │
 └────────────────────────────────────────┘                   └─────────────────────────────────────────────┘
```

## Which Jetson

| | Jetson **Orin** Nano / NX (recommended) | original Jetson **Nano** (4 GB) |
|---|---|---|
| JetPack / OS | 6.x, Ubuntu 22.04 | 4.6.x, Ubuntu 18.04 (end of life) |
| ROS 2 | Humble from apt (in Docker) | Humble from source (dusty-nv image) |
| RealSense | `ros-humble-realsense2-camera` apt | librealsense + realsense-ros built from source (1-2 h) |
| Dockerfile | `Dockerfile.orin` | `Dockerfile.nano` (untested on hardware here: check the base tag first) |

Only the camera runs on the Jetson, so even the old Nano is enough. Humble on the Jetson and Jazzy on the PC
are fine together: only standard message types cross, and they are wire-compatible.

## Setup (once)

1. **Flash** JetPack, then: `sudo apt update && sudo apt install -y docker.io docker-compose-v2 chrony`
   (JetPack 6 usually has Docker already). Add yourself to the docker group: `sudo usermod -aG docker $USER`.
2. **Clocks.** The PC rejects messages stamped > 0.5 s in the future (zenoh), and latency numbers need the same
   time base. Point chrony on the Jetson at the same NTP source as Windows (or at the PC itself), then check
   `chronyc tracking` (offset should be a few ms). On Windows, the Time service must be running.
3. **Network.** Fixed IP for the Jetson (a DHCP reservation in your router). Wired Ethernet is best; 5 GHz Wi-Fi
   works for 15 fps. The Jetson must accept TCP 7447 (Ubuntu has no firewall by default; with `ufw`:
   `sudo ufw allow from <PC-IP> to any port 7447 proto tcp`).
4. **Power** (original Nano): 5 V 4 A barrel jack + `sudo nvpmodel -m 0 && sudo jetson_clocks`. The D435i needs a
   **USB 3** port.
5. **Copy the files** from the PC (only `config/` and `jetson/` are needed on the Jetson):
   ```bash
   # on the PC, in WSL:
   rsync -av ~/pnp_repo/config ~/pnp_repo/jetson <user>@<jetson-ip>:~/pnp/
   ```

## Run

```bash
cd ~/pnp/jetson
JETSON_MODEL=orin docker compose up -d --build     # or JETSON_MODEL=nano
docker compose logs -f camera                      # "RealSense Node Is Up!"
docker compose exec camera bash -c 'source /opt/ros/humble/setup.bash && ros2 topic hz /camera/camera/color/image_raw/compressed'
```
Both containers restart automatically after a reboot (`restart: unless-stopped`).

On the **PC**: set `config/remote_sensors.yaml` (`enabled: true`, `endpoint: tcp/<jetson-ip>:7447`), restart the
container's router, start the stack with the real camera, check the stream:
```bash
wsl -d Ubuntu-24.04 -u root --exec bash /root/pnp_repo/docker/run_dev.sh "bash /repo/setup/restart_zenoh_container.sh"
wsl -d Ubuntu-24.04 -u root --exec bash /root/pnp_repo/docker/run_dev.sh "ros2 launch pnp_bringup bringup.launch.py sim:=false camera:=jetson"
wsl -d Ubuntu-24.04 -u root --exec bash /root/pnp_repo/docker/run_dev.sh "python3 /repo/tools/camera_check.py"
```
(Isaac Sim must not run at the same time: it publishes the same camera topic names.)

## No Jetson yet? Emulate it

`emulate_on_pc.sh` runs a second container on the PC that replays a recorded camera bag, compresses it like the
Jetson and serves it through the same zenoh config. It tests everything except the camera and the Jetson OS.
See "Testing without hardware" in docs/learning/08_real_sensors.md.

## Tuning

- Lower bandwidth/CPU: `rgb_camera.color_profile` / `depth_module.depth_profile` in `config/realsense_d435i.yaml`
  (e.g. `424x240x15`), or the `pub_max_frequencies` in `zenoh_jetson.json5`.
- PNG compression of depth costs CPU on a Nano; image_transport's `png_level` parameter (on the camera node)
  trades CPU for bandwidth.
