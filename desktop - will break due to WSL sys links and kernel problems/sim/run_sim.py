"""Main simulation entry point: scene + ROS 2 graphs, stepping until closed.

The ROS side (container) sees exactly the topics a real FR3 + D435i setup would publish
(see docs/interfaces.md). Start the zenoh bridge first (setup/zenoh_windows.ps1).

Run:  .\\run_isaac.ps1 sim\\run_sim.py                 # GUI
      .\\run_isaac.ps1 sim\\run_sim.py --headless --seconds 60
"""

import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--scene", default="config/scenes/basic.yaml")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--seconds", type=float, default=0.0, help="stop after this much sim time (0 = run until closed)")
parser.add_argument("--allow-multiple", action="store_true", help="don't refuse to start if another sim is running")
args = parser.parse_args()
if not args.scene.endswith(".yaml"):
    args.scene = f"config/scenes/{args.scene}.yaml"          # --scene so101_lego works too
__import__("os").environ["PNP_SCENE"] = args.scene.replace("\\", "/")   # robot, camera, ... follow this scene


def preflight() -> None:
    """Catch the two mistakes that silently break the ROS link (both happened; docs/learning/00_setup.md)."""
    import os
    import sys
    from pathlib import Path

    import psutil
    import yaml

    # 1. Same ROS domain as the container. run_isaac.ps1 sets it; launching python.bat directly may not.
    contract = yaml.safe_load((Path(__file__).resolve().parents[1] / "config/interfaces.yaml").read_text())
    want = str(contract["ros_domain_id"])
    if os.environ.get("ROS_DOMAIN_ID") != want:
        print(f"[sim] ROS_DOMAIN_ID was {os.environ.get('ROS_DOMAIN_ID')!r}; setting it to {want} "
              f"(start Isaac with run_isaac.ps1 to get the full environment)", flush=True)
        os.environ["ROS_DOMAIN_ID"] = want
    # 2. Only one sim: two Isaac instances publish the same /clock and joint topics and confuse
    #    everything downstream (bridge routes, controllers, sim time).
    me = os.getpid()
    # Only the interpreter itself (kit.exe / python.exe), not the cmd/powershell wrappers that launched it.
    others = [p.pid for p in psutil.process_iter(["pid", "name", "cmdline"])
              if p.info["pid"] != me and (p.info["name"] or "").lower() in ("kit.exe", "python.exe", "python")
              and any("run_sim.py" in c for c in (p.info["cmdline"] or []))]
    if others and not args.allow_multiple:
        print(f"[sim] ERROR: another run_sim.py is already running (pid {others}). Close it first "
              f"(or pass --allow-multiple).", flush=True)
        sys.exit(2)
    # 3. Fresh link per sim run: a long-lived Windows zenoh bridge keeps routes from previous (killed) Isaac
    #    instances and then forwards every new message once per stale route (measured: 3.6x duplicates after
    #    4 restarts). Restart it: setup/zenoh_windows.ps1 relaunches it in 3 s, and the container side resets
    #    itself when the bridge leaves.
    bridges = [p for p in psutil.process_iter(["name"]) if (p.info["name"] or "").lower() == "zenoh-bridge-ros2dds.exe"]
    for b in bridges:
        b.kill()
    if bridges:
        print("[sim] restarted the zenoh bridge for a clean link (its window relaunches it)", flush=True)
    else:
        print("[sim] WARNING: no zenoh bridge running - start setup\\zenoh_windows.ps1 or ROS won't see the sim", flush=True)
    print(f"[sim] ROS_DOMAIN_ID={os.environ['ROS_DOMAIN_ID']}. If bringup.launch.py is already running, restart it "
          f"once the sim is up: sim time restarts at 0 and running controllers stall.", flush=True)


preflight()

from isaacsim import SimulationApp

app = SimulationApp({"headless": args.headless})

import numpy as np
from common import log, shutdown
from pnp_core.catalog import load_fixed_camera, load_robot  # noqa: E402
from pnp_core.wrist import wrist_camera  # noqa: E402
from ros_graph import build_camera_graph, build_robot_graph
from scene import build_scene, reset_scene


def setup_viewports(h) -> None:
    """GUI only: aim the main viewport at the workcell and open a second viewport that looks through
    the simulated D435i (exactly what the perception pipeline sees, before the 15 Hz link downsampling)."""
    from isaacsim.core.utils.viewports import set_camera_view
    from omni.kit.viewport.utility import create_viewport_window

    set_camera_view(eye=np.array([1.6, -1.3, 1.1]), target=np.array([0.35, 0.0, 0.1]))
    cam = load_fixed_camera()
    win = create_viewport_window("D435i camera (sim)", width=cam["width"], height=cam["height"] + 40,
                                 position_x=40, position_y=80, camera_path=h.paths["camera"])
    if win is None:
        log("[sim] could not create the camera viewport window")
    else:
        log("[sim] opened viewport 'D435i camera (sim)' (Window > Viewport to find it if it's hidden)")
    if "wrist_camera" in h.paths:          # the eye-in-hand camera, next to it
        w = wrist_camera(load_robot())
        win = create_viewport_window("D405 wrist camera (sim)", width=w["width"], height=w["height"] + 40,
                                     position_x=40, position_y=80 + cam["height"] + 60,
                                     camera_path=h.paths["wrist_camera"])
        log("[sim] opened viewport 'D405 wrist camera (sim)'" if win else "[sim] could not open the wrist viewport")


def main() -> int:
    h = build_scene(args.scene)
    robot_cfg = load_robot()
    base_link = f"{h.paths['robot_links']}/{robot_cfg['frames']['base']}"
    build_robot_graph(h.paths["robot"], base_link,          # TF frames: target, each extra object's name, goal
                      gt_prims=[h.paths["target"], *h.paths.get("extra_objects", {}).values(), "/World/goal_marker"])
    build_camera_graph(h.paths["camera"])
    if "wrist_camera" in h.paths:
        build_camera_graph(h.paths["wrist_camera"], robot_cfg["wrist_camera"], "/ROS_WristCamera")
        w = wrist_camera(robot_cfg)
        log(f"[sim] wrist camera ({w['name']}) on {w['mount']['parent']} -> /wrist_camera/...")
    app.update()

    if not args.headless:
        setup_viewports(h)
    reset_scene(h, target_xy=h.cfg["target"]["default_xy"])
    h.world.play()
    from sim_services import SimServices      # after the ROS 2 bridge extension is loaded (it provides rclpy)
    services = SimServices(h)
    log("[sim] running; ROS 2 topics are live; sim commands via /sim/request -> /sim/response (tools/sim_client.py)")
    # Timing model (easy to get wrong!): World(physics_dt=1/120, rendering_dt=1/30). Each
    # world.step(render=True) advances ONE rendering_dt = 4 physics substeps (OnPhysicsStep fires 4x ->
    # 120 Hz joint states) and renders ONE camera frame (30 Hz). Don't mix in render=False steps:
    # those advance a single physics_dt and silently change how fast sim time runs.
    frame_dt = h.world.get_rendering_dt()
    frames = int(args.seconds / frame_dt) if args.seconds > 0 else None
    # Pace to real time: uncapped, the sim can run faster than the wall clock, which makes everything
    # downstream (controllers, MoveIt timeouts, the network link) see bursts of data.
    t_start = time.perf_counter()
    i = 0
    while app.is_running() and (frames is None or i < frames):
        h.world.step(render=True)
        services.spin_once()                 # scene changes happen only here, between steps
        i += 1
        ahead = i * frame_dt - (time.perf_counter() - t_start)
        if ahead > 0.002:
            time.sleep(ahead)
    wall = time.perf_counter() - t_start
    log(f"[sim] stopped after {i} frames = {i * frame_dt:.1f} s sim time in {wall:.1f} s wall time")
    return 0


if __name__ == "__main__":
    shutdown(app, main())
