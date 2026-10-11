# Isaac Sim RGB-D Pick & Place

A simulated **RealSense D435i** sees a **YCB tomato soup can**; a **YOLO-seg** detector plus aligned depth
locates it in 3D; a **Franka FR3** picks it and places it on a goal, planned with **MoveIt 2**. Later phases add
obstacle avoidance and the move to a real robot. The project is also a guided path into Isaac Sim: every phase
has a learning note in [`docs/learning/`](docs/learning/). Every technology and library used, and why:
[`docs/tech_stack.md`](docs/tech_stack.md).

| Component | Version |
|---|---|
| Isaac Sim | 5.1 (Windows, `C:\isaac-sim`) |
| ROS 2 | Jazzy, in Docker image `pnp:jazzy` (WSL Ubuntu 24.04 Docker Engine) |
| Isaac ↔ ROS link | zenoh-bridge-ros2dds 1.10.1 (TCP 127.0.0.1:7447) |

## Quick start

See [docs/learning/00_setup.md](docs/learning/00_setup.md) for the one-time setup and the reasons behind it.

```powershell
.\run_isaac.ps1 sim\check_assets.py      # are robot/object assets reachable?
.\run_isaac.ps1 sim\preview_scene.py     # build scene headless, save camera images, frame-alignment check
C:\isaac-sim\python.bat -m pytest -q     # pnp_core unit tests (no ROS, no sim)
```

**Play with it** (Isaac Sim GUI + RViz + an interactive menu: pick & place, randomize, obstacles, speed):
```powershell
.\demo.ps1               # known obstacles
.\demo.ps1 -Perceived    # obstacles only from the depth camera (Octomap)
.\demo.ps1 -Sensors      # + every sensor stage in RViz and the camera-measured scene (docs/learning/09_sensor_view.md)
```

**Test it** (tiered; the sim + stack keep running between calls; a headless sim is started if none runs):
```powershell
.\test.ps1                              # smoke: 5 trials, "fast" speed profile   -> after any change
.\test.ps1 -Tier regression             # 20 trials                              -> before a commit
.\test.ps1 -Tier gate -Obstacles        # 50 trials with random boxes            -> phase gates
.\test.ps1 -Fresh                       # rebuild + restart sim and stack first (after sim/ROS code changes)
.\test.ps1 -Pose                        # with the 6-D pose node: grasps from FoundationPose, YOLO as fallback
```

Full system by hand (three terminals; details in [docs/learning/05_moveit_ros2_control.md](docs/learning/05_moveit_ros2_control.md)):
```powershell
.\setup\zenoh_windows.ps1
.\run_isaac.ps1 sim\run_sim.py
wsl -d Ubuntu-24.04 -u root -- bash /root/pnp_repo/docker/run_dev.sh "ros2 launch pnp_bringup bringup.launch.py"
```

## Layout

```
config/       robot, object catalog, camera, scenes, interface contract (all tunables live here)
pnp_core/     pure-Python perception & grasp math (unit-tested)
sim/          Isaac Sim scripts (scene builder, previews, ROS graph, synthetic data)
ros2_ws/src/  ROS 2 packages (bringup, perception, manipulation, interfaces)
docker/       ROS image + dev container scripts
setup/        environment + link scripts (Windows and WSL)
tools/        dataset conversion, training, benchmarks, system checks
docs/         interface contract and learning notes
```

## Status

- [x] Phase 0: environment, container image, Isaac ↔ ROS link (verified)
- [x] Phase 1: scene + sensors + ROS graph (`tools/system_check.py --sim` passes in the container)
- [x] Phase 2: ros2_control + MoveIt (`tools/motion_smoke_test.py` passes: controller, OMPL, Pilz LIN, gripper)
- [x] Phase 3: Replicator data (2000 frames) → YOLO11n-seg (test mAP50-95 0.926) → detector node;
      closed-loop 3D error p95 2.5 mm, 0 false positives (`tools/eval_perception.py` passes)
- [x] Phase 4: `/pick_place` action (state machine over MoveIt + perception), randomized can and goal marker;
      `tools/run_benchmark.py` gate (see docs/learning/06_pick_place.md for results)
- [x] Speed-ups: `fast` speed profile for tests (median cycle 20.1 → 11.6 s), tiered `test.ps1`, headless sim,
      `demo.ps1` playground (docs/learning/06_pick_place.md)
- [x] Phase 5A: known box obstacles (random solvable layouts, MoveIt collision objects): gate 50/50, 0 collisions
      (docs/learning/07_obstacle_avoidance.md)
- [x] Phase 5B: obstacles from the depth camera (filtered, columnized cloud → MoveIt Octomap): gate 48/50,
      0 collisions (`test.ps1 -Obstacles -Perceived`, `demo.ps1 -Perceived`)
- [x] 6-D object pose: FoundationPose with the NGC ONNX networks (image `pnp:pose`): 0.5 mm / 0.6° vs sim
      ground truth (docs/learning/10_6d_pose.md)
- [x] Grasping from the 6-D pose (grasps in the object's frame, detector as fallback, ATTACHED pose while carried):
      20/20 (docs/learning/11_pose_grasping.md)
- [x] Second object, the YCB cracker box (yaw matters, 4 mm finger margin): scene files pick the target
      (`-Scene cracker_box`), 2-class detector (test mAP50-95 0.951), box grasp rules; 20/20 with obstacles, pose vs
      detector 10/10 vs 9/10 (docs/learning/12_second_object.md)
- [x] Map modes, switchable live: continuous or frozen per task + change monitor (appeared / gone / newly seen;
      policy warn / refresh / stop): 20/20 frozen with obstacles, 0 collisions, 0 false alarms
      (docs/learning/13_map_modes.md)
- [x] Wrist camera (D405, eye-in-hand) + close look before grasping (FoundationPose refined in the wrist view,
      size-aware look pose): detector start on the box 6.9 -> ~3.5 mm; camera keep-alive fixes the camera stalls
      after stack restarts (docs/learning/14_wrist_camera.md)
- [x] Sim realism, switchable live (ideal / realistic / stress): stereo depth noise, flying pixels, holes, camera
      placement error + drift; realistic 17/20, stress 0/10, all failures safe 
      (docs/learning/15_sim_realism.md)
- [x] Camera calibration with feedback: ChArUco hand-eye for both cameras (eye-to-hand with the board in the gripper,
      eye-in-hand over the table board) + pixel-space refinement: 0.3 mm / <0.1° from the true poses; TF from the
      calibration files, live reload; `calibration_check.py` system test (8 checks); AprilTag drift monitor
      (`/calibration/status`, policy warn / stop). Realistic 17/20 -> 20/20 (pick error 3.5 -> 1.0 mm), stress
      0/10 -> 7/10, to re-run (docs/learning/16_calibration.md)
- [x] Robot-agnostic stack + a second robot: the SO-101 (5 joints, one moving jaw) in a desk cell picking LEGO bricks.
      A scene names its robot; profiles generate the description + controller / MoveIt configs; numerical IK for
      <6-joint arms; LEGO detector (synthetic data). Sim: detector + obstacles 20/20 (known boxes) and 20/20 (camera Octomap), pick error ~1 mm, 0 collisions; cell camera calibration 0.4-0.5 mm. Real-arm path ready: servo bridge
      (Feetech bus <-> the same ros2_control topics as the sim), setup / motion-check tools, deployment checklist
      (docs/learning/17_so101.md)
- [ ] Phase 6: sim-to-real. Real-camera path (D435i on this PC via usbipd, or on a Jetson over the network)
      implemented with a mock arm (docs/learning/08_real_sensors.md, jetson/); arm driver once the robot is chosen
