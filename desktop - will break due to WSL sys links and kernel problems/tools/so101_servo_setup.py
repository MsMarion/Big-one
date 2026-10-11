"""Bring-up helper for the REAL SO-101's servo bus (no ROS needed). docs/learning/17_so101.md

    python3 /repo/tools/so101_servo_setup.py scan                    # which servo IDs answer? voltage, temperature
        --compare FILE...  which calibration JSON (e.g. LeRobot's) belongs to THESE servos: its homing offsets must
                           equal the ones stored in the servos (LeRobot writes them in at calibration)
    python3 /repo/tools/so101_servo_setup.py monitor                 # torque OFF; live joint angles (calibrated)
    python3 /repo/tools/so101_servo_setup.py calibrate --out config/calibration/so101_desk/so101_follower.yaml
        --port /dev/ttyACM0 (default from config/robot_so101.yaml real.port)   --calibration FILE (monitor)

The recommended calibration is LeRobot's own (`lerobot-calibrate --robot.type=so101_follower ...`): it also writes a
homing offset into each servo so no joint's range crosses the 0/4095 wrap. Copy its JSON to the path in
config/robot_so101.yaml real.calibration. `calibrate` here is a fallback that only RECORDS the ranges (no writes to
the servos): move every joint slowly through its whole range, then Ctrl+C.

The SIGN check (always, after any calibration): `monitor`, then move each joint by hand in the direction the robot
description calls positive (RViz with the stack running shows it: bringup sim:=false hardware:=real, torque off).
A joint whose number goes the wrong way: set drive_mode 1 for it in the calibration file.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pnp_core.catalog import load_robot, repo_root  # noqa: E402
from pnp_core.feetech import Calibration, FeetechBus, SerialPort, decode_signed  # noqa: E402


def open_bus(port, baud):
    try:
        return FeetechBus(SerialPort(port, baud))
    except OSError as e:
        sys.exit(f"cannot open {port}: {e}\n(USB into WSL: usbipd attach --wsl --busid <id>; the container needs the "
                 f"device: run_dev.sh with PNP_USB=1)")


def main():
    cfg = load_robot("so101")
    real = cfg["real"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["scan", "monitor", "calibrate"])
    ap.add_argument("--port", default=real["port"])
    ap.add_argument("--baud", type=int, default=int(real["baud"]))
    ap.add_argument("--calibration", default=real["calibration"])
    ap.add_argument("--out", default="config/calibration/so101_desk/so101_follower.yaml")
    ap.add_argument("--compare", nargs="*", default=[], help="calibration JSONs to compare with the servos (scan)")
    a = ap.parse_args()
    bus = open_bus(a.port, a.baud)
    ids = {n: int(i) for n, i in real["motor_ids"].items()}

    if a.cmd == "scan":
        found, homing = 0, {}
        for i in range(1, 11):
            if bus.ping(i):
                found += 1
                name = next((n for n, j in ids.items() if j == i), "?")
                pos, volt, temp = (bus.read(i, r) for r in ("present_position", "present_voltage", "present_temperature"))
                homing[name] = decode_signed("homing_offset", bus.read(i, "homing_offset"))
                print(f"  id {i:2d} ({name:13s}) position {pos}  {volt / 10 if volt else '?'} V  {temp} C  "
                      f"homing offset {homing[name]}")
        missing = [f"{n} ({i})" for n, i in ids.items() if not bus.ping(i)]
        print(f"{found} servos answer" + (f"; MISSING: {missing}" if missing else ": all 6 present"))
        for f in a.compare:
            cal = json.load(open(f, encoding="utf-8"))
            diff = {n: (homing.get(n), m.get("homing_offset")) for n, m in cal.items() if homing.get(n) != m.get("homing_offset")}
            print(f"  {'MATCH' if not diff else 'differs'}  {f}" + (f"  (servo, file: {diff})" if diff else ""))
        return 0 if not missing else 1

    names = list(cfg["arm"]["joints"]) + [cfg["gripper"]["driven_joint"]]
    bus.sync_write("torque_enable", {ids[n]: 0 for n in names})
    print("torque OFF: the arm is limp (support it!)")

    if a.cmd == "monitor":
        path = a.calibration if a.calibration.startswith("/") else repo_root() / a.calibration
        cal = Calibration.load(path, gripper=cfg["gripper"]["driven_joint"],
                               gripper_limits=tuple(cfg["gripper"]["joint_limits"]))
        print("move the joints by hand; Ctrl+C to stop.  rad (deg)")
        try:
            while True:
                raw = bus.sync_read("present_position", [ids[n] for n in names])
                print("  " + "  ".join(f"{n[:8]} {cal.to_rad(n, raw[ids[n]]):+.3f} ({cal.to_rad(n, raw[ids[n]]) * 57.3:+5.0f})"
                                       if ids[n] in raw else f"{n[:8]} ?" for n in names), end="\r", flush=True)
                time.sleep(0.1)
        except KeyboardInterrupt:
            print()
        return 0

    # calibrate: record each joint's range of motion
    lo = {n: 10 ** 6 for n in names}
    hi = {n: -1 for n in names}
    print("move EVERY joint slowly through its whole range (gripper: fully closed and fully open); Ctrl+C when done")
    try:
        while True:
            raw = bus.sync_read("present_position", [ids[n] for n in names])
            for n in names:
                if ids[n] in raw:
                    lo[n], hi[n] = min(lo[n], raw[ids[n]]), max(hi[n], raw[ids[n]])
            print("  " + "  ".join(f"{n[:8]} {lo[n]}-{hi[n]}" for n in names), end="\r", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()
    out = {n: {"id": ids[n], "drive_mode": 0, "homing_offset": 0, "range_min": lo[n], "range_max": hi[n]} for n in names}
    for n in names:
        if hi[n] - lo[n] < 500:
            print(f"WARNING {n}: range only {hi[n] - lo[n]} ticks: moved through its whole range?")
        if lo[n] < 30 or hi[n] > 4065:
            print(f"WARNING {n}: its range touches the 0/4095 wrap: use LeRobot's calibration (homing offsets)")
    path = Path(a.out) if Path(a.out).is_absolute() else repo_root() / a.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) if path.suffix == ".json" else
                    "\n".join(f"{n}: {json.dumps(v)}" for n, v in out.items()) + "\n")
    print(f"written {path}: now check the SIGNS (monitor + RViz) and set drive_mode 1 where a joint is mirrored")
    return 0


if __name__ == "__main__":
    sys.exit(main())
