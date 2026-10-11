"""Pick and place, over and over, on the REAL SO-101: place the held brick at a random goal, the D435i says where it landed,
grab it again from there, repeat. Every cycle -> one line in data/real_runs.jsonl (docs/learning/23: "reliable").
Run on the robot side (the Jetson), with the brick in the gripper or lying in the work band:
    docker exec jetson-robot-1 bash -c "source /repo/setup/ros_env.sh; source /ws/install/setup.bash; cd /repo;
                                        python3 -u tools/pick_place_real.py --cycles 5"
    --cycles N        how many place+grab cycles              --seed S       which random goals
    --region X0 X1 Y0 Y1   where goals may be (m); default 0.15-0.22 x -0.08-0.08: the band where the 12 V 2 A supply
                      keeps up (at ~28 cm reach the elbow hit 81 % load and every servo dropped off the bus)
    --min-move 0.04   a new goal is at least this far from the last one
    --desk-z Z        the plate in the arm's frame (the grab's close look measured -16..-18 mm on the plate)
Stops at the FIRST failure (missed grab, unreachable goal, no look), holding or releasing nothing more: a person
decides what's next. It drives tools/place.py and tools/grab.py (one process each, as by hand) and reads their output.
"""
import argparse
import datetime
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/repo")
LOG = REPO / "data" / "real_runs.jsonl"
NUM = r"(-?\d+\.\d+)"


def run(cmd: list, timeout: float = 300.0) -> tuple[int, str, float]:
    t0 = time.time()
    p = subprocess.run(["python3", "-u", *cmd], cwd=REPO, capture_output=True, text=True, timeout=timeout)
    out = "\n".join(line for line in (p.stdout + p.stderr).splitlines()
                    if line.strip() and not re.search(r"RTPS|SHM|message was lost|total count", line))
    print("\n".join("    " + line for line in out.splitlines()), flush=True)
    return p.returncode, out, time.time() - t0


def fixed_find(near=None):
    """The D435i's bricks (tools/fixed_look.py) -> (name, x, y, yaw_deg) of the one nearest `near`, or None."""
    out = run(["tools/fixed_look.py"], timeout=60.0)[1]
    found = [(m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4)))
             for m in re.finditer(rf"(lego_\w+)\s+at \({NUM}, {NUM}\) m, .*?yaw (-?\d+) deg", out)]
    if not found:
        return None, out
    if near is None:
        return found[0], out
    return min(found, key=lambda f: np.hypot(f[1] - near[0], f[2] - near[1])), out


def grab_at(b):
    """grab.py hovering straight over the D435i's position (no wrist look / scan: less motion, the scan's wrist swings
    snag the D405 cable), the close looks refine it."""
    return run(["tools/grab.py", "--at", f"{b[1]:.4f}", f"{b[2]:.4f}", "--at-yaw", f"{b[3]:.0f}"])


def parse_grab(out: str) -> dict:
    r = {"holding": bool(re.search(r"lift: ok.*still holding", out))}
    m = re.search(rf"target: (\w+) at \({NUM}, {NUM}\)", out)
    if m:
        r["name"], r["xy"] = m.group(1), [float(m.group(2)), float(m.group(3))]
    m = re.search(rf"closed: jaw width {NUM} mm", out)
    if m:
        r["jaw_mm"] = float(m.group(1))
    r["close_looks"] = len(re.findall(r"close look \d", out))
    return r


def parse_place(out: str, goal, look_out: str) -> dict:
    """The D435i's look after the release (tools/fixed_look.py: it sees the whole band; the wrist camera's single look
    pose doesn't reach y ~ +-7 cm): the found brick nearest the goal -> placement error."""
    r = {"released": "released (jaw" in out}
    if look_out:
        found = [(m.group(1), float(m.group(2)), float(m.group(3)))
                 for m in re.finditer(rf"(lego_\w+|\?)\s+at \({NUM}, {NUM}\)", look_out)]
        bricks = [f for f in found if f[0] != "?"] or found
        if bricks:
            name, x, y = min(bricks, key=lambda f: np.hypot(f[1] - goal[0], f[2] - goal[1]))
            r["landed"], r["seen_as"] = [x, y], name
            r["error_mm"] = round(float(np.hypot(x - goal[0], y - goal[1])) * 1000, 1)
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cycles", type=int, default=5)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--region", type=float, nargs=4, default=[0.15, 0.22, -0.08, 0.08])
    ap.add_argument("--min-move", type=float, default=0.04)
    ap.add_argument("--desk-z", type=float, default=-0.017)
    ap.add_argument("--grab-first", action="store_true", help="the gripper is empty: grab the brick before cycle 1")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    session = datetime.datetime.now().isoformat(timespec="seconds")
    last = None
    ok_cycles, errs = 0, []
    if a.grab_first:
        print("=== first grab", flush=True)
        b, _ = fixed_find()
        if b is None:
            print("STOP: the D435i sees no brick")
            return 1
        code, out, t = grab_at(b)
        g = parse_grab(out)
        if not g["holding"]:
            print("STOP: the first grab failed")
            return 1
        last = g.get("xy")
    for c in range(1, a.cycles + 1):
        while True:
            goal = [float(rng.uniform(a.region[0], a.region[1])), float(rng.uniform(a.region[2], a.region[3]))]
            if last is None or np.hypot(goal[0] - last[0], goal[1] - last[1]) >= a.min_move:
                break
        yaw = float(rng.uniform(-90, 90))
        rec = {"session": session, "cycle": c, "goal": [round(goal[0], 4), round(goal[1], 4)], "yaw_deg": round(yaw, 1)}
        print(f"\n=== cycle {c}/{a.cycles}: place at ({goal[0]:.3f}, {goal[1]:.3f}), yaw {yaw:.0f} deg", flush=True)
        code, out, t = run(["tools/place.py", "--xy", f"{goal[0]:.4f}", f"{goal[1]:.4f}", "--yaw", f"{yaw:.1f}",
                            "--desk-z", f"{a.desk_z}", "--no-look"])
        b, look_out = fixed_find(goal) if code == 0 else (None, "")
        rec["place"] = {"code": code, "s": round(t, 1), **parse_place(out, goal, look_out)}
        if code != 0 or not rec["place"]["released"] or "landed" not in rec["place"]:
            rec["result"] = "place failed"
        else:
            last = rec["place"]["landed"]
            print(f"=== cycle {c}: grab it again", flush=True)
            code, out, t = grab_at(b)
            rec["grab"] = {"code": code, "s": round(t, 1), **parse_grab(out)}
            rec["result"] = "ok" if rec["grab"]["holding"] else "grab failed"
        with LOG.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        p = rec["place"]
        print(f"=== cycle {c}: {rec['result']}; placed {p.get('error_mm', '?')} mm from the goal"
              + (f", grabbed (jaw {rec['grab'].get('jaw_mm')} mm, {rec['grab'].get('close_looks')} close look(s))"
                 if rec.get("grab", {}).get("holding") else ""), flush=True)
        if rec["result"] != "ok":
            print("STOP: a person decides what's next (the brick / the arm may need a look)")
            break
        ok_cycles += 1
        errs.append(p["error_mm"])
    print(f"\n{ok_cycles}/{a.cycles} cycles ok" + (f"; placement error mean {np.mean(errs):.1f} mm, max {max(errs):.1f} mm"
                                                 if errs else "") + f"; log: {LOG.relative_to(REPO)}")
    return 0 if ok_cycles == a.cycles else 1


if __name__ == "__main__":
    sys.exit(main())
