"""Phase 3 gate: is the perception accurate enough to GRASP? Closed loop, against simulator ground truth.

Run in the pnp container while run_sim.py + bringup (with perception) are running:
    python3 /repo/tools/eval_perception.py --trials 50 --negatives 10

For each trial: the sim's "randomize_target" command puts the can at a random upright pose in the pick region, we wait
for it to settle, ask /perception/detect for a FRESH detection, and compare with the ground-truth TF
(fr3_link0 -> target, published by Isaac). Negative trials move the can out of view and expect nothing.

Gate (plan, Phase 3):  detection rate >= 98%,  XY error p95 < 10 mm,  top-height error p95 < 10 mm,
                       zero detections on negative trials.
Results: printed summary + /data/runs/<run_id>/perception_eval.jsonl (one line per trial).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rclpy
import yaml
from pnp_interfaces.srv import DetectObjects
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from sim_client import SimClient  # noqa: E402


class Eval(Node):
    def __init__(self):
        super().__init__("pnp_eval_perception")
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.sim = SimClient(self)                     # /sim/request -> /sim/response (topics, see sim_client.py)
        self.detect = self.create_client(DetectObjects, "/perception/detect")

    def call(self, client, req, timeout=10.0):
        if not client.wait_for_service(timeout_sec=15.0):
            raise RuntimeError(f"service {client.srv_name} not available")
        fut = client.call_async(req)
        end = time.time() + timeout
        while not fut.done() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
        if not fut.done():
            raise RuntimeError(f"{client.srv_name} timed out")
        return fut.result()

    def wait(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def ground_truth(self, base, frame):
        t = self.tf.lookup_transform(base, frame, Time()).transform.translation
        return np.array([t.x, t.y, t.z])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--negatives", type=int, default=10)
    ap.add_argument("--settle", type=float, default=1.0, help="seconds to wait after moving the can")
    a = ap.parse_args()

    contract = yaml.safe_load((REPO / "config/interfaces.yaml").read_text())
    base, gt_frame = contract["frames"]["base"], contract["frames"]["object_ground_truth"]
    can = yaml.safe_load((REPO / "config/objects.yaml").read_text())["tomato_soup_can"]
    height = can["dims_m"][2]
    git = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + (f"_{git}" if git else "")
    out_dir = Path(os.environ.get("PNP_DATA", "/data")) / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / "perception_eval.jsonl", "w")

    rclpy.init()
    n = Eval()
    xy_err, z_err, found, false_pos, latencies = [], [], 0, 0, []
    for k in range(a.trials + a.negatives):
        negative = k >= a.trials
        moved = n.sim.call("target_away" if negative else "randomize_target")
        n.wait(a.settle)
        t0 = time.time()
        req = DetectObjects.Request(object_class=can["yolo_class"], timeout_s=3.0)
        res = n.call(n.detect, req)
        latencies.append(time.time() - t0)
        rec = {"trial": k, "negative": negative, "sim": moved, "detect_ok": res.success,
               "n_detections": len(res.detections.detections)}
        if not res.success:
            rec["error"] = res.message
        elif negative:
            false_pos += len(res.detections.detections)
        else:
            gt = n.ground_truth(base, gt_frame)
            rec["gt"] = gt.round(4).tolist()
            if res.detections.detections:
                # nearest detection to the truth (there is only one can in the scene)
                dets = [np.array([d.bbox.center.position.x, d.bbox.center.position.y, d.bbox.center.position.z])
                        for d in res.detections.detections]
                best = min(dets, key=lambda p: np.linalg.norm(p[:2] - gt[:2]))
                e_xy = float(np.linalg.norm(best[:2] - gt[:2]))
                e_z = float(abs((best[2] + height / 2) - (gt[2] + height / 2)))   # top-height error (center->top same offset)
                xy_err.append(e_xy)
                z_err.append(e_z)
                found += 1
                rec.update({"est": best.round(4).tolist(), "xy_err_mm": round(e_xy * 1000, 2), "z_err_mm": round(e_z * 1000, 2)})
        log.write(json.dumps(rec) + "\n")
        status = (f"xy {rec['xy_err_mm']:5.1f} mm  z {rec['z_err_mm']:4.1f} mm" if "xy_err_mm" in rec
                  else ("negative: " + str(rec["n_detections"]) + " detections") if negative else "NOT FOUND")
        print(f"[{k + 1:3d}/{a.trials + a.negatives}] {status}", flush=True)
    n.sim.call("reset_target")
    log.close()

    xy, z = np.array(xy_err) * 1000, np.array(z_err) * 1000
    rate = found / max(a.trials, 1)
    summary = {
        "run_id": run_id, "trials": a.trials, "negatives": a.negatives,
        "detection_rate": round(rate, 3),
        "xy_err_mm": {"mean": round(float(xy.mean()), 2), "p50": round(float(np.percentile(xy, 50)), 2),
                      "p95": round(float(np.percentile(xy, 95)), 2), "max": round(float(xy.max()), 2)} if len(xy) else None,
        "z_err_mm": {"p95": round(float(np.percentile(z, 95)), 2), "max": round(float(z.max()), 2)} if len(z) else None,
        "false_positives_on_negatives": false_pos,
        "detect_latency_s_p50": round(float(np.median(latencies)), 3),
    }
    gates = {"detection_rate>=0.98": rate >= 0.98,
             "xy_p95<10mm": bool(len(xy) and np.percentile(xy, 95) < 10),
             "z_p95<10mm": bool(len(z) and np.percentile(z, 95) < 10),
             "no_false_positives": false_pos == 0}
    summary["gates"] = gates
    summary["passed"] = all(gates.values())
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"RESULT: {'PASS' if summary['passed'] else 'FAIL'}   (details: {out_dir})")
    n.destroy_node()
    rclpy.try_shutdown()
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
