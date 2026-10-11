"""One line per trial of the newest benchmark run: configuration + what happened."""
import glob
import json
import os

run = max(glob.glob("/data/runs/*_benchmark_*"), key=os.path.getmtime)
print(run)
for line in open(f"{run}/episodes.jsonl"):
    e = json.loads(line)
    boxes = ", ".join(f"{o['id']}@({o['center'][0]:.3f},{o['center'][1]:.3f})" for o in e["obstacles"])
    look = (e.get("look") or "-").replace(" -> grasp corrected", "")
    print(f"{e['trial']:2d} {e['grasp_source_requested']:9s} brick ({e['start_xy'][0]:.3f}, {e['start_xy'][1]:+.3f})  "
          f"goal ({e['goal_xy'][0]:.3f}, {e['goal_xy'][1]:+.3f})  {e['outcome']:13s} tries {e['grasp_attempts']}  "
          f"pick err {e['pick_err_mm']} mm  place err {e['gt_place_err_mm']} mm  | {look} | {boxes}")
