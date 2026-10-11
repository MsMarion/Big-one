import glob, json, os
run = max(glob.glob("/data/runs/*_benchmark_*"), key=os.path.getmtime)
for line in open(f"{run}/episodes.jsonl"):
    e = json.loads(line)
    if e["outcome"] != "SUCCESS":
        print(e["trial"], e["outcome"], e["last_state"], "|", e["message"][:300])
