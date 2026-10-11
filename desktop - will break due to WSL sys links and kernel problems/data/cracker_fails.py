import glob, json, os
for d in sorted(glob.glob("/data/runs/*_benchmark_*")):
    s = os.path.join(d, "summary.json")
    if not os.path.exists(s) or json.load(open(s)).get("target") != "cracker_box":
        continue
    for l in open(os.path.join(d, "episodes.jsonl")):
        e = json.loads(l)
        if e["outcome"] == "MOTION_FAILED":
            print(os.path.basename(d)[:15], e["message"][:120])
