import glob, json, os, collections
for d in sorted(glob.glob("/data/runs/*_benchmark_*")):
    s = os.path.join(d, "summary.json")
    if not os.path.exists(s) or json.load(open(s)).get("target") != "cracker_box":
        continue
    src, look = collections.Counter(), collections.Counter()
    for l in open(os.path.join(d, "episodes.jsonl")):
        e = json.loads(l)
        src[(e.get("grasp_source") or "-").split(" (")[0]] += 1
        look[(e.get("look") or "no look").split(":")[0]] += 1
    print(os.path.basename(d)[:15], dict(src), dict(look))
