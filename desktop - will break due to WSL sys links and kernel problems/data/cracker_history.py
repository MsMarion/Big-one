import glob, json, os
for d in sorted(glob.glob("/data/runs/*_benchmark_*")):
    s = os.path.join(d, "summary.json")
    if not os.path.exists(s):
        continue
    j = json.load(open(s))
    if j.get("target") != "cracker_box":
        continue
    out = j.get("outcomes", {})
    print(os.path.basename(d), j.get("trials"), "obst" if j.get("obstacles") else "", j.get("obstacle_source"), "succ", j.get("success_rate"), dict(out))
