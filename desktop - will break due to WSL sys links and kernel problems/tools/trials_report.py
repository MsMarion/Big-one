"""The trial log's summary (pnp_core/trials.py, /data/trials.sqlite).

    python3 /repo/tools/trials_report.py              the newest run
    python3 /repo/tools/trials_report.py 4            run 4
    python3 /repo/tools/trials_report.py --ab 3 4     A/B: run 3 vs run 4, side by side
    python3 /repo/tools/trials_report.py --runs       every run, one line each
    python3 /repo/tools/trials_report.py --events [run] [--task ID]   events (calibration, grip, scene, ...)
    python3 /repo/tools/trials_report.py --artifacts [run]            bags / snapshots / scans with their sizes

Per run: success rate; failures by class, by object, by location (inside / edge / far: from the true position vs
the pick area) and by yaw; the seconds per stage (detect / plan / move / look / total); the grip widths.
"""
import json
import math
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, "/repo")
from pnp_core import trials  # noqa: E402
from pnp_core.catalog import load_scene  # noqa: E402


def zone(a, rg):
    if a["true_x"] is None:
        return "?"
    x, y = a["true_x"], a["true_y"]
    (x0, x1), (y0, y1) = rg["x"], rg["y"]
    near = [abs(x - x0), abs(x - x1), abs(y - y0), abs(y - y1)]
    if abs(x - x1) < 0.012 and min(abs(y - y0), abs(y - y1)) < 0.03:
        return "far corner"
    return "edge" if min(near) < 0.006 else "inside"


def yaw_bin(a):
    if a["true_yaw"] is None:
        return "?"
    d = abs(math.degrees(a["true_yaw"])) % 90
    return f"{int(d // 30) * 30:2d}-{int(d // 30) * 30 + 30} deg"


def summary(c, run):
    r = c.execute("SELECT * FROM runs WHERE id = ?", (run,)).fetchone()
    A = c.execute("SELECT * FROM attempts WHERE run = ? ORDER BY n", (run,)).fetchall()
    rg = load_scene()["target"]["spawn_region"]
    out = {"run": run, "n": len(A), "ok": sum(a["failure"] == "ok" for a in A), "flags": json.loads(r["flags"] or "{}"),
           "tier": r["tier"], "commit": r["commit_sha"], "note": r["note"]}
    out["rate"] = out["ok"] / out["n"] if A else float("nan")
    out["classes"] = Counter(a["failure"] for a in A)
    for key, f in (("object", lambda a: a["object"]), ("location", lambda a: zone(a, rg)), ("yaw", yaw_bin)):
        d = defaultdict(lambda: [0, 0])
        for a in A:
            d[f(a)][0] += a["failure"] == "ok"
            d[f(a)][1] += 1
        out[key] = dict(d)
    stages = ("s_detect", "s_plan", "s_move", "s_look", "s_total")
    out["time"] = {s[2:]: sum(a[s] or 0 for a in A) / max(1, len(A)) for s in stages}
    w = [(a["jaw_width"] - a["expected_width"]) * 1000 for a in A if a["jaw_width"] is not None]
    out["width_err_mm"] = (min(w), max(w)) if w else None
    out["retries"] = sum(a["retries"] or 0 for a in A)
    out["fails"] = [(a["n"], a["object"], a["failure"], (a["message"] or "")[:80]) for a in A if a["failure"] != "ok"]
    out["events"] = Counter(f"{e['kind']}:{e['severity']}" for e in
                            c.execute("SELECT kind, severity FROM events WHERE run = ?", (run,)))
    return out


def show(s):
    print(f"run {s['run']} ({s['tier']}, commit {s['commit']}) {s['note']}")
    print(f"  flags: {s['flags']}")
    print(f"  success: {s['ok']}/{s['n']} = {100 * s['rate']:.0f} %   retries: {s['retries']}")
    print(f"  failure classes: {dict(s['classes'])}")
    for key in ("object", "location", "yaw"):
        print(f"  by {key:8s}: " + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(s[key].items())))
    t = s["time"]
    print("  seconds per grab: " + ", ".join(f"{k} {v:.1f}" for k, v in t.items()))
    if s["width_err_mm"]:
        print(f"  jaw width - expected: {s['width_err_mm'][0]:+.1f} .. {s['width_err_mm'][1]:+.1f} mm")
    for f in s["fails"]:
        print(f"    #{f[0]} {f[1]}: {f[2]}  {f[3]}")


def ab(a, b):
    print(f"{'':22s}{'run ' + str(a['run']):>14s}{'run ' + str(b['run']):>14s}")
    print(f"{'success':22s}{a['ok']:>8d}/{a['n']:<5d}{b['ok']:>8d}/{b['n']:<5d}")
    print(f"{'rate':22s}{100 * a['rate']:>13.0f}%{100 * b['rate']:>13.0f}%")
    for cls in trials.FAILURE_CLASSES:
        if a["classes"].get(cls) or b["classes"].get(cls):
            print(f"{'  ' + cls:22s}{a['classes'].get(cls, 0):>14d}{b['classes'].get(cls, 0):>14d}")
    for k in a["time"]:
        print(f"{'  s ' + k:22s}{a['time'][k]:>14.1f}{b['time'][k]:>14.1f}")
    print(f"{'retries':22s}{a['retries']:>14d}{b['retries']:>14d}")
    for k in sorted(set(a["events"]) | set(b["events"])):
        print(f"{'  event ' + k:22s}{a['events'].get(k, 0):>14d}{b['events'].get(k, 0):>14d}")
    diff = {k: (a["flags"].get(k), b["flags"].get(k)) for k in set(a["flags"]) | set(b["flags"])
            if a["flags"].get(k) != b["flags"].get(k)}
    print(f"flags that differ: {diff}")


def show_events(c, run=None, task=None):
    q, args = "SELECT * FROM events", []
    if task:
        q, args = q + " WHERE task = ?", [task]
    elif run is not None:
        q, args = q + " WHERE run = ?", [run]
    rows = c.execute(q + " ORDER BY id", args).fetchall()
    for e in rows[-200:]:
        tm = time.strftime("%m-%d %H:%M:%S", time.localtime(e["t"]))
        print(f"  {tm} run {e['run']} {e['source']:19s} {e['kind']:20s} {e['severity']:5s} "
              f"task {e['task'] or '-':22s} {e['data'][:110]}")
    print(f"{len(rows)} events" + (" (last 200 shown)" if len(rows) > 200 else ""))


def show_artifacts(c, run=None):
    q, args = ("SELECT * FROM artifacts WHERE run = ?", [run]) if run is not None else ("SELECT * FROM artifacts", [])
    for r in c.execute(q + " ORDER BY id", args):
        tm = time.strftime("%m-%d %H:%M:%S", time.localtime(r["t"]))
        mb = f"{r['bytes'] / 1e6:8.1f} MB" if r["bytes"] is not None else "       ? MB"
        print(f"  {tm} run {r['run']} attempt {r['attempt'] or '-'} {r['kind']:14s} {mb}  {r['path']}  {r['note'] or ''}")


def main():
    c = trials.connect()
    nums = [int(a) for a in sys.argv[1:] if a.isdigit()]
    if "--events" in sys.argv:
        show_events(c, nums[0] if nums else None,
                    sys.argv[sys.argv.index("--task") + 1] if "--task" in sys.argv else None)
        return
    if "--artifacts" in sys.argv:
        show_artifacts(c, nums[0] if nums else None)
        return
    if "--runs" in sys.argv:
        for r in c.execute("SELECT r.*, (SELECT COUNT(*) FROM attempts WHERE run = r.id) AS n, (SELECT COUNT(*) FROM "
                           "attempts WHERE run = r.id AND failure = 'ok') AS ok FROM runs r ORDER BY id"):
            print(f"  run {r['id']:3d} {r['tier']:7s} {r['ok']}/{r['n']} commit {r['commit_sha']} {r['flags']} {r['note']}")
        return
    if "--ab" in sys.argv:
        i = sys.argv.index("--ab")
        ab(summary(c, int(sys.argv[i + 1])), summary(c, int(sys.argv[i + 2])))
        return
    run = nums[0] if nums else c.execute("SELECT MAX(id) FROM runs").fetchone()[0]
    show(summary(c, run))


if __name__ == "__main__":
    main()
