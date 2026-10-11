"""Try to BREAK the workcell (node + page API + pick & place), any workcell scene (fr3_workcell, so101_workcell: the
objects, goal spots and clearances come from the running scene). Runs in the container:

    bash /repo/data/rospy.sh /repo/data/workcell_stress.py [A B C D E] [--soak N] [--quick]

A  bad input        malformed / wrong-typed / NaN / huge values, unknown ids and paths: answered, node alive,
                    /api/world stays STRICT JSON (a browser's JSON.parse rejects NaN)
B  races            double pick, pick during a scan, plan edits while running, Stop mid-carry, 200 parallel GETs
C  world changes    the chosen object removed / moved after the scan, a goal on a box, a plan whose step fails
D  soak             N random moves of random objects to random reachable goals
E  too close        an obstacle 1 / 2 / 3 / 4 / 5.5 cm from a brick, on every side, at several brick yaws: the pick
                    must refuse (BLOCKED: no room for the open jaw, the arm doesn't move) or succeed -- never hit it
                    (ground truth: the box didn't move); at 5.5 cm it must pick. The page's verdict must agree.
                    --quick: one yaw, two sides

Each check prints PASS / FAIL / NOTE; the summary lists the failures. The arm moves (C, D, parts of B).
"""
import json
import math
import random
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, "/repo/tools")
sys.path.insert(0, "/repo")
BASE = "http://localhost:8088"
results = []
CELL = {}                                                # filled from the running scene by setup_cell()
# per cell: an object at a spot where the overhead estimate is poor (the edge of the camera's view / near the robot)
EDGE_SPOTS = {"fr3_workcell": ("cracker_box", [0.205, -0.365], -0.36),
              "so101_workcell": ("lego_2x2", [0.14, 0.10], 0.4)}


def http(method, path, body=None, raw=None, timeout=15):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(BASE + path, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")


def strict(text):
    def bad(c):
        raise ValueError(f"non-standard JSON constant {c}")
    return json.loads(text, parse_constant=bad)


def world():
    code, text = http("GET", "/api/world")
    assert code == 200, f"/api/world -> {code}"
    return strict(text)


def post(path, body=None, raw=None):
    code, text = http("POST", path, body, raw)
    try:
        return code, strict(text)
    except ValueError:
        return code, {"_raw": text[:200]}


def check(name, ok, detail="", note=False):
    tag = "NOTE" if note and not ok else ("PASS" if ok else "FAIL")
    results.append((tag, name, detail))
    print(f"  {tag:4s}  {name}" + (f"  -- {detail}" if detail else ""), flush=True)


def alive():
    try:
        world()
        return True
    except Exception:  # noqa: BLE001
        return False


def wait_idle(timeout=300):
    end = time.time() + timeout
    while time.time() < end:
        w = world()
        if not w["task"]["busy"] and not w["scan"]["busy"]:
            return w
        time.sleep(1.0)
    raise TimeoutError("still busy")


def scan():
    wait_idle()
    post("/api/scan", {})
    time.sleep(1.0)
    return wait_idle()


def setup_cell():
    """The scene the stack runs: object classes, goal candidates, the clearance between objects, special spots."""
    from pnp_core.catalog import load_yaml
    sc = load_yaml(f"config/scenes/{world()['cell']['scene']}.yaml")
    cat = load_yaml("config/objects.yaml")
    names = [sc["target"]["object"]] + [e["object"] for e in sc.get("extra_objects", [])]
    half = {n: math.hypot(*cat[n]["dims_m"][:2]) / 2 for n in names}
    r = sorted(half.values())[-2:]
    CELL.update(
        scene=sc["name"], target=cat[names[0]]["yolo_class"], other=cat[names[1]]["yolo_class"],
        extra_name={cat[e["object"]]["yolo_class"]: e["name"] for e in sc.get("extra_objects", [])},
        # object <-> goal center: both footprints + a margin, at least the scene's gripper clearance
        margin=max(sc["obstacles"]["clearance_m"], sum(r) + 0.02), desk=sc["table"], reach=sc["ui"]["reach_m"])
    # goal candidates: a grid over the pick + place regions (inside the reach band and the camera's view by design)
    regs = [sc["goal"]["region"], sc["target"]["spawn_region"]]
    step = CELL["margin"] / 3
    cands = []
    for rg in regs:
        nx, ny = max(1, int((rg["x"][1] - rg["x"][0]) / step)), max(1, int((rg["y"][1] - rg["y"][0]) / step))
        cands += [[round(rg["x"][0] + (rg["x"][1] - rg["x"][0]) * (i + 0.5) / nx, 4),
                   round(rg["y"][0] + (rg["y"][1] - rg["y"][0]) * (j + 0.5) / ny, 4)] for i in range(nx) for j in range(ny)]
    random.Random(3).shuffle(cands)
    CELL["cands"] = cands
    t = sc["table"]
    xmin, ymax = t["center"][0] - t["size"][0] / 2, t["center"][1] + t["size"][1] / 2
    CELL["desk_edge_behind"] = [round(xmin + 0.005, 4), 0.0]
    CELL["desk_edge_side"] = [round(t["center"][0], 4), round(ymax - 0.005, 4)]
    # behind the robot, on the desk, in the reach band (if the desk reaches there): out of the camera's view
    rm = (CELL["reach"][0] + CELL["reach"][1]) / 2
    behind = [[round(rm * math.cos(a), 4), round(rm * math.sin(a), 4)] for a in [math.radians(d) for d in range(120, 181, 5)]]
    CELL["behind"] = next((b for b in behind if b[0] > xmin + max(half.values()) + 0.01), behind[0])
    print(f"cell {CELL['scene']}: target {CELL['target']}, other {CELL['other']}, margin {CELL['margin']:.3f} m, "
          f"{len(cands)} goal candidates, behind {CELL['behind']}")


def free_goal(w, exclude=None):
    """A goal candidate clear of every object by the cell's margin (80 % of it on a crowded desk: the SO-101's
    7 x 20 cm work band with 3 bricks + a planned one; still more than the node's own footprint check)."""
    for f in (1.0, 0.8):
        for xy in CELL["cands"]:
            if all(math.hypot(xy[0] - o["xy"][0], xy[1] - o["xy"][1]) > f * CELL["margin"] for o in w["objects"]):
                return xy
    raise RuntimeError("no free goal")


def obj_of(w, cls):
    return next((o for o in w["objects"] if o["class"] == cls), None)


def pick(cls, xy, yaw=0.0, wait=True):
    w = world()
    o = obj_of(w, cls)
    if o is None:
        return None, "not in the scan"
    code, r = post("/api/pick", {"id": o["id"], "xy": list(xy), "yaw_deg": yaw})
    if not r.get("ok"):
        return r, f"REFUSED: {r.get('why')}"
    if not wait:
        return r, None
    time.sleep(1.0)
    w = wait_idle(400)
    return r, w["task"]["result"]


# --- A: bad input ---------------------------------------------------------------------------------------------------
def phase_a():
    print("A  bad input")
    cases = [
        ("malformed JSON", "/api/goal", None, b"{xy: [1,"),
        ("goal without xy", "/api/goal", {}, None),
        ("goal xy strings", "/api/goal", {"xy": ["a", "b"]}, None),
        ("goal xy one number", "/api/goal", {"xy": [0.5]}, None),
        ("goal xy null", "/api/goal", {"xy": None}, None),
        ("body is a list", "/api/select", None, b"[1, 2]"),
        ("select unknown id", "/api/select", {"id": "o999"}, None),
        ("pick unknown id", "/api/pick", {"id": "nope", "xy": [0.5, 0.0]}, None),
        ("pick with string xy", "/api/pick", {"id": "o1", "xy": "here"}, None),
        ("plan/remove index text", "/api/plan/remove", {"index": "abc"}, None),
        ("plan/remove index 99", "/api/plan/remove", {"index": 99}, None),
        ("plan/remove index -1", "/api/plan/remove", {"index": -1}, None),
        ("plan/add unknown id", "/api/plan/add", {"id": "zzz", "xy": [0.5, 0]}, None),
        ("unknown POST path", "/api/explode", {}, None),
        ("plan/update no such step", "/api/plan/update", {"index": 42, "yaw_deg": 10}, None),
        ("plan/update index text", "/api/plan/update", {"index": "x", "yaw_deg": 10}, None),
    ]
    for name, path, body, raw in cases:
        try:
            code, r = post(path, body, raw)
            check(f"{name}: answered ({code}), node alive", code in (200, 400, 404, 500) and alive(), str(r)[:90])
        except Exception as e:  # noqa: BLE001
            check(f"{name}: answered", False, repr(e)[:120])
    code, text = http("GET", "/api/feeds")
    feeds = strict(text) if code == 200 else []
    check(f"camera feeds listed: {', '.join(f['id'] for f in feeds)}", code == 200 and len(feeds) >= 4)
    for f in feeds:
        code, body = 404, ""
        for _ in range(4):                               # the first request subscribes; frames follow
            code, body = http("GET", f"/api/feed/{f['id']}.jpg")
            if code == 200:
                break
            time.sleep(1.0)
        check(f"feed {f['id']}: a frame ({len(body)} B)", code == 200)
    code, _ = http("GET", "/api/feed/../../etc/passwd.jpg")
    check(f"feed with a bad id: 404 ({code})", code == 404)
    for name, path in [("GET path traversal", "/../../etc/passwd"), ("GET unknown", "/api/nothing"),
                       ("GET index with query", "/?x=<script>")]:
        code, text = http("GET", path)
        check(f"{name}: {code}", code in (200, 404) and "root:x:0" not in text)
    # values that parse but are nonsense: the world JSON must stay valid for the page
    for name, xy in [("NaN goal", [float("nan"), 0.1]), ("Infinity goal", [float("inf"), 0.0]),
                     ("huge goal", [1e300, -1e300]), ("negative-zero goal", [-0.0, 0.0])]:
        raw = json.dumps({"xy": xy, "yaw_deg": 0}).encode()            # Python writes NaN / Infinity
        code, r = post("/api/goal", raw=raw)
        try:
            w = world()
            g = w.get("goal")
            sane = g is None or all(math.isfinite(v) for v in g["xy"])
            check(f"{name}: /api/world still strict JSON, a finite goal", sane, f"{code} {r}")
        except Exception as e:  # noqa: BLE001
            check(f"{name}: /api/world still strict JSON", False, repr(e)[:120])
    post("/api/goal", {"xy": CELL["cands"][0], "yaw_deg": 0})
    code, r = post("/api/goal", {"xy": CELL["cands"][0], "yaw_deg": "ninety"})
    check("yaw as text: answered, node alive", alive(), f"{code} {r}")
    # a goal off the table / behind the robot: refused by the pick node, quickly, without a collision
    w = scan()
    t0 = time.time()
    T, O = CELL["target"], CELL["other"]
    r, res = pick(T, [5.0, 5.0])
    check("pick to (5, 5) m: refused", res is not None and not res.startswith("SUCCESS"), f"{res} ({time.time() - t0:.0f} s)")
    r, res = pick(T, CELL["desk_edge_behind"])
    check("pick to the desk's back edge: refused before moving", res is not None and res.startswith("REFUSED"), res)
    r, res = pick(T, CELL["behind"])
    check(f"pick to behind the robot {CELL['behind']}: refused before moving", res is not None and res.startswith("REFUSED"), res)
    r, res = pick(O, CELL["desk_edge_side"])
    check("pick to the desk's side edge (would overhang): refused", res is not None and res.startswith("REFUSED"), res)
    w = world()
    box = obj_of(w, O)
    r, res = pick(T, box["xy"])
    check("pick onto the other object: refused before moving", res is not None and res.startswith("REFUSED"), res)
    post("/api/plan/clear", {})
    can = obj_of(w, T)
    _, r = post("/api/plan/add", {"id": can["id"], "xy": box["xy"]})
    check("plan: a move onto an object nobody moves away: refused", not r.get("ok"), str(r))
    _, r1 = post("/api/plan/add", {"id": box["id"], "xy": free_goal(w)})
    _, r2 = post("/api/plan/add", {"id": can["id"], "xy": box["xy"]})
    check("plan: onto an object an earlier step moves away: accepted", r1.get("ok") and r2.get("ok"), f"{r1} {r2}")
    _, r = post("/api/plan/update", raw=json.dumps({"index": 0, "yaw_deg": float("nan")}).encode())
    check("plan/update with a NaN yaw: refused", not r.get("ok"), str(r))
    _, r = post("/api/plan/update", {"index": 0, "xy": CELL["behind"]})
    check("plan/update to behind the robot: refused", not r.get("ok"), str(r))
    _, r = post("/api/plan/update", {"index": 0, "yaw_deg": 30})
    st = world()["plan"][0]
    check("plan/update yaw 30: stored", r.get("ok") and abs(st["yaw_deg"] - 30) < 1e-6, f"{r} {st['yaw_deg']}")
    post("/api/plan/clear", {})
    post("/api/goal", {"xy": CELL["cands"][0], "yaw_deg": 0})


# --- B: races -------------------------------------------------------------------------------------------------------
def phase_b():
    print("B  races")
    w = scan()
    can, box = obj_of(w, CELL["target"]), obj_of(w, CELL["other"])
    free = free_goal(w)
    # double pick
    _, r1 = post("/api/pick", {"id": can["id"], "xy": free})
    _, r2 = post("/api/pick", {"id": box["id"], "xy": free_goal(w)})
    check("second pick while one runs: refused", r1.get("ok") and not r2.get("ok"), f"{r1} / {r2}")
    _, r = post("/api/scan", {})
    check("scan while a pick runs: refused", not r.get("ok"), str(r))
    _, r = post("/api/home", {})
    check("home while a pick runs: refused", not r.get("ok"), str(r))
    post("/api/plan/clear", {})
    _, r = post("/api/plan/run", {})
    check("plan/run while a pick runs: refused", not r.get("ok"), str(r))
    # 200 parallel GETs during the pick
    errs = []

    def hammer():
        for _ in range(20):
            try:
                world()
                http("GET", "/api/camera.jpg")
            except Exception as e:  # noqa: BLE001
                errs.append(repr(e))
    ts = [threading.Thread(target=hammer) for _ in range(10)]
    t0 = time.time()
    [t.start() for t in ts]
    [t.join() for t in ts]
    check(f"400 parallel GETs during a pick: no errors ({time.time() - t0:.1f} s)", not errs, "; ".join(errs[:3]))
    w = wait_idle(400)
    check("the first pick still finished", w["task"]["result"] is not None, w["task"]["result"])
    # pick right after starting a scan (the scan moves the arm home; a pick must not start under it)
    _, r = post("/api/scan", {})
    w2 = world()
    o = obj_of(w2, CELL["other"]) or obj_of(w2, CELL["target"])
    _, r = post("/api/pick", {"id": o["id"], "xy": free_goal(w2)}) if o else (0, {"ok": False})
    check("pick while a scan runs: refused", not r.get("ok"), str(r), note=False)
    wait_idle(400)
    # plan edits while it runs
    w = scan()
    post("/api/plan/clear", {})
    can = obj_of(w, CELL["target"])
    dst = free_goal(w)
    post("/api/plan/add", {"id": can["id"], "xy": dst, "yaw_deg": 0})
    post("/api/plan/add", {"id": can["id"], "xy": can["xy"], "yaw_deg": 0})        # and back (chained)
    _, r = post("/api/plan/run", {})
    time.sleep(1.0)
    _, rr = post("/api/plan/remove", {"index": 1})
    _, rc = post("/api/plan/clear", {})
    _, ra = post("/api/plan/add", {"id": can["id"], "xy": free_goal(w)})
    check("plan remove / clear while running: refused", r.get("ok") and not rr.get("ok") and not rc.get("ok"), f"{rr} {rc}")
    check("plan add while running: refused (or harmless)", not ra.get("ok"), str(ra), note=True)
    w = wait_idle(800)
    steps = [s["status"] for s in w["plan"]]
    check("plan there-and-back (chained) completed", steps[:2] == ["done", "done"], f"{steps} {w['task']['result']}")
    post("/api/plan/clear", {})
    # Stop mid-carry: the object is in the gripper when the action is canceled
    w = scan()
    can = obj_of(w, CELL["target"])
    dst = free_goal(w)
    post("/api/pick", {"id": can["id"], "xy": dst})
    end, seen = time.time() + 200, False
    while time.time() < end:
        log = world()["task"]["log"]
        if any("VERIFY_GRASP" in line for line in log):
            seen = True
            break
        time.sleep(0.3)
    time.sleep(1.5)                                       # lifting / carrying
    _, r = post("/api/cancel", {})
    w = wait_idle(300)
    check("Stop mid-carry: task ends, node idle", seen and not w["task"]["busy"], f"{w['task']['result']}")
    w = scan()
    can2 = obj_of(w, CELL["target"])
    back = can2 is not None and math.hypot(can2["xy"][0] - can["xy"][0], can2["xy"][1] - can["xy"][1]) < 0.02
    check("after Stop mid-carry: the can was put back where it was picked (recovery)", back,
          f"picked at {can['xy']}, now {can2['xy'] if can2 else None}, z_center {can2['z_center'] if can2 else None}")
    if can2:
        dst = free_goal(w)
        _, res = pick(CELL["target"], dst)
        check("after Stop mid-carry: the next pick works", res and res.startswith("SUCCESS"), res)


# --- C: the world changes under it ----------------------------------------------------------------------------------
def phase_c(sim):
    print("C  world changes")
    sim.call("reset_target")
    time.sleep(2.0)
    w = scan()
    can = obj_of(w, CELL["target"])
    sim.call("target_away")                            # the can leaves the table after the scan
    time.sleep(1.0)
    _, r = post("/api/pick", {"id": can["id"], "xy": free_goal(w)})
    w = wait_idle(300)
    res = w["task"]["result"] or ""
    check("chosen object gone after the scan: NO_OBJECT, no grasp attempt", res.startswith("NO_OBJECT"), res)
    names = [o["class"] for o in w["objects"]]
    check("rescan after it: the object is not in the world any more", CELL["target"] not in names, str(names))
    sim.call("reset_target")
    time.sleep(2.0)
    w = scan()
    can = obj_of(w, CELL["target"])
    moved = sim.call("randomize_target")              # the can moves somewhere else after the scan
    time.sleep(1.5)
    d = math.hypot(moved["xy"][0] - can["xy"][0], moved["xy"][1] - can["xy"][1])
    _, r = post("/api/pick", {"id": can["id"], "xy": free_goal(w)})
    w = wait_idle(300)
    res = w["task"]["result"] or ""
    if d > 0.06:
        check(f"chosen object moved {d * 100:.0f} cm after the scan: refused (scan again), nothing grabbed",
              res.startswith("NO_OBJECT") and "scan again" in res, res)
    else:
        check(f"chosen object moved {d * 100:.0f} cm (< 6 cm hint): still picked", res.startswith("SUCCESS"), res)
    # the box where the overhead detector is ~3 cm off (near the robot, the edge of the camera's view): seen once,
    # the 6-D pose was right, the detector's estimate was taken, the close look failed and the grasp knocked the box
    # off the table. Now: a success, or UNCERTAIN with the box left where it was -- never a knocked box.
    if CELL["scene"] in EDGE_SPOTS:
        cls, exy, eyaw = EDGE_SPOTS[CELL["scene"]]
        name = CELL["extra_name"].get(cls, cls)
        sim.call("reset_target")
        sim.call("place_object", name=name, xy=exy, yaw=eyaw)
        time.sleep(2.0)
        w = scan()
        _, res = pick(cls, free_goal(w))
        after = sim.call("state")["objects"][name]
        moved = math.hypot(after[0] - exy[0], after[1] - exy[1])
        ok = res is not None and (res.startswith("SUCCESS") or (res.startswith("UNCERTAIN") and moved < 0.01))
        check(f"{cls} at a poor-view spot {exy}: placed, or left alone as UNCERTAIN (moved {moved * 100:.1f} cm)",
              ok, res)
    # a goal on top of a box
    sim.call("reset_target")
    time.sleep(2.0)
    blk = sim.call("block_goal")
    gxy = blk["obstacles"][0]["center"][:2]
    time.sleep(2.0)
    w = scan()
    _, res = pick(CELL["target"], gxy)
    check("goal on top of a box: refused, no collision", res is not None and not res.startswith("SUCCESS"), res)
    sim.call("clear_obstacles")
    time.sleep(1.0)
    # a plan whose first step must fail: the rest is skipped, the node ends idle
    w = scan()
    post("/api/plan/clear", {})
    can, box = obj_of(w, CELL["target"]), obj_of(w, CELL["other"])
    _, r = post("/api/plan/add", {"id": can["id"], "xy": [2.0, 0.0]})
    check("plan: a move off the desk: refused when added", not r.get("ok"), str(r))
    g1 = free_goal(w)
    post("/api/plan/add", {"id": can["id"], "xy": g1})
    w2 = {**w, "objects": w["objects"] + [{**can, "xy": g1}]}
    post("/api/plan/add", {"id": box["id"], "xy": free_goal(w2)})
    sim.call("target_away")                            # step 1's object leaves the table after planning
    time.sleep(1.0)
    post("/api/plan/run", {})
    time.sleep(1.0)
    w = wait_idle(400)
    st = [s["status"] for s in w["plan"]]
    check("plan with a failing first step: stops, rest skipped", st == ["failed", "skipped"], f"{st} {w['task']['result']}")
    post("/api/plan/clear", {})
    sim.call("reset_target")


# --- D: soak ---------------------------------------------------------------------------------------------------------
def phase_d(n):
    print(f"D  soak: {n} random moves")
    rng = random.Random(7)
    ok = 0
    for k in range(n):
        w = scan()
        objs = w["objects"]
        if not objs:
            check(f"soak {k + 1}: objects seen", False, "scan found nothing")
            continue
        o = rng.choice(objs)
        clear = [c for c in CELL["cands"]
                 if all(math.hypot(c[0] - p["xy"][0], c[1] - p["xy"][1]) > CELL["margin"] for p in objs)]
        if not clear:      # the objects happen to block every candidate spot: a layout, not a robot failure
            check(f"soak {k + 1}: a free goal spot", False, "every candidate spot is next to an object: skipped", note=True)
            continue
        xy = rng.choice(clear)
        yaw = rng.choice([-90, -45, 0, 30, 60, 90])
        t0 = time.time()
        _, res = pick(o["class"], xy, yaw)
        good = bool(res) and res.startswith("SUCCESS")
        ok += good
        log = world()["task"]["log"]
        src = next((line.split("grasp from ")[1].split(":")[0] for line in log if "grasp from" in line), "?")
        check(f"soak {k + 1}: {o['class']} -> ({xy[0]:.2f}, {xy[1]:.2f}) {yaw} deg", good,
              f"{res} [{src}, {time.time() - t0:.0f} s]")
    print(f"  soak: {ok}/{n} succeeded")


def phase_e(sim, quick=False, only_gaps=None):
    """An obstacle (the pool's smallest box) at a measured gap beside a brick; the brick picked to a clear goal."""
    from pnp_core.catalog import load_yaml
    print("E  too close: obstacles beside the object being picked")
    sc = load_yaml(f"config/scenes/{CELL['scene']}.yaml")
    cat = load_yaml("config/objects.yaml")
    ex = sc.get("extra_objects", [])
    subjects = [(e["name"], e["object"]) for e in ex[:1]] + [("target", sc["target"]["object"])]
    others = {e["name"] for e in ex} | {"target"}
    box = min((b for b in sc["obstacles"]["pool"] if b.get("role") != "between"), key=lambda b: max(b["size"]))
    bh = box["size"][0] / 2
    rg, t = sc["target"]["spawn_region"], sc["table"]
    S = [round((rg["x"][0] + rg["x"][1]) / 2 + 0.01, 3), round(rg["y"][0] + 0.75 * (rg["y"][1] - rg["y"][0]), 3)]
    G = [round((rg["x"][0] + rg["x"][1]) / 2 - 0.01, 3), round(rg["y"][0] + 0.15 * (rg["y"][1] - rg["y"][0]), 3)]
    park = [[round(t["center"][0] + t["size"][0] / 2 - 0.06, 3), round(sy * (t["size"][1] / 2 - 0.06), 3)] for sy in (-1, 1)]
    gaps = only_gaps or [0.01, 0.02, 0.03, 0.04, 0.055]
    sides = [(1, 0), (0, 1)] if quick else [(1, 0), (-1, 0), (0, 1), (0, -1)]
    yaws = [0.0] if quick else [0.0, 45.0]
    tally = {}
    for name, objname in subjects:
        obj = cat[objname]
        half = [d / 2 for d in obj["dims_m"][:2]]
        cases = [(g, sd, yw) for g in gaps for sd in sides for yw in yaws] if name != "target" else \
            [(g, sd, 0.0) for g in (only_gaps or (0.01, 0.03, 0.055)) for sd in [(1, 0), (0, 1)]]
        for gap, side, ydeg in cases:
            y = math.radians(ydeg)
            c, s_ = math.cos(y), math.sin(y)
            u = (c * side[0] - s_ * side[1], s_ * side[0] + c * side[1])          # the side, in the base frame
            reach = abs(side[0]) * half[0] + abs(side[1]) * half[1]               # the brick's half size that way
            bxy = [S[0] + u[0] * (reach + gap + bh), S[1] + u[1] * (reach + gap + bh)]
            sim.call("clear_obstacles")
            sim.call("place_object", name=name, xy=S, yaw=y)
            for k, o in enumerate(sorted(others - {name})):
                sim.call("place_object", name=o, xy=park[k % 2], yaw=0.0)
            sim.call("set_obstacles", boxes=[{"id": box["id"], "xy": bxy, "yaw": y}])
            time.sleep(2.0)
            w = scan()
            o = obj_of(w, obj["yolo_class"])
            label = f"{obj['yolo_class']} gap {gap * 100:.1f} cm side {side} yaw {ydeg:.0f}"
            if o is None:
                check(f"too close: {label}: seen", False, "not in the scan")
                continue
            # the page's verdict as the user sees it before pressing Pick: selected, re-checked at the 6-D pose
            post("/api/select", {"id": o["id"]})
            for _ in range(16):
                time.sleep(0.5)
                o = obj_of(world(), obj["yolo_class"]) or o
                if o.get("checked_with") == "6-D pose":
                    break
            page, src = o.get("pickable"), o.get("checked_with")
            _, res = pick(obj["yolo_class"], G, 0.0)
            res = res or ""
            moved = max((b.get("displacement_m", 0.0) for b in sim.call("state").get("obstacles", [])), default=0.0)
            hit = moved > 0.005
            outcome = res.split(":")[0]
            ok = not hit and outcome in ("SUCCESS", "BLOCKED", "UNCERTAIN") and (gap < 0.05 or outcome == "SUCCESS")
            check(f"too close: {label}", ok, f"{res[:120]} [box moved {moved * 1000:.1f} mm, page pickable={page} ({src})]")
            if ok and outcome == "UNCERTAIN":    # refused because perception wasn't sure: safe, but not the answer
                check(f"too close: {label}: refused as UNCERTAIN, not BLOCKED", False, res[:120], note=True)
            if page is not None and (page is False) != (outcome == "BLOCKED"):
                check(f"too close: {label}: page agrees with the pick", False,
                      f"page pickable={page} ({src}), pick {outcome}", note=True)
            tally.setdefault((obj["yolo_class"], gap), []).append(outcome + (" HIT" if hit else ""))
    print("  gap -> outcomes")
    for (cls, gap), outs in sorted(tally.items()):
        print(f"    {cls:9s} {gap * 100:4.1f} cm: " + ", ".join(f"{k} x{outs.count(k)}" for k in sorted(set(outs))))
    sim.call("clear_obstacles")


def main():
    phases = [a for a in sys.argv[1:] if a in "ABCDE" and len(a) == 1] or ["A", "B", "C", "D"]
    soak = int(sys.argv[sys.argv.index("--soak") + 1]) if "--soak" in sys.argv else 8
    sim = None
    if "C" in phases or "E" in phases:
        import rclpy
        from sim_client import SimClient
        rclpy.init()
        sim = SimClient(rclpy.create_node("workcell_stress_sim"))
    setup_cell()
    t0 = time.time()
    for ph in phases:
        try:
            {"A": phase_a, "B": phase_b, "C": lambda: phase_c(sim), "D": lambda: phase_d(soak),
             "E": lambda: phase_e(sim, "--quick" in sys.argv,
                                  [float(g) / 100 for g in sys.argv[sys.argv.index("--gaps") + 1].split(",")]
                                  if "--gaps" in sys.argv else None)}[ph]()
        except Exception as e:  # noqa: BLE001
            check(f"phase {ph} ran to the end", False, repr(e)[:200])
        check(f"after phase {ph}: node alive", alive())
    fails = [r for r in results if r[0] == "FAIL"]
    notes = [r for r in results if r[0] == "NOTE"]
    print(f"\n{len(results)} checks in {(time.time() - t0) / 60:.1f} min: {len(results) - len(fails) - len(notes)} pass, "
          f"{len(fails)} FAIL, {len(notes)} note")
    for tag, name, detail in fails + notes:
        print(f"  {tag}  {name}  -- {detail}")


if __name__ == "__main__":
    main()
