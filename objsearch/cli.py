"""objsearch command line.

    python -m objsearch db init | import [catalog/seed.yaml] [--update] | export [path] | stats | reset --yes
    python -m objsearch gallery [--out file.html] [--filter owned]
    python -m objsearch demo [--fail-at identify|acquire|convert] | demo --clean
    python -m objsearch refimages [--ids a b] [--n 3] [--force] [--query "text"]
    python -m objsearch grasp [--verdict no]
    python -m objsearch identify photo.jpg [more.jpg | folder/] [--hint ".."] [--dims L W H] [--confirm yes|no]
    python -m objsearch parts list | index | add <key> <step|url> --license .. [--footprint Lib:Name] | rebuild
    python -m objsearch acquire <object_id>... [--no-agent] [--force]
    python -m objsearch models check
    python -m objsearch quota
"""
import argparse
import json
import os
import sys

from . import config, db, track

SEED = os.path.join(config.ROOT, "catalog", "seed.yaml")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="objsearch")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("db", help="database admin")
    p.add_argument("action", choices=["init", "import", "export", "stats", "reset"])
    p.add_argument("path", nargs="?")
    p.add_argument("--update", action="store_true", help="import: overwrite fields of existing ids")
    p.add_argument("--yes", action="store_true", help="reset: confirm clearing the inventory")

    p = sub.add_parser("gallery", help="write the static gallery HTML")
    p.add_argument("--out")
    p.add_argument("--filter", choices=sorted(db.FLAGS))
    p.add_argument("--grip", choices=["good", "borderline", "no", "ok"], help="only items the gripper can/can't pick")

    p = sub.add_parser("grasp", help="SO-ARM101 graspability report (limits in configs/objsearch.yaml -> gripper)")
    p.add_argument("--verdict", choices=["good", "borderline", "no"])

    p = sub.add_parser("demo", help="fake end-to-end run (no keys, no Blender)")
    p.add_argument("--fail-at", choices=["identify", "acquire", "convert"])
    p.add_argument("--clean", action="store_true", help="remove demo runs and demo-* objects")

    p = sub.add_parser("refimages", help="fetch openly licensed reference images (Openverse) for objects")
    p.add_argument("--ids", nargs="*", help="object ids (default: all objects without reference images)")
    p.add_argument("--n", type=int, default=3, help="images per object")
    p.add_argument("--force", action="store_true", help="also objects that already have images")
    p.add_argument("--query", help="override the search text")

    p = sub.add_parser("identify", help="identify ONE object from photos (files, or a folder of angles)")
    p.add_argument("paths", nargs="+")
    p.add_argument("--hint", help="anything you know: 'blue board, says ESP-01', 'from the robot kit'...")
    p.add_argument("--dims", nargs=3, type=float, metavar=("L", "W", "H"), help="measured size in mm")
    p.add_argument("--confirm", choices=["yes", "no"], help="answer right away instead of in the dashboard")
    p.add_argument("--no-acquire", action="store_true", help="don't fetch a 3D model after identifying")

    p = sub.add_parser("model", help="attach a 3D model (URL or file) to an object: convert + Blender -> USD")
    p.add_argument("action", choices=["add"])
    p.add_argument("object_id")
    p.add_argument("src", help="URL or local path (.step/.stp/.glb/.gltf/.obj/.stl/.ply/.fbx)")
    p.add_argument("--license", help="license of the model (stored with every asset)")
    p.add_argument("--source-url", help="page the model came from")

    p = sub.add_parser("parts", help="internal 3D parts library (what board builds use / still lack)")
    p.add_argument("action", choices=["list", "index", "add", "rebuild"])
    p.add_argument("key", nargs="?", help="add: part key (KiCad model path w/o ext, or eagle:<footprint>)")
    p.add_argument("src", nargs="?", help="add: STEP file or URL")
    p.add_argument("--license")
    p.add_argument("--source-url")
    p.add_argument("--footprint", help="add: KiCad footprint Lib:Name to auto-align the STEP to")
    p.add_argument("--object", help="rebuild: only this object (default: all objects that used approximations)")

    p = sub.add_parser("acquire", help="find + build a 3D model automatically (known source, KiCad, open libs, agent)")
    p.add_argument("object_ids", nargs="+")
    p.add_argument("--no-agent", action="store_true", help="skip the LLM research step")
    p.add_argument("--force", action="store_true", help="even if the object already has a model")

    p = sub.add_parser("docs", help="datasheet + physical details (mass, materials, friction) -> physics + sim.usda")
    p.add_argument("object_ids", nargs="+")
    p.add_argument("--force", action="store_true", help="research again even if done before")
    p = sub.add_parser("sim", help="(re)build the sim-ready USD layer(s): physics + semantic labels")
    p.add_argument("object_ids", nargs="*", help="default: every object with a model")
    p = sub.add_parser("export", help="objects -> folder for DINOv3 / FoundationPose / Isaac Sim (mesh, USD, info)")
    p.add_argument("out_dir")
    p.add_argument("--ids", nargs="*")
    p.add_argument("--owned", action="store_true", help="only items you own")
    p.add_argument("--real-only", action="store_true", help="leave out placeholder boxes")
    p = sub.add_parser("models", help="LLM model availability")
    p.add_argument("action", choices=["check"])

    sub.add_parser("quota", help="today's free-tier usage")

    a = ap.parse_args(argv)
    conn = db.connect()

    if a.cmd == "db":
        if a.action == "init":
            print(f"db ready: {config.db_path()} (schema v{conn.execute('PRAGMA user_version').fetchone()[0]})")
        elif a.action == "import":
            print(db.import_seed(conn, a.path or SEED, update=a.update))
        elif a.action == "reset":
            if not a.yes:
                print("This clears your inventory (owned items, photos, runs, models, added objects) and keeps the "
                      "catalog, learned model sources, parts library, reference images and build cache. "
                      "Re-run with --yes.")
                return 1
            print(db.reset_inventory(conn))
        elif a.action == "export":
            path = a.path or SEED
            db.export_seed(conn, path)
            print(f"wrote {path}")
        else:
            s = db.summary(conn)
            print(json.dumps({k: v for k, v in s.items() if k != "categories"}, indent=1))
    elif a.cmd == "gallery":
        from . import gallery
        print(f"wrote {gallery.write(conn, a.out, a.filter, a.grip)}")
    elif a.cmd == "grasp":
        from . import grasp
        g = grasp.limits()
        print(f"gripper: >= {g['min_height_mm']:g} mm tall, {g['min_width_mm']:g}-{g['max_open_mm'] - g['clearance_mm']:g} mm "
              f"across the jaws (opens ~{g['max_open_mm']:g} mm), borderline band {g['tolerance_mm']:g} mm")
        objs = grasp.annotate(db.list_objects(conn))
        for v in ("good", "borderline", "no"):
            sel = [o for o in objs if o["grasp"]["verdict"] == v and a.verdict in (None, v)]
            if sel:
                print(f"\n{v} ({len(sel)})")
            for o in sel:
                gr = o["grasp"]
                print(f"  {o['id']:<30} {gr['height_mm']:>5g} h {gr['grasp_width_mm']:>5g} w  {'; '.join(gr['reasons'])}")
    elif a.cmd == "demo":
        from . import demo
        if a.clean:
            print(f"removed {demo.clean(conn)} demo runs")
        else:
            rid = demo.run_demo(conn, a.fail_at)
            r = db.get_run(conn, rid)
            print(f"run {rid}: {r['status']}, {len(r['steps'])} steps, {len(r['llm_calls'])} llm calls, "
                  f"{len(r['tool_calls'])} tool calls -> http://127.0.0.1:8765/objects/runs/{rid}")
    elif a.cmd == "refimages":
        from . import refimages
        print(refimages.fetch_all(conn, a.ids, n=a.n, force=a.force, query=a.query))
    elif a.cmd == "identify":
        from . import identify
        files = identify.expand_paths(a.paths)
        st = identify.backend_status()
        print(f"identifying from {len(files)} photo(s) with {st['mode']} ({st['model']})")
        res = identify.identify(conn, files, hint=a.hint, dims_mm=list(a.dims) if a.dims else None)
        print(json.dumps({k: res.get(k) for k in ("match", "object_id", "name", "confidence", "reasoning",
                                                   "evidence_urls", "stop_reason", "run_id")}, indent=1))
        from . import acquire
        if res.get("object_id") and not a.no_acquire and acquire.auto_enabled():
            import time as _t
            t0 = _t.time()
            out = acquire.acquire(conn, res["object_id"])
            print(f"3D model: {out['how'] or 'none found'} ({_t.time() - t0:.1f} s)")
            from . import docs
            if docs.auto_enabled():
                t0 = _t.time()
                d = docs.safe_research(conn, res["object_id"])
                print(f"docs: {'datasheet saved' if d.get('datasheet') else 'no datasheet'}, "
                      f"mass {(d.get('physics') or {}).get('mass_g', '?')} g ({_t.time() - t0:.1f} s)"
                      + (f" | {d.get('skipped') or d.get('error')}" if d.get('skipped') or d.get('error') else ""))
        if a.confirm:
            print(identify.confirm(conn, res["run_id"], a.confirm == "yes"))
        else:
            print(f"review / confirm: http://127.0.0.1:8765/objects/identify?run={res['run_id']}")
    elif a.cmd == "model":
        from . import models
        res = models.add_model(conn, a.object_id, a.src, license=a.license, source_url=a.source_url)
        print(json.dumps({k: res.get(k) for k in ("dims_mm", "scale_factor", "dims_fit", "tri_count", "parts", "flags",
                                                   "files", "out_dir", "run_id")}, indent=1))
        print(f"view: http://127.0.0.1:8765/objects/{a.object_id}")
    elif a.cmd == "parts":
        from . import kicad, models, parts
        if a.action == "list":
            for p_ in parts.listing(conn):
                print(f"{p_['status']:<11} {p_['key']:<60} used by {', '.join(p_.get('seen_in') or []) or '-'}")
        elif a.action == "index":
            print(f"indexed {len(kicad.footprint_index(rebuild=True))} KiCad footprints")
        elif a.action == "add":
            src = a.src
            if src and src.startswith(("http://", "https://")):
                with track.run(conn, "parts", notes=f"fetch {a.key}"):
                    src, _, _ = models.fetch("_parts", src)
            p_ = parts.add_real(conn, a.key, src, a.license or "unknown", a.source_url or a.src, footprint=a.footprint)
            print(p_["status"], p_["path"], p_["notes"])
        else:
            for oid in [a.object] if a.object else parts.affected_objects(conn):
                try:
                    r_ = models.rebuild(conn, oid)
                    print(oid, "rebuilt:", r_.get("flags") or "no flags")
                except (ValueError, KeyError) as e:
                    print(oid, "skipped:", e)
    elif a.cmd == "acquire":
        import time as _t
        from . import acquire
        for oid in a.object_ids:
            t0 = _t.time()
            out = acquire.acquire(conn, oid, use_agent=not a.no_agent, force=a.force)
            res = out.get("result") or {}
            print(f"{oid}: {out['how'] or 'nothing found'} in {_t.time() - t0:.1f} s"
                  + (f" | {res.get('dims_mm')} mm | flags: {len(res.get('flags') or [])}" if res else "")
                  + (f" | tried: {', '.join(out['tried'])}" if out.get("tried") else ""))
    elif a.cmd == "docs":
        import time as _t
        from . import docs
        for oid in a.object_ids:
            t0 = _t.time()
            d = docs.research(conn, oid, force=a.force)
            ph = d.get("physics") or {}
            print(f"{oid}: {d.get('skipped') or ''} datasheet={'yes' if d.get('datasheet') else 'no'} "
                  f"mass={ph.get('mass_g', '-')} g ({ph.get('mass_source', '-')}) "
                  f"friction={ph.get('static_friction', '-')} in {_t.time() - t0:.1f} s")
    elif a.cmd == "sim":
        from . import sim
        ids = a.object_ids or [r[0] for r in conn.execute("SELECT DISTINCT object_id FROM assets WHERE kind='usd'")]
        with track.run(conn, "sim", notes=f"sim layers for {len(ids)}"):
            for oid in ids:
                r_ = sim.safe_build(conn, oid)
                print(oid, f"{r_['path']} | {r_['colliders']} colliders | {r_['mass_g']} g ({r_['mass_source']})"
                      if r_ else "no model / failed (see run log)")
    elif a.cmd == "export":
        from . import export
        r_ = export.export(conn, a.out_dir, ids=a.ids, owned=a.owned, real_only=a.real_only)
        print(f"exported {len(r_['classes'])} objects to {r_['out_dir']} | quality {r_['quality']}"
              + (f" | skipped {r_['skipped']}" if r_["skipped"] else ""))
    elif a.cmd == "models":
        from .llm import gemini_backend
        probe = gemini_backend.check_key()
        print(f"key check: {'OK' if probe['ok'] else 'PROBLEM'} - {probe['message']}")
        have = set(gemini_backend.list_models())
        for role, m in config.load()["models"]["gemini"].items():
            print(f"{role:<9} {m:<34} {'ok' if m in have else 'NOT AVAILABLE to this key'}")
        print(f"\n{len(have)} models available; flash/lite ones: " +
              ", ".join(sorted(m for m in have if "flash" in m or "lite" in m)[:20]))
    elif a.cmd == "quota":
        print(json.dumps(track.quota_status(conn), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
