"""One command for "new data -> train -> test benchmark -> report".

    python -m dino.experiment experiments/<name>.yaml [--force] [--report-only]

For every task x backbone in the experiment file it trains a run (skipped if an identical run
already finished, unless --force), evaluates it on the test split, exports sample predictions,
and benchmarks any `baseline_runs` on the same test split. Then it writes
    /workspace/dino/experiments/<name>/{experiment.yaml, dataset.json, results.json, configs/, eval/}
    /workspace/dino/reports/<name>.html   (self-contained; open in JupyterLab: dino/reports/<name>.html)
Each run is a subprocess, so one failure doesn't stop the others.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import yaml

from .config import DEFAULTS, _expand, _merge, runs_root
from .data.split import make_splits
from .data.validate import summarize
from .model import write_json, write_yaml

ROOT = os.environ.get("DINO_ROOT", "/workspace/dino")


def exp_dir(name):
    return os.path.join(ROOT, "experiments", name)


def load_experiment(path):
    with open(path) as f:
        e = _expand(yaml.safe_load(f))
    e.setdefault("tasks", ["cls"])
    e.setdefault("backbones", ["dinov2_s"])
    e.setdefault("baseline_runs", [])
    e.setdefault("infer_samples", 12)
    d = e["data"]
    d.setdefault("train", "train")
    d.setdefault("val", "val")
    if not d.get("test"):
        d["test"] = "test" if (d.get("split") or os.path.isdir(os.path.join(d["root"], "test"))) else d["val"]
    return e


def run_name(e, task, bb):
    return f"{e['name']}__{task}__{bb}"


def run_config(e, task, bb):
    cfg = _merge(DEFAULTS, e.get("common", {}))
    cfg = _merge(cfg, e.get(task, {}))
    cfg.update(name=run_name(e, task, bb), task=task, backbone=bb)
    cfg["data"] = {"root": e["data"]["root"], "train": e["data"]["train"], "val": e["data"]["val"]}
    return cfg


def sh(args, log):
    """Run a module as a subprocess, tee output to the experiment log. Returns (ok, tail)."""
    print(f"$ {' '.join(args)}", flush=True)
    p = subprocess.Popen([sys.executable, "-m"] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    for line in p.stdout:
        print(line, end="", flush=True)
        log.write(line)
        lines.append(line)
    log.flush()
    return p.wait() == 0, "".join(lines[-25:])


def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--force", action="store_true", help="retrain even if an identical finished run exists")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()

    e = load_experiment(a.experiment)
    ed = exp_dir(e["name"])
    os.makedirs(os.path.join(ed, "configs"), exist_ok=True)
    os.makedirs(os.path.join(ed, "eval"), exist_ok=True)
    write_yaml(os.path.join(ed, "experiment.yaml"), e)

    from .report import write_report  # after env is set up; report only needs stdlib + PIL

    if a.report_only:
        print(write_report(e["name"]))
        return

    log = open(os.path.join(ed, "experiment.log"), "a")
    t0 = time.time()
    status = {"name": e["name"], "started": t0, "state": "running", "runs": {}}
    write_json(os.path.join(ed, "status.json"), status)

    # 1. data: optional auto-split, then validate
    d = e["data"]
    if d.get("split"):
        sp = d["split"]
        if a.force or not all(os.path.isdir(os.path.join(d["root"], s)) for s in ("train", "val", "test")):
            counts = make_splits(d["root"], sp.get("from", "all"), sp.get("ratios", (0.7, 0.15, 0.15)), sp.get("seed", 0))
            print(f"[exp] split {sp.get('from', 'all')} -> {counts}", flush=True)
    summary = summarize(d["root"], [d["train"], d["val"], d["test"]], e["tasks"])
    write_json(os.path.join(ed, "dataset.json"), summary)
    for w in summary["warnings"]:
        print(f"[exp] warning: {w}", flush=True)
    if summary["errors"]:
        status.update(state="failed", error="dataset validation failed: " + "; ".join(summary["errors"][:5]))
        write_json(os.path.join(ed, "status.json"), status)
        write_report(e["name"])
        sys.exit("[exp] dataset validation failed:\n  " + "\n  ".join(summary["errors"]))

    # 2. train + test-benchmark each task x backbone
    results = []
    jobs = [(t, bb) for t in e["tasks"] for bb in e["backbones"]]
    for i, (task, bb) in enumerate(jobs, 1):
        name = run_name(e, task, bb)
        rd = os.path.join(runs_root(), name)
        cfg = run_config(e, task, bb)
        cfg_path = os.path.join(ed, "configs", name + ".yaml")
        prev = _load_json(os.path.join(rd, "metrics.json"))
        same = os.path.exists(cfg_path) and yaml.safe_load(open(cfg_path)) == cfg
        write_yaml(cfg_path, cfg)
        r = {"run": name, "task": task, "backbone": bb, "baseline": False}
        print(f"\n[exp] ({i}/{len(jobs)}) {name}", flush=True)
        status["runs"][name] = "training"
        write_json(os.path.join(ed, "status.json"), status)

        if prev and same and not a.force:
            print("[exp] identical finished run found, skipping training", flush=True)
            ok = True
        else:
            ok, tail = sh(["dino.train", "--config", cfg_path], log)
            if not ok:
                r.update(state="failed", stage="train", error=tail)
        if ok:
            status["runs"][name] = "testing"
            write_json(os.path.join(ed, "status.json"), status)
            ok, tail = sh(["dino.evaluate", "--run", rd, "--data_root", d["root"], "--split", d["test"],
                           "--out", os.path.join(ed, "eval", name + ".json")], log)
            if not ok:
                r.update(state="failed", stage="test", error=tail)
        if ok:
            out = os.path.join(rd, "test_preds")
            sh(["dino.infer", "--run", rd, "--input", os.path.join(d["root"], d["test"]), "--out", out,
                "--limit", str(e["infer_samples"])], log)
            r.update(state="ok", preds=out)
        status["runs"][name] = r["state"]
        results.append(r)
        write_json(os.path.join(ed, "results.json"), results)
        write_report(e["name"])  # partial report after every run

    # 3. baselines: existing runs benchmarked on this test split
    for name in e["baseline_runs"]:
        rd = os.path.join(runs_root(), name)
        print(f"\n[exp] baseline {name}", flush=True)
        ok, tail = sh(["dino.evaluate", "--run", rd, "--data_root", d["root"], "--split", d["test"],
                       "--out", os.path.join(ed, "eval", name + ".json")], log)
        m = _load_json(os.path.join(rd, "metrics.json")) or {}
        results.append({"run": name, "task": m.get("task"), "backbone": m.get("backbone"), "baseline": True,
                        "state": "ok" if ok else "failed", "stage": None if ok else "test", "error": None if ok else tail})
        write_json(os.path.join(ed, "results.json"), results)

    status.update(state="done", finished=time.time(), duration_s=time.time() - t0)
    write_json(os.path.join(ed, "status.json"), status)
    path = write_report(e["name"])
    n_ok = sum(r["state"] == "ok" for r in results)
    print(f"\n[exp] done: {n_ok}/{len(results)} ok in {(time.time() - t0) / 60:.1f} min\n[exp] report: {path}", flush=True)
    print(f"[exp] JupyterLab: open dino/reports/{e['name']}.html", flush=True)


if __name__ == "__main__":
    main()
