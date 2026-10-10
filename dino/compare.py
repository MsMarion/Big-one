"""Aggregate runs into a comparison table + bar plot.

    python -m dino.compare [--glob 'smoke_*'] [--out $DINO_RUNS/_compare]
"""
import argparse
import csv
import fnmatch
import json
import os

from tabulate import tabulate

from .config import runs_root


def load(root, pattern):
    rows = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name, "metrics.json")
        if not fnmatch.fnmatch(name, pattern) or not os.path.exists(p):
            continue
        with open(p) as f:
            m = json.load(f)
        bench = m.get("eval", {}).get("benchmark", {})
        rows.append({
            "run": m["name"], "task": m["task"], "backbone": m["backbone"], "unfreeze": m["unfreeze_last"],
            "metric": m["primary_metric"], "best": round(m["best"], 4),
            "eval": round(m["eval"]["metrics"][m["primary_metric"]], 4) if "eval" in m else None,
            "train_img/s": round(m["train_imgs_per_s"]),
            "infer_ms@bs1": round(bench["bs1"]["ms_per_batch"], 1) if bench else None,
            "infer_img/s@bs32": round(bench["bs32"]["imgs_per_s"]) if bench else None,
            "peak_GB": round(m["peak_vram_gb"], 2), "train_s": round(m["train_time_s"]),
            "params_M": round(m["total_params"] / 1e6, 1),
        })
    return rows


def plot(rows, path):
    import matplotlib.pyplot as plt

    tasks = sorted({r["task"] for r in rows})
    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 3.5), squeeze=False)
    for ax, t in zip(axes[0], tasks):
        rs = [r for r in rows if r["task"] == t]
        ax.barh([r["run"] for r in rs], [r["best"] for r in rs], color="#4C78A8")
        ax.set_title(f"{t}: {rs[0]['metric']}")
        ax.set_xlim(0, 1)
        for i, r in enumerate(rs):
            ax.text(r["best"] + 0.01, i, f"{r['best']:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=runs_root())
    ap.add_argument("--glob", default="*")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.root, "_compare")
    rows = load(a.root, a.glob)
    if not rows:
        raise SystemExit(f"no runs matching {a.glob!r} in {a.root}")
    os.makedirs(out, exist_ok=True)
    table = tabulate(rows, headers="keys", tablefmt="github")
    print(table)
    with open(os.path.join(out, "compare.md"), "w") as f:
        f.write(table + "\n")
    with open(os.path.join(out, "compare.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    plot(rows, os.path.join(out, "compare.png"))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
