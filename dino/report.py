"""Static, self-contained HTML experiment report (no JavaScript: renders in JupyterLab's sandboxed viewer).

    python -m dino.report <experiment_name>        # rewrites reports/<name>.html and reports/index.html
"""
import argparse
import base64
import csv
import glob
import html
import io
import json
import os
import time

import numpy as np
import yaml
from PIL import Image

ROOT = os.environ.get("DINO_ROOT", "/workspace/dino")
RUNS = os.environ.get("DINO_RUNS", os.path.join(ROOT, "runs"))
REPORTS = os.path.join(ROOT, "reports")
SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated categorical slots 1-4
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500"]
MAX_LINES = 4
PRIMARY = {"cls": "acc", "seg": "miou"}
SECONDARY = {"cls": ("macro_acc", "macro acc"), "seg": ("miou_fg", "mIoU (objects only)")}
TASK_NAME = {"cls": "Classification", "seg": "Segmentation"}
e_ = html.escape

CSS = """
:root{color-scheme:light;--s0:#f4f3f0;--s1:#fcfcfb;--bd:#e2e1dc;--t1:#0b0b0b;--t2:#52514e;--t3:#8a8984;--grid:#ecebe7;
%LIGHT%--good:#0ca30c;--warning:#fab219;--critical:#d03b3b}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;--s0:#111110;--s1:#1a1a19;--bd:#2e2e2b;
--t1:#fff;--t2:#c3c2b7;--t3:#8d8c84;--grid:#272724;%DARK%}}
:root[data-theme="dark"]{color-scheme:dark;--s0:#111110;--s1:#1a1a19;--bd:#2e2e2b;--t1:#fff;--t2:#c3c2b7;--t3:#8d8c84;--grid:#272724;%DARK%}
*{box-sizing:border-box}body{margin:0;background:var(--s0);color:var(--t1);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1200px;margin:0 auto;padding:20px 16px 48px;display:grid;gap:16px}
h1{font-size:20px;margin:0}h2{font-size:15px;margin:0 0 10px}h3{font-size:13px;margin:14px 0 6px;color:var(--t2)}
.card{background:var(--s1);border:1px solid var(--bd);border-radius:10px;padding:16px;min-width:0}
.muted{color:var(--t3)}.sec{color:var(--t2)}
.meta{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13px;color:var(--t2);margin-top:6px}
.badge{display:inline-flex;align-items:center;gap:6px;font-size:13px;font-weight:600}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}
.tile .v{font-size:28px;font-weight:650;font-variant-numeric:tabular-nums}.tile .s{font-size:12px;color:var(--t3)}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--bd);white-space:nowrap}th{color:var(--t2);font-weight:600;font-size:12px}
td.num,th.num{text-align:right}tr.best td{font-weight:650}
.meter{display:inline-block;width:70px;height:7px;border-radius:4px;background:var(--grid);vertical-align:middle;margin-left:6px;overflow:hidden}
.meter i{display:block;height:100%;background:var(--c1);border-radius:4px}
svg{display:block;width:100%;height:auto}svg text{fill:var(--t3);font-size:11px}svg .lbl{fill:var(--t1);font-size:12px}
.legend{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12px;color:var(--t2);margin-bottom:4px}
.legend span::before{content:"";display:inline-block;width:12px;height:3px;border-radius:2px;background:var(--c);vertical-align:middle;margin-right:6px}
.err{border-left:3px solid var(--critical);padding:8px 12px;background:var(--s0);border-radius:6px;margin:6px 0;font-size:13px}
.warn{border-left:3px solid var(--warning);padding:6px 12px;background:var(--s0);border-radius:6px;margin:4px 0;font-size:13px}
pre{background:var(--s0);border:1px solid var(--bd);border-radius:6px;padding:10px;overflow:auto;font-size:12px;max-height:360px;white-space:pre-wrap}
.gal{display:grid;gap:8px}.gal .row{display:grid;gap:6px;align-items:start}
figure{margin:0}figure img{width:100%;display:block;border-radius:4px}figcaption{font-size:11px;color:var(--t2);margin-top:2px}
.cls-gal{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px}
details summary{cursor:pointer;color:var(--t2);font-size:13px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px}
@media (max-width:520px){.grid2{grid-template-columns:1fr}}
"""


def _css():
    light = "".join(f"--c{i+1}:{c};" for i, c in enumerate(SERIES_LIGHT))
    dark = "".join(f"--c{i+1}:{c};" for i, c in enumerate(SERIES_DARK))
    return CSS.replace("%LIGHT%", light).replace("%DARK%", dark)


def _json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def fmt(v, d=3):
    return "–" if v is None else f"{v:.{d}f}"


def status_badge(state):
    level, icon, text = {"done": ("good", "✓", "Done"), "running": ("warning", "…", "Running"),
                         "failed": ("critical", "✕", "Failed"), "ok": ("good", "✓", "OK")}.get(state, ("warning", "?", state))
    return f'<span class="badge"><span class="dot" style="background:var(--{level})"></span><span aria-hidden="true">{icon}</span>{e_(text)}</span>'


# ---------- images ----------
def _b64(img, width=220, q=78):
    img = img.convert("RGB")
    if img.width > width:
        img = img.resize((width, round(img.height * width / img.width)), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _overlay(rgb_path, fg):
    img = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.float32)
    if fg.shape != img.shape[:2]:
        fg = np.asarray(Image.fromarray(fg.astype(np.uint8) * 255).resize(img.shape[1::-1], Image.NEAREST)) > 0
    tint = np.array([42, 120, 214], np.float32)
    img[fg] = img[fg] * 0.45 + tint * 0.55
    # outline for shape legibility
    edge = fg ^ np.pad(fg, 1, mode="edge")[1:-1, 2:] | fg ^ np.pad(fg, 1, mode="edge")[2:, 1:-1]
    img[edge] = (255, 255, 255)
    return Image.fromarray(img.clip(0, 255).astype(np.uint8))


def _find_rgb(split_dir, fid):
    for p in glob.glob(os.path.join(split_dir, "rgb", glob.escape(fid) + ".*")):
        return p
    return None


# ---------- charts (inline SVG, single axis, thin marks, direct labels) ----------
def bar_chart(rows, vmax=1.0):
    """rows: [(label, value)] -> horizontal single-series bars with value labels."""
    W, bh, gap, L, R = 640, 22, 10, 190, 60
    H = len(rows) * (bh + gap) + 24
    x = lambda v: L + (W - L - R) * (0 if v is None else v) / vmax
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="bar chart">']
    for f in (0, .25, .5, .75, 1):
        gx = L + (W - L - R) * f
        s.append(f'<line x1="{gx:.1f}" x2="{gx:.1f}" y1="0" y2="{H-18}" stroke="var(--grid)"/>'
                 f'<text x="{gx:.1f}" y="{H-4}" text-anchor="middle">{f*vmax:g}</text>')
    for i, (lab, v) in enumerate(rows):
        y = i * (bh + gap) + 4
        w = max(0.0, x(v) - L)
        r = min(4, w)
        s.append(f'<text class="lbl" x="{L-8}" y="{y+bh/2+4}" text-anchor="end">{e_(lab)}</text>')
        if w > 0:
            s.append(f'<path d="M{L},{y}h{w-r:.1f}a{r},{r} 0 0 1 {r},{r}v{bh-2*r}a{r},{r} 0 0 1 -{r},{r}h-{w-r:.1f}z" '
                     f'fill="var(--c1)"><title>{e_(lab)}: {fmt(v, 4)}</title></path>')
        s.append(f'<text class="lbl" x="{L+w+6:.1f}" y="{y+bh/2+4}">{fmt(v)}</text>')
    s.append("</svg>")
    return "".join(s)


def line_chart(series, ylabel):
    """series: [(name, color_var, [y per epoch])] -> multi-series lines, legend + end labels."""
    series = [s for s in series if s[2]]
    if not series:
        return '<p class="muted">no history</p>'
    W, H, L, R, T, B = 640, 230, 42, 120, 10, 26
    n = max(len(s[2]) for s in series)
    ys = [v for s in series for v in s[2] if v is not None]
    lo = max(0.0, min(ys) - 0.05)
    hi = min(1.0, max(ys) + 0.02) if max(ys) <= 1 else max(ys) * 1.05
    hi = hi if hi > lo else lo + 1
    x = lambda i: L + (W - L - R) * (i / max(1, n - 1))
    y = lambda v: T + (H - T - B) * (1 - (v - lo) / (hi - lo))
    out = ['<div class="legend">' + "".join(f'<span style="--c:var({c})">{e_(nm)}</span>' for nm, c, _ in series) + "</div>",
           f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{e_(ylabel)} per epoch">']
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        out.append(f'<line x1="{L}" x2="{W-R}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="var(--grid)"/>'
                   f'<text x="{L-6}" y="{y(v)+4:.1f}" text-anchor="end">{v:.2f}</text>')
    for i in range(n):
        if n <= 12 or i % max(1, n // 10) == 0 or i == n - 1:
            out.append(f'<text x="{x(i):.1f}" y="{H-8}" text-anchor="middle">{i+1}</text>')
    out.append(f'<text x="{W-R}" y="{H-8}" text-anchor="start" dx="8">epoch</text>')
    ends = []
    for nm, c, vals in series:
        pts = [(x(i), y(v), v, i) for i, v in enumerate(vals) if v is not None]
        d = "".join(f"{'M' if j == 0 else 'L'}{px:.1f},{py:.1f}" for j, (px, py, _, _) in enumerate(pts))
        out.append(f'<path d="{d}" fill="none" stroke="var({c})" stroke-width="2" stroke-linejoin="round"/>')
        for px, py, v, i in pts:
            out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="var({c})" stroke="var(--s1)" stroke-width="2">'
                       f'<title>{e_(nm)} · epoch {i+1}: {v:.4f}</title></circle>')
        ends.append([pts[-1][1], nm, c, pts[-1][2]])
    ends.sort()
    for k in range(1, len(ends)):  # de-collide end labels
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + 14)
    for ey, nm, c, v in ends:
        out.append(f'<text class="lbl" x="{W-R+8}" y="{ey+4:.1f}">{e_(nm)} {v:.3f}</text>')
    out.append("</svg>")
    return "".join(out)


def confusion_svg(conf, classes):
    n = len(classes)
    cell = max(16, min(40, 520 // max(n, 1)))
    L, T = 110, 20
    W, H = L + n * cell + 10, T + n * cell + 40
    out = [f'<svg viewBox="0 0 {W} {H}" style="max-width:{W}px" role="img" aria-label="confusion matrix">']
    for i, row in enumerate(conf):
        tot = sum(row) or 1
        out.append(f'<text class="lbl" x="{L-6}" y="{T+i*cell+cell/2+4}" text-anchor="end">{e_(classes[i][:14])}</text>')
        for j, v in enumerate(row):
            f = v / tot
            out.append(f'<rect x="{L+j*cell+1}" y="{T+i*cell+1}" width="{cell-2}" height="{cell-2}" rx="2" '
                       f'fill="var(--c1)" fill-opacity="{0.06 + 0.94*f:.3f}"><title>true {e_(classes[i])} → pred {e_(classes[j])}: {v} ({f:.0%})</title></rect>')
            if cell >= 24 and v:
                out.append(f'<text x="{L+j*cell+cell/2}" y="{T+i*cell+cell/2+4}" text-anchor="middle" '
                           f'style="fill:{"#fff" if f > .5 else "var(--t1)"};font-size:10px">{v}</text>')
    for j, c in enumerate(classes):
        out.append(f'<text x="{L+j*cell+cell/2}" y="{T+n*cell+14}" text-anchor="middle">{e_(c[:6])}</text>')
    out.append(f'<text x="{L+n*cell/2}" y="{H-4}" text-anchor="middle">predicted →</text>'
               f'<text x="{L-6}" y="12" text-anchor="end">true ↓</text></svg>')
    return "".join(out)


# ---------- sections ----------
def gather(name):
    ed = os.path.join(ROOT, "experiments", name)
    with open(os.path.join(ed, "experiment.yaml")) as f:
        e = yaml.safe_load(f)
    results = _json(os.path.join(ed, "results.json")) or []
    for r in results:
        r["metrics"] = _json(os.path.join(RUNS, r["run"], "metrics.json")) or {}
        r["eval"] = _json(os.path.join(ed, "eval", r["run"] + ".json"))
    return e, ed, results, _json(os.path.join(ed, "dataset.json")) or {}, _json(os.path.join(ed, "status.json")) or {}


def section_task(e, task, rs, classes):
    key = PRIMARY[task]
    sk, sl = SECONDARY[task]
    ok = [r for r in rs if r.get("eval")]
    if not ok:
        return ""
    ok.sort(key=lambda r: -r["eval"]["metrics"][key])
    best = ok[0]
    colors = {bb: f"--c{i+1}" for i, bb in enumerate(e["backbones"][:MAX_LINES])}
    rows = []
    for r in ok:
        m, ev, b = r["metrics"], r["eval"], r["eval"]["benchmark"]
        tag = ' <span class="muted">(baseline)</span>' if r["baseline"] else ""
        train_t = "–" if r["baseline"] else f'{m.get("train_time_s", 0):.0f} s'
        rows.append(
            f'<tr class="{"best" if r is best else ""}"><td>{e_(r["backbone"] or "?")}{tag}</td>'
            f'<td class="num">{fmt(ev["metrics"][key])}<span class="meter"><i style="width:{ev["metrics"][key]*100:.1f}%"></i></span></td>'
            f'<td class="num">{fmt(ev["metrics"].get(sk))}</td><td class="num">{fmt(m.get("best"))}</td>'
            f'<td class="num">{fmt(b["bs1"]["ms_per_batch"], 1)} ms</td><td class="num">{b["bs32"]["imgs_per_s"]:.0f}</td>'
            f'<td class="num">{fmt(b["bs32"]["peak_vram_gb"], 2)} GB</td>'
            f'<td class="num">{train_t}</td>'
            f'<td class="num">{(m.get("total_params") or 0)/1e6:.1f}M</td><td class="muted">{e_(r["run"])}</td></tr>')
    n_test = best["eval"]["n"]
    h = [f'<section class="card"><h2>{TASK_NAME[task]} · test benchmark <span class="muted">({n_test} test images, metric: {key})</span></h2>',
         '<div class="scroll"><table><thead><tr><th>Backbone</th><th class="num">Test ' + key + f'</th><th class="num">Test {sl}</th>'
         '<th class="num">Val best</th><th class="num">Latency bs1</th><th class="num">img/s bs32</th><th class="num">VRAM bs32</th>'
         '<th class="num">Train time</th><th class="num">Params</th><th>Run</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>"]
    h.append('<div class="grid2">')
    h.append(f'<div><h3>Test {key} by backbone</h3>{bar_chart([(r["backbone"] + (" (base)" if r["baseline"] else ""), r["eval"]["metrics"][key]) for r in ok])}</div>')
    curves = [(r["backbone"], colors[r["backbone"]], [hh.get(key) for hh in r["metrics"].get("history", [])])
              for r in rs if not r["baseline"] and r["backbone"] in colors and r.get("metrics")]
    more = len([r for r in rs if not r["baseline"]]) - len(curves)
    h.append(f'<div><h3>Validation {key} per epoch</h3>{line_chart(curves, key)}'
             + (f'<p class="muted">+{more} more backbones in the table</p>' if more > 0 else "") + "</div></div>")

    per_key = "per_class_iou" if task == "seg" else "per_class_acc"
    names = list(best["eval"]["metrics"][per_key].keys())
    h.append(f'<h3>Per-class test {"IoU" if task == "seg" else "accuracy"}</h3><div class="scroll"><table><thead><tr><th>Class</th>'
             + "".join(f'<th class="num">{e_(r["backbone"])}{" (base)" if r["baseline"] else ""}</th>' for r in ok) + "</tr></thead><tbody>")
    for c in names:
        h.append(f"<tr><td>{e_(c)}</td>" + "".join(
            f'<td class="num">{fmt(r["eval"]["metrics"][per_key].get(c))}<span class="meter"><i style="width:{(r["eval"]["metrics"][per_key].get(c) or 0)*100:.1f}%"></i></span></td>'
            for r in ok) + "</tr>")
    h.append("</tbody></table></div>")
    if task == "cls" and len(names) <= 25:
        h.append(f'<h3>Confusion matrix · {e_(best["backbone"])} (test)</h3>{confusion_svg(best["eval"]["metrics"]["confusion"], names)}')
    h.append(samples_seg(e, ok, classes) if task == "seg" else samples_cls(e, best))
    h.append("</section>")
    return "".join(h)


def samples_seg(e, ok, classes, n=6):
    test_dir = os.path.join(e["data"]["root"], e["data"]["test"])
    with_preds = [r for r in ok if not r["baseline"] and r.get("preds") and os.path.isdir(os.path.join(r["preds"], "sem"))][:3]
    if not with_preds:
        return ""
    ids = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(with_preds[0]["preds"], "sem")))[:n]
    cols = 2 + len(with_preds)
    out = [f'<h3>Sample test predictions <span class="muted">(blue = object pixels; IoU of object vs background)</span></h3>'
           f'<div class="gal">']
    for fid in ids:
        rgb = _find_rgb(test_dir, fid)
        gt_path = os.path.join(test_dir, "masks", fid + ".png")
        if not rgb or not os.path.exists(gt_path):
            continue
        gt = np.asarray(Image.open(gt_path))
        gt_fg = (gt > 0) & (gt != 255)
        gt_cls = [classes[v - 1] for v in np.unique(gt) if 0 < v <= len(classes)]
        cells = [f'<figure><img src="{_b64(Image.open(rgb))}" alt="rgb {fid}"><figcaption>{fid}</figcaption></figure>',
                 f'<figure><img src="{_b64(_overlay(rgb, gt_fg))}" alt="ground truth {fid}"><figcaption>truth: {e_(", ".join(gt_cls) or "none")}</figcaption></figure>']
        for r in with_preds:
            sem = np.asarray(Image.open(os.path.join(r["preds"], "sem", fid + ".png")))
            fg = sem > 0
            if fg.shape != gt_fg.shape:
                fg = np.asarray(Image.fromarray(fg.astype(np.uint8)).resize(gt_fg.shape[::-1], Image.NEAREST)) > 0
            inter, union = (fg & gt_fg).sum(), (fg | gt_fg).sum()
            vals, cnt = np.unique(sem[sem > 0], return_counts=True)  # classes by area, ignore <5% specks
            pc = [classes[v - 1] for v, c in sorted(zip(vals, cnt), key=lambda t: -t[1]) if v <= len(classes) and c >= 0.05 * cnt.sum()]
            cells.append(f'<figure><img src="{_b64(_overlay(rgb, fg))}" alt="{e_(r["backbone"])} prediction {fid}">'
                         f'<figcaption>{e_(r["backbone"])}: {e_(", ".join(pc) or "none")} · IoU {inter / union if union else 1:.2f}</figcaption></figure>')
        out.append(f'<div class="row" style="grid-template-columns:repeat({cols},minmax(0,1fr))">{"".join(cells)}</div>')
    out.append("</div>")
    return "".join(out)


def samples_cls(e, best, n=16):
    pcsv = os.path.join(best.get("preds") or "", "predictions.csv")
    if not os.path.exists(pcsv):
        return ""
    test_dir = os.path.join(e["data"]["root"], e["data"]["test"])
    truth = {}
    if os.path.exists(os.path.join(test_dir, "labels.csv")):
        with open(os.path.join(test_dir, "labels.csv")) as f:
            truth = {r["id"]: r["class"] for r in csv.DictReader(f)}
    with open(pcsv) as f:
        rows = list(csv.DictReader(f))[:n]
    cells = []
    for r in rows:
        path = r.get("path") or _find_rgb(test_dir, r["id"])
        if not path or not os.path.exists(path):
            continue
        t = truth.get(r["id"]) or (r["id"].split("/")[0] if "/" in r["id"] else None)
        okc = t == r["pred"]
        mark = f'<span style="color:var(--{"good" if okc else "critical"})" aria-hidden="true">{"✓" if okc else "✕"}</span>'
        cells.append(f'<figure><img src="{_b64(Image.open(path), 200)}" alt="{e_(r["id"])}"><figcaption>{mark} pred <b>{e_(r["pred"])}</b> '
                     f'({float(r["conf"]):.2f}) · true {e_(t or "?")}{"" if okc else " (wrong)"}</figcaption></figure>')
    return f'<h3>Sample test predictions · {e_(best["backbone"])}</h3><div class="cls-gal">{"".join(cells)}</div>' if cells else ""


def section_dataset(ds):
    if not ds:
        return ""
    h = [f'<section class="card"><h2>Dataset</h2><div class="meta"><span>root <code>{e_(ds.get("root", ""))}</code></span>'
         f'<span>{len(ds.get("classes", []))} classes</span></div>']
    for msg in ds.get("errors", []):
        h.append(f'<div class="err">✕ {e_(msg)}</div>')
    for msg in ds.get("warnings", []):
        h.append(f'<div class="warn">! {e_(msg)}</div>')
    sp = ds.get("splits", [])
    if sp:
        h.append('<div class="scroll"><table><thead><tr><th>Split</th><th class="num">Images</th><th>Layout</th><th>Sizes</th>'
                 '<th>Depth</th><th>Intrinsics</th><th class="num">Mean object area</th></tr></thead><tbody>')
        for s in sp:
            h.append(f'<tr><td>{e_(s["split"])}</td><td class="num">{s.get("n", 0)}</td><td>{e_(s.get("layout", "?"))}</td>'
                     f'<td>{e_(", ".join(s.get("image_sizes", {})))}</td><td>{"yes" if s.get("has_depth") else "no"}</td>'
                     f'<td>{"yes" if s.get("has_K") else "no"}</td>'
                     f'<td class="num">{fmt(s.get("seg_mean_fg_frac"), 3) if "seg_mean_fg_frac" in s else "–"}</td></tr>')
        h.append("</tbody></table></div>")
        classes = ds.get("classes", [])
        key = "cls_counts" if any("cls_counts" in s for s in sp) else "seg_frames_with_class"
        if any(key in s for s in sp) and len(classes) <= 60:
            h.append(f'<h3>Class balance ({"labels" if key == "cls_counts" else "frames containing class"})</h3>'
                     '<div class="scroll"><table><thead><tr><th>Class</th>' + "".join(f'<th class="num">{e_(s["split"])}</th>' for s in sp) + "</tr></thead><tbody>")
            for c in classes:
                h.append(f"<tr><td>{e_(c)}</td>" + "".join(f'<td class="num">{s.get(key, {}).get(c, 0)}</td>' for s in sp) + "</tr>")
            h.append("</tbody></table></div>")
    h.append("</section>")
    return "".join(h)


def page(title, body):
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{e_(title)}</title><style>{_css()}</style></head><body><main>{body}</main></body></html>")


def write_report(name):
    e, ed, results, ds, st = gather(name)
    classes = ds.get("classes", [])
    state = st.get("state", "running")
    started = time.strftime("%Y-%m-%d %H:%M", time.localtime(st["started"])) if st.get("started") else "?"
    dur = f'{st["duration_s"] / 60:.1f} min' if st.get("duration_s") else "in progress" if state == "running" else "–"
    body = [f'<header class="card"><div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap"><h1>{e_(name)}</h1>{status_badge(state)}</div>'
            f'<div class="meta"><span>started {started}</span><span>duration {dur}</span>'
            f'<span>tasks: {e_(", ".join(e["tasks"]))}</span><span>backbones: {e_(", ".join(e["backbones"]))}</span>'
            f'<span>test split: {e_(e["data"]["test"])}</span></div>'
            + (f'<p class="sec" style="margin:8px 0 0">{e_(e["description"])}</p>' if e.get("description") else "") + "</header>"]
    if st.get("error"):
        body.append(f'<div class="card"><div class="err">✕ {e_(st["error"])}</div></div>')
    failed = [r for r in results if r.get("state") == "failed"]
    if failed:
        body.append('<section class="card"><h2>Failures</h2>' + "".join(
            f'<div class="err">✕ <b>{e_(r["run"])}</b> failed during {e_(r.get("stage") or "?")}<pre>{e_(r.get("error") or "")}</pre></div>' for r in failed) + "</section>")

    tiles = []
    for task in e["tasks"]:
        ok = [r for r in results if r["task"] == task and r.get("eval")]
        if ok:
            b = max(ok, key=lambda r: r["eval"]["metrics"][PRIMARY[task]])
            tiles.append(f'<div class="card tile"><h2>{TASK_NAME[task]} · best test {PRIMARY[task]}</h2>'
                         f'<div class="v">{b["eval"]["metrics"][PRIMARY[task]]:.3f}</div>'
                         f'<div class="s">{e_(b["backbone"])} · {b["eval"]["benchmark"]["bs1"]["ms_per_batch"]:.1f} ms/img</div></div>')
    if tiles:
        body.append(f'<section class="tiles">{"".join(tiles)}</section>')
    for task in e["tasks"]:
        body.append(section_task(e, task, [r for r in results if r["task"] == task], classes))
    body.append(section_dataset(ds))
    body.append(f'<section class="card"><details><summary>Experiment file</summary><pre>{e_(yaml.safe_dump(e, sort_keys=False))}</pre></details>'
                f'<p class="muted" style="font-size:12px;margin:8px 0 0">Generated {time.strftime("%Y-%m-%d %H:%M:%S")} · '
                f'artifacts in <code>{e_(ed)}</code> · runs in <code>{e_(RUNS)}</code></p></section>')

    os.makedirs(REPORTS, exist_ok=True)
    path = os.path.join(REPORTS, f"{name}.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(page(f"{name} report", "".join(body)))
    write_index()
    return path


def write_index():
    rows = []
    for ed in sorted(glob.glob(os.path.join(ROOT, "experiments", "*")), key=os.path.getmtime, reverse=True):
        name = os.path.basename(ed)
        st = _json(os.path.join(ed, "status.json")) or {}
        res = _json(os.path.join(ed, "results.json")) or []
        best = []
        for r in res:
            ev = _json(os.path.join(ed, "eval", r["run"] + ".json"))
            if ev and r.get("task"):
                best.append((r["task"], ev["metrics"][PRIMARY[r["task"]]], r["backbone"]))
        summ = ", ".join(f"{t} {PRIMARY[t]} {v:.3f} ({bb})" for t in PRIMARY
                         for (tt, v, bb) in [max([b for b in best if b[0] == t], key=lambda b: b[1], default=(t, None, ""))] if v is not None)
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(st["started"])) if st.get("started") else ""
        rows.append(f'<tr><td><a href="{e_(name)}.html">{e_(name)}</a></td><td>{status_badge(st.get("state", "?"))}</td>'
                    f"<td>{when}</td><td>{e_(summ) or '–'}</td></tr>")
    body = ('<header class="card"><h1>DINO experiments</h1><p class="sec" style="margin:6px 0 0">Open a report from the JupyterLab '
            'file browser (<code>dino/reports/</code>); links work when this page is opened the same way.</p></header>'
            '<section class="card"><div class="scroll"><table><thead><tr><th>Experiment</th><th>State</th><th>Started</th><th>Best test results</th>'
            f'</tr></thead><tbody>{"".join(rows) or "<tr><td colspan=4 class=muted>no experiments yet</td></tr>"}</tbody></table></div></section>')
    os.makedirs(REPORTS, exist_ok=True)
    with open(os.path.join(REPORTS, "index.html"), "w", encoding="utf-8") as f:
        f.write(page("DINO experiments", body))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    a = ap.parse_args()
    print(write_report(a.name))


if __name__ == "__main__":
    main()
