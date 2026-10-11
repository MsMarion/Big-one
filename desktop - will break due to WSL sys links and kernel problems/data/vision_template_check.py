"""How well do DINO embeddings tell the catalog objects apart? Leave-one-out over a template library: each template
classified by its nearest OTHER template (and by the mean of each object's others). Also: the shape vs color split.
    /opt/fp_venv/bin/python /repo/data/vision_template_check.py models/vision/templates_bricks_v1.npz
"""
import sys
import numpy as np

lib = np.load(sys.argv[1], allow_pickle=True)
names, E = np.array(lib["names"]), lib["embeddings"].astype(np.float32)
S = E @ E.T
np.fill_diagonal(S, -2)
nn = names[S.argmax(axis=1)]
print(f"nearest-neighbour leave-one-out: {np.mean(nn == names) * 100:.1f}% of {len(names)}")
objs = sorted(set(names))
conf = {a: {b: 0 for b in objs} for a in objs}
for t, p in zip(names, nn):
    conf[t][p] += 1
print("confusion (rows = true):  " + "  ".join(f"{o[-3:]:>5s}" for o in objs))
for a in objs:
    print(f"  {a:10s}" + "".join(f"{conf[a][b]:7d}" for b in objs))

# kinds of objects: every lego_* is one kind "brick"
kind = np.array(["brick" if n.startswith("lego_") else n for n in names])
if len(set(kind)) > 1:
    nk = kind[S.argmax(axis=1)]
    print(f"KIND leave-one-out (brick vs other objects): {np.mean(nk == kind) * 100:.1f}%")
    for k in sorted(set(kind)):
        sel = kind == k
        print(f"  {k:16s} {np.mean(nk[sel] == k) * 100:5.1f}%  of {sel.sum()}")
