"""The pick & place node's state log, compact: one line per state change, with wall-clock gaps (where it hovered)."""
import re
import sys

n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
pat = re.compile(r"\[pnp_pick_place\]: (.*)$")
ts = re.compile(r"\[INFO\] \[(\d+\.\d+)\]")
rows = []
for line in open("/data/logs_bringup.log", errors="ignore"):
    if "pick_place_node" not in line:
        continue
    m, t = pat.search(line), ts.search(line)
    if m and t:
        rows.append((float(t.group(1)), m.group(1)))
rows = rows[-n:]
prev = rows[0][0] if rows else 0
for t, msg in rows:
    print(f"+{t - prev:5.1f}s  {msg[:200]}")
    prev = t
