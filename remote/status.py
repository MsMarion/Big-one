"""One-shot JSON snapshot of the box for the local dashboard (stdlib only).

    python remote/status.py
"""
import glob
import json
import os
import shutil
import subprocess
import time

ROOT = os.environ.get("DINO_ROOT", "/workspace/dino")
RUNS = os.path.join(ROOT, "runs")
AMD_SMI = "/opt/venv/bin/amd-smi"


def _val(x):
    return x.get("value") if isinstance(x, dict) else None


def gpu():
    try:
        out = subprocess.run([AMD_SMI, "metric", "--usage", "--power", "--temperature", "--mem-usage", "--json"],
                             capture_output=True, text=True, timeout=10).stdout
        g = json.loads(out)["gpu_data"][0]
        mem = g.get("mem_usage", {})
        temp = g.get("temperature", {})
        return {
            "gfx_util": _val(g["usage"]["gfx_activity"]), "mem_util": _val(g["usage"]["umc_activity"]),
            "power_w": _val(g["power"]["socket_power"]), "throttle": g["power"].get("throttle_status"),
            "temp_edge": _val(temp.get("edge")), "temp_hotspot": _val(temp.get("hotspot")), "temp_mem": _val(temp.get("mem")),
            "vram_used_mb": _val(mem.get("used_vram")), "vram_total_mb": _val(mem.get("total_vram")),
        }
    except Exception as e:  # dashboard should still render without GPU stats
        return {"error": str(e)}


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def container():
    cpu_max = (_read("/sys/fs/cgroup/cpu.max") or "max 100000").split()
    stat = dict(line.split() for line in (_read("/sys/fs/cgroup/cpu.stat") or "").splitlines() if line)
    mem_max = _read("/sys/fs/cgroup/memory.max")
    du = shutil.disk_usage("/workspace")
    return {
        "cpus": None if cpu_max[0] == "max" else int(cpu_max[0]) / int(cpu_max[1]),
        "cpu_usage_usec": int(stat.get("usage_usec", 0)),
        "mem_used_gb": int(_read("/sys/fs/cgroup/memory.current") or 0) / 1e9,
        "mem_max_gb": None if mem_max in (None, "max") else int(mem_max) / 1e9,
        "workspace_used_gb": du.used / 1e9, "workspace_total_gb": du.total / 1e9,
        "loadavg_host": os.getloadavg(),
    }


def processes():
    """Python processes doing real work (training, inference, servers)."""
    out = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        cmd = (_read(f"/proc/{pid}/cmdline") or "").replace("\0", " ").strip()
        if not cmd or "python" not in cmd or "status.py" in cmd or "jupyterlab" in cmd:
            continue
        if any(k in cmd for k in ("dino.", "uvicorn", "ping_app", "tests.smoke")):
            out.append({"pid": int(pid), "cmd": cmd[:300]})
    return out


def jobs():
    out = []
    for log in sorted(glob.glob(os.path.join(RUNS, "_jobs", "*.log")), key=os.path.getmtime, reverse=True)[:10]:
        out.append({"name": os.path.basename(log)[:-4], "mtime": os.path.getmtime(log), "size": os.path.getsize(log)})
    return out


def runs():
    out = []
    for m in glob.glob(os.path.join(RUNS, "*", "metrics.json")):
        try:
            with open(m) as f:
                d = json.load(f)
        except (OSError, ValueError):
            continue
        b = d.get("eval", {}).get("benchmark", {})
        out.append({
            "name": d["name"], "task": d["task"], "backbone": d["backbone"], "metric": d["primary_metric"],
            "best": d["best"], "epochs": d["epochs"], "train_s": d["train_time_s"], "peak_gb": d["peak_vram_gb"],
            "ms_bs1": b.get("bs1", {}).get("ms_per_batch"), "mtime": os.path.getmtime(m),
            "history": [h.get(d["primary_metric"]) for h in d.get("history", [])],
        })
    # in-progress runs: log exists, no metrics yet
    for log in glob.glob(os.path.join(RUNS, "*", "log.txt")):
        d = os.path.dirname(log)
        if not os.path.exists(os.path.join(d, "metrics.json")):
            out.append({"name": os.path.basename(d), "in_progress": True, "mtime": os.path.getmtime(log)})
    return sorted(out, key=lambda r: r["mtime"], reverse=True)


if __name__ == "__main__":
    print(json.dumps({"time": time.time(), "host": os.uname().nodename, "gpu": gpu(), "container": container(),
                      "processes": processes(), "jobs": jobs(), "runs": runs()}))
