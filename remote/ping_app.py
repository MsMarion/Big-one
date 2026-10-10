"""Minimal FastAPI app on the box: liveness + GPU check.

SSH port forwarding is disabled on this host, so this listens on 127.0.0.1 only and is reached
from the box itself (the local dashboard calls it via `ssh dino-box curl ...`).

    bash scripts/bg.sh ping uvicorn remote.ping_app:app --host 127.0.0.1 --port 8000

The GPU check runs in a short-lived subprocess: holding a HIP context in this long-lived server
leaves a runtime thread spinning at 100% CPU after the first call.
"""
import json
import os
import socket
import subprocess
import sys
import time

from fastapi import FastAPI

app = FastAPI(title="dino-box ping")
STARTED = time.time()

GPU_CHECK = """
import json, time, torch
t = time.perf_counter()
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
(x @ x).sum().item()
print(json.dumps({"device": torch.cuda.get_device_name(), "torch": torch.__version__, "hip": torch.version.hip,
                  "matmul_ms": round((time.perf_counter() - t) * 1e3, 2)}))
"""


@app.get("/ping")
def ping():
    return {"pong": True, "host": socket.gethostname(), "uptime_s": round(time.time() - STARTED, 1), "pid": os.getpid()}


@app.get("/gpu")
def gpu():
    t = time.perf_counter()
    r = subprocess.run([sys.executable, "-c", GPU_CHECK], capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        return {"error": r.stderr[-500:]}
    out = json.loads(r.stdout.strip().splitlines()[-1])
    out["total_ms_incl_startup"] = round((time.perf_counter() - t) * 1e3)
    return out
