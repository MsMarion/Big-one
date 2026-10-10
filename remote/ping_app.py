"""Minimal FastAPI app on the box: liveness + GPU check.

SSH port forwarding is disabled on this host, so this listens on 127.0.0.1 only and is reached
from the box itself (the local dashboard calls it via `ssh dino-box curl ...`).

    bash scripts/bg.sh ping uvicorn remote.ping_app:app --host 127.0.0.1 --port 8000
"""
import os
import socket
import time

import torch
from fastapi import FastAPI

app = FastAPI(title="dino-box ping")
STARTED = time.time()


@app.get("/ping")
def ping():
    return {"pong": True, "host": socket.gethostname(), "uptime_s": round(time.time() - STARTED, 1), "pid": os.getpid()}


@app.get("/gpu")
def gpu():
    t = time.perf_counter()
    x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    (x @ x).sum().item()
    return {"device": torch.cuda.get_device_name(), "torch": torch.__version__, "hip": torch.version.hip,
            "matmul_ms": round((time.perf_counter() - t) * 1e3, 2),
            "mem_allocated_mb": round(torch.cuda.memory_allocated() / 1e6, 1)}
