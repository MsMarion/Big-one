#!/usr/bin/env bash
# One-time (per checkout) setup of FoundationPose inside the pnp:pose container, then a self-check.
#   bash /repo/setup/fp_setup.sh
# 1. builds NVlabs' small C++ helper (mycpp: pose clustering) into models/foundationpose/FoundationPose/mycpp/build
# 2. checks the environment: imports, GPU, ONNX Runtime on CUDA with both networks, nvdiffrast rendering
set -eo pipefail
FP=/repo/models/foundationpose/FoundationPose
PY=/opt/fp_venv/bin/python
[ -d "$FP" ] || { echo "missing $FP (download the NVlabs code there, docs/learning/10_6d_pose.md)"; exit 1; }
[ -x "$PY" ] || { echo "missing $PY: this container isn't from pnp:pose (PNP_IMAGE=pnp:pose run_dev.sh)"; exit 1; }

echo "== building mycpp"
mkdir -p "$FP/mycpp/build"
cd "$FP/mycpp/build"
cmake .. -DPYTHON_EXECUTABLE="$PY" -Dpybind11_DIR="$($PY -m pybind11 --cmakedir)" -DCMAKE_BUILD_TYPE=Release > /tmp/mycpp_cmake.log
make -j"$(nproc)" 2>&1 | tail -2

echo "== self-check"
cd /repo
PYTHONPATH=/repo:/ws/install/pnp_perception/lib/python3.12/site-packages:$PYTHONPATH "$PY" -I - <<'EOF' 2>&1 | grep -v Warning
import sys, time
sys.path[:0] = ["/repo", "/repo/ros2_ws/src/pnp_perception"]
import numpy as np, torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
from pnp_perception import foundationpose_onnx as F
r, s = F._refiner(), F._scorer()
A = torch.rand(252, 6, 160, 160, device="cuda"); B = torch.rand(252, 6, 160, 160, device="cuda")
for name, net in (("refine", r.model), ("score", s.model)):
    out = net(A, B); torch.cuda.synchronize(); t = time.time()
    for _ in range(3): out = net(A, B)
    torch.cuda.synchronize()
    print(f"{name:6s} onnx on {net.sess.get_providers()[0]}: {[(k, tuple(v.shape)) for k, v in out.items()]}  {(time.time()-t)/3*1000:.0f} ms / 252 crops")
import nvdiffrast.torch as dr
ctx = dr.RasterizeCudaContext()
pos = torch.tensor([[[-0.5,-0.5,0,1],[0.5,-0.5,0,1],[0,0.5,0,1]]], device="cuda"); tri = torch.tensor([[0,1,2]], dtype=torch.int32, device="cuda")
rast, _ = dr.rasterize(ctx, pos, tri, resolution=[64, 64])
print("nvdiffrast ok, triangle covers", int((rast[..., 3] > 0).sum()), "of 4096 pixels")
import mycpp; print("mycpp ok")
EOF
