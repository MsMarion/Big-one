"""FoundationPose with the NGC ONNX networks: NVlabs' pipeline, our two network adapters.

Runs in /opt/fp_venv (image pnp:pose: NumPy 2, nvdiffrast, PyTorch3D). docs/learning/10_6d_pose.md

    est = make_estimator("tomato_soup_can")                  # loads mesh + ONNX networks once
    pose = est.register(K, rgb, depth_m, mask, iteration=5)   # first pose: 252 hypotheses -> refine -> score
    pose = est.track_one(rgb, depth_m, K, iteration=2)        # later frames: refine from the last pose
    # pose = 4x4, object frame (the mesh's frame) in the camera optical frame

What NVlabs' code does (unchanged): hypotheses (rotations on an icosphere x in-plane, translation from mask +
depth), rendering the mesh at each with nvdiffrast, 160x160 crops of render and camera image, refine loops,
scoring. What this file replaces: NVlabs' PoseRefinePredictor/ScorePredictor load `weights/<run>/config.yml` +
`model_best.pth`; here the config comes from config/foundationpose.yaml and the networks from the ONNX files,
run with ONNX Runtime directly on the GPU tensors (no copies through the CPU).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(os.environ.get("PNP_REPO", "/repo"))
CFG = yaml.safe_load((REPO / "config/foundationpose.yaml").read_text())
sys.path.insert(0, str(REPO / CFG["code_dir"]))     # NVlabs code: Utils, estimater, learning.*

import torch  # noqa: E402  (first: its pip-installed CUDA 12 / cuDNN 9 libraries are what ONNX Runtime uses)
import onnxruntime as ort  # noqa: E402

if hasattr(ort, "preload_dlls"):
    ort.preload_dlls()          # load CUDA/cuDNN from the nvidia-* pip packages (no system CUDA runtime needed)


def _preload_tensorrt() -> bool:
    """ONNX Runtime's TensorRT engine dlopens libnvinfer.so.10; the pip package keeps it in site-packages/tensorrt_libs,
    which isn't on the library path. Load those libraries globally first, then ONNX Runtime finds them."""
    try:
        import ctypes
        import glob

        import tensorrt_libs
        d = os.path.dirname(tensorrt_libs.__file__)
        for lib in ("libnvinfer.so.10", "libnvinfer_plugin.so.10", "libnvonnxparser.so.10"):
            for path in glob.glob(f"{d}/{lib}"):
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        return True
    except Exception:  # noqa: BLE001
        return False


HAVE_TRT = CFG.get("backend", "tensorrt") == "tensorrt" and _preload_tensorrt()
import trimesh  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from estimater import FoundationPose  # noqa: E402  (NVlabs)
from learning.datasets.h5_dataset import PoseRefinePairH5Dataset, ScoreMultiPairH5Dataset  # noqa: E402
from learning.training.predict_pose_refine import PoseRefinePredictor  # noqa: E402
from learning.training.predict_score import ScorePredictor  # noqa: E402
import nvdiffrast.torch as dr  # noqa: E402


class _OnnxNet:
    """Looks like NVlabs' torch network to their predictors: model(A, B) with A, B (N, 6, 160, 160) CUDA tensors.
    The ONNX networks want channels-last (N, 160, 160, 6); inputs and outputs stay on the GPU (IO binding)."""

    def __init__(self, path: Path, outputs: list[str], max_batch: int = 252, opt_batch: int = 252):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # The GPU is shared with Isaac Sim's renderer (and whatever else runs on it). ONNX Runtime's defaults grow its
        # memory pool without limit and try every cuDNN algorithm with big scratch buffers ("EXHAUSTIVE"): together
        # that once filled the 32 GB and froze Isaac Sim. Cap the pool and pick algorithms by heuristic instead.
        cuda = {"device_id": 0, "arena_extend_strategy": "kSameAsRequested",
                "gpu_mem_limit": int(CFG.get("gpu_mem_limit_gb", 6) * (1 << 30)),
                "cudnn_conv_algo_search": "HEURISTIC"}
        providers = [("CUDAExecutionProvider", cuda), "CPUExecutionProvider"]
        if HAVE_TRT:
            # TensorRT compiles the network once into an engine tuned for this GPU (fused layers, best kernels) and
            # caches it on disk (first start: minutes; later: seconds). FP32 like Isaac ROS: its docs report precision
            # loss with FP16 on TensorRT 10.3+. Batch sizes must be declared up front (min/opt/max "profiles").
            shp = lambda b: f"input1:{b}x160x160x6,input2:{b}x160x160x6"  # noqa: E731
            cache = os.path.join(CFG.get("trt_cache_dir", "/ws/trt_cache"), path.stem)
            os.makedirs(cache, exist_ok=True)
            trt = {"device_id": 0, "trt_fp16_enable": False, "trt_max_workspace_size": 4 << 30,
                   "trt_engine_cache_enable": True, "trt_engine_cache_path": cache,
                   "trt_timing_cache_enable": True, "trt_timing_cache_path": cache,
                   "trt_profile_min_shapes": shp(1), "trt_profile_opt_shapes": shp(opt_batch),
                   "trt_profile_max_shapes": shp(max_batch)}
            providers.insert(0, ("TensorrtExecutionProvider", trt))
        self.sess = ort.InferenceSession(str(path), so, providers=providers)
        if not ({"TensorrtExecutionProvider", "CUDAExecutionProvider"} & set(self.sess.get_providers())):
            raise RuntimeError(f"ONNX Runtime has no GPU engine for {path.name}: {self.sess.get_providers()}")
        self.engine = self.sess.get_providers()[0]
        self.out_names = [o.name for o in self.sess.get_outputs()]
        self.keys = outputs

    def _run(self, A, B):
        A = A.permute(0, 2, 3, 1).contiguous().float()
        B = B.permute(0, 2, 3, 1).contiguous().float()
        io = self.sess.io_binding()
        for name, t in (("input1", A), ("input2", B)):
            io.bind_input(name, "cuda", 0, np.float32, tuple(t.shape), t.data_ptr())
        for name in self.out_names:
            io.bind_output(name, "cuda", 0)
        torch.cuda.synchronize()
        self.sess.run_with_iobinding(io)
        return [_ort_to_torch(o) for o in io.get_outputs()]

    # torch-module look-alikes used by NVlabs' code
    def eval(self):
        return self

    def to(self, *_):
        return self

    def cuda(self):
        return self


def _ort_to_torch(ov) -> torch.Tensor:
    """OrtValue on the GPU -> torch tensor (via DLPack when available, else through numpy)."""
    try:
        return torch.from_dlpack(ov._ortvalue.to_dlpack())
    except Exception:  # noqa: BLE001
        return torch.as_tensor(ov.numpy(), device="cuda")


class OnnxRefineNet(_OnnxNet):
    """Each hypothesis is refined independently, so it can run in chunks (Isaac ROS uses 42): far less GPU memory
    than all 252 at once. (The score network can't: it compares all hypotheses with each other.)"""
    chunk = int(CFG.get("refine_chunk", 42))

    def __call__(self, A, B):
        parts = [self._run(A[i:i + self.chunk], B[i:i + self.chunk]) for i in range(0, len(A), self.chunk)]
        return {k: torch.cat([p[j] for p in parts]).float() for j, k in enumerate(self.keys)}  # trans/rot (N,3)


class OnnxScoreNet(_OnnxNet):
    def __call__(self, A, B, L=None):
        return {"score_logit": self._run(A, B)[0].reshape(-1).float()}  # one logit per hypothesis


def _refiner() -> PoseRefinePredictor:
    r = PoseRefinePredictor.__new__(PoseRefinePredictor)              # skip NVlabs __init__ (needs .pth + yml)
    r.amp, r.run_name = False, "ngc-onnx-1.0.1"
    r.cfg = OmegaConf.create(CFG["refine"])
    r.dataset = PoseRefinePairH5Dataset(cfg=r.cfg, h5_file="", mode="test")
    chunk = int(CFG.get("refine_chunk", 42))
    r.model = OnnxRefineNet(REPO / CFG["onnx"]["refine"], CFG["onnx"]["refine_outputs"], max_batch=chunk, opt_batch=chunk)
    r.last_trans_update = r.last_rot_update = None
    return r


def _scorer() -> ScorePredictor:
    s = ScorePredictor.__new__(ScorePredictor)
    s.amp, s.run_name = False, "ngc-onnx-1.0.1"
    s.cfg = OmegaConf.create(CFG["score"])
    s.dataset = ScoreMultiPairH5Dataset(cfg=s.cfg, mode="test", h5_file=None, max_num_key=1)
    s.model = OnnxScoreNet(REPO / CFG["onnx"]["score"], ["score_logit"], max_batch=252, opt_batch=252)
    return s


def make_estimator(object_name: str, debug_dir: str = "/tmp/foundationpose_debug") -> FoundationPose:
    mesh = trimesh.load(str(REPO / CFG["meshes"][object_name]), force="mesh")
    return FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
                          scorer=_scorer(), refiner=_refiner(), debug=0, debug_dir=debug_dir,
                          glctx=dr.RasterizeCudaContext())
