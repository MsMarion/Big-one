"""Render template images of a catalog object from its MESH (for the dino_templates identity, tools/build_templates.py).
Runs in the FoundationPose env on the PC's GPU (nvdiffrast, the renderer FoundationPose itself uses):

    /opt/fp_venv/bin/python /repo/tools/render_templates.py lego_2x4 [lego_2x2 ...] --out data/templates [--views 60]

Views: the camera on a hemisphere around the object resting on the desk -- elevations 25-90 deg (a robot looks from
above), all around; distance so the object fills ~60 % of a 224 px crop; soft lighting with random strength;
background flat gray (what crop_for_embedding gives a real crop); each view re-tinted a random real-world color
(white, red, yellow, blue, green, black, gray) so the match is by shape. Mesh: the catalog entry's `mesh`, else the brick
meshes (models/foundationpose/meshes/<name>/mesh.obj, tools/make_brick_meshes.py).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PALETTE = [(235, 235, 230), (200, 30, 30), (240, 200, 20), (30, 80, 200), (30, 150, 60), (35, 35, 35), (150, 150, 150)]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "models" / "foundationpose" / "FoundationPose"))

import torch  # noqa: E402
import trimesh  # noqa: E402
import nvdiffrast.torch as dr  # noqa: E402
from PIL import Image  # noqa: E402

from pnp_core.catalog import load_yaml  # noqa: E402


def look_at(eye, target, up=(0, 0, 1)) -> np.ndarray:
    """T_obj_cam (OpenCV camera: z forward, y down) looking from eye at target."""
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:                      # straight down: any x
        x = np.array([1.0, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = np.column_stack([x, y, z]), eye
    return T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("objects", nargs="+")
    ap.add_argument("--out", default="data/templates")
    ap.add_argument("--views", type=int, default=60)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-colors", action="store_true", help="the mesh's own colors (default: random real colors)")
    a = ap.parse_args()
    from Utils import make_mesh_tensors, nvdiffrast_render          # NVlabs FoundationPose
    cat = load_yaml("config/objects.yaml")
    glctx = dr.RasterizeCudaContext()
    rng = np.random.default_rng(a.seed)
    S = a.size
    for name in a.objects:
        mp = cat.get(name, {}).get("mesh") or f"models/foundationpose/meshes/{name}/mesh.obj"
        mesh = trimesh.load(str(REPO / mp), force="mesh")
        mt = make_mesh_tensors(mesh)
        r = float(np.linalg.norm(mesh.extents)) / 2
        f = S * 0.6 / (2 * r) * 1.0
        out = REPO / a.out / name
        out.mkdir(parents=True, exist_ok=True)
        for k in range(a.views):
            el = np.radians(rng.uniform(25, 90))
            az = rng.uniform(0, 2 * np.pi)
            dist = 0.6 * f / f * 3.0 * r / np.tan(np.radians(20))          # ~20 deg half field of view
            K = np.array([[S / 2 / np.tan(np.radians(20)), 0, S / 2], [0, S / 2 / np.tan(np.radians(20)), S / 2],
                          [0, 0, 1.0]])
            eye = dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
            T_cam_obj = np.linalg.inv(look_at(eye, [0, 0, 0]))
            light = rng.uniform(0.6, 1.0)
            rgb, depth, _ = nvdiffrast_render(K=K, H=S, W=S, ob_in_cams=torch.tensor(T_cam_obj[None], dtype=torch.float,
                                                                                        device="cuda"),
                                              glctx=glctx, mesh_tensors=mt, use_light=True,
                                              light_dir=np.array([0, 0, 1.0]), w_ambient=0.5 * light,
                                              w_diffuse=0.6 * light)
            img = (rgb[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            fg = depth[0].cpu().numpy() > 0
            if not a.keep_colors:
                # re-tint: the shading (studs, edges) stays, the color is random -- identity by SHAPE, not by the
                # catalog's sim color (the real 2x4 is white, the sim one red); color is read separately
                lum = img.astype(np.float32).mean(axis=2, keepdims=True)
                lum = lum / max(1.0, float(np.percentile(lum[fg], 95)))
                col = np.array(PALETTE[rng.integers(len(PALETTE))], np.float32)
                img = np.where(fg[..., None], np.clip(lum * col, 0, 255), img).astype(np.uint8)
            img[~fg] = 124                                                   # the gray of crop_for_embedding
            Image.fromarray(img).save(out / f"render_{k:03d}.png")
        print(f"{name}: {a.views} renders -> {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
