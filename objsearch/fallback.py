"""Last-resort models that always build: a box at the object's size with its photo on top.

Written straight to GLB with trimesh (no Blender, no network). glTF is Y-up, so the box's height runs along Y;
Blender's importer turns that into Z-up, and normalize then scales it against dims_mm like any other mesh.
"""
import os

import numpy as np
from PIL import Image, ImageOps

SWATCH = 32  # px strip at the right of the texture atlas holding the side color


def _photo_and_color(photo, aspect, size=512):
    """Center-crop the photo to the box's L:W aspect -> (top image, side RGB). The side color is the median of
    the photo's middle, where the object usually is."""
    im = ImageOps.exif_transpose(Image.open(photo)).convert("RGB")
    w, h = im.size
    if w / h > aspect:
        cw, ch = int(h * aspect), h
    else:
        cw, ch = w, int(w / aspect)
    im = im.crop(((w - cw) // 2, (h - ch) // 2, (w - cw) // 2 + cw, (h - ch) // 2 + ch))
    top = im.resize((size, max(8, int(size / aspect))) if aspect >= 1 else (max(8, int(size * aspect)), size))
    a = np.asarray(top)
    mid = a[a.shape[0] // 3: 2 * a.shape[0] // 3, a.shape[1] // 3: 2 * a.shape[1] // 3].reshape(-1, 3)
    side = tuple(int(x * 0.85) for x in np.median(mid, axis=0))
    return top, side


def photo_box(dims_mm, out_path, photo=None, color=(150, 150, 155)):
    """Box of dims_mm [L, W, H] (written in mm) -> GLB at out_path. With a photo, it is the top face's texture."""
    import trimesh
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    L, W, H = (float(x) for x in dims_mm)
    x0, x1, z0, z1 = -L / 2, L / 2, -W / 2, W / 2
    top_img, side = (None, tuple(color))
    if photo:
        top_img, side = _photo_and_color(photo, L / W if W else 1.0)
    # faces as quads (4 corners each), top first; each gets its own vertices so UVs can differ per face
    quads = [
        [(x0, H, z0), (x1, H, z0), (x1, H, z1), (x0, H, z1)],   # top (+Y)
        [(x0, 0, z0), (x0, 0, z1), (x1, 0, z1), (x1, 0, z0)],   # bottom
        [(x0, 0, z1), (x0, H, z1), (x1, H, z1), (x1, 0, z1)],   # +Z
        [(x0, 0, z0), (x1, 0, z0), (x1, H, z0), (x0, H, z0)],   # -Z
        [(x1, 0, z0), (x1, 0, z1), (x1, H, z1), (x1, H, z0)],   # +X
        [(x0, 0, z0), (x0, H, z0), (x0, H, z1), (x0, 0, z1)],   # -X
    ]
    verts = np.array([v for q in quads for v in q], dtype=np.float64)
    faces = []
    for i in range(6):
        a, b, c, d = 4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3
        for tri in ((a, b, c), (a, c, d)):
            p = verts[list(tri)]
            n = np.cross(p[1] - p[0], p[2] - p[0])
            outward = p.mean(axis=0) - np.array([0, H / 2, 0])
            faces.append(tri if np.dot(n, outward) > 0 else (tri[0], tri[2], tri[1]))
    faces = np.array(faces)
    if top_img is None:  # plain colored box (material color; glTF factors are linear)
        lin = [((c / 255) / 12.92 if c / 255 <= 0.04045 else ((c / 255 + 0.055) / 1.055) ** 2.4) for c in side]
        mat = trimesh.visual.material.PBRMaterial(baseColorFactor=lin + [1.0], metallicFactor=0.0,
                                                  roughnessFactor=0.7)
        trimesh.Trimesh(verts, faces, process=False, visual=trimesh.visual.TextureVisuals(material=mat)).export(out_path)
        return out_path
    tw, th = top_img.size
    atlas = Image.new("RGB", (tw + SWATCH, th), side)
    atlas.paste(top_img, (0, 0))
    u1 = tw / (tw + SWATCH)
    su = (tw + SWATCH / 2) / (tw + SWATCH)
    # OpenGL-style UVs (v up): the top face's (x0, z0) corner shows the photo's top-left
    uv = [[0, 1], [u1, 1], [u1, 0], [0, 0]] + [[su, 0.5]] * 20
    mat = trimesh.visual.material.PBRMaterial(baseColorTexture=atlas, metallicFactor=0.0, roughnessFactor=0.7)
    mesh = trimesh.Trimesh(verts, faces, process=False,
                           visual=trimesh.visual.TextureVisuals(uv=np.array(uv, dtype=np.float64), material=mat))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    mesh.export(out_path)
    return out_path
