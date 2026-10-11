"""Export an object's USD asset as a textured OBJ (for FoundationPose), in the frame the simulation moves it in.

    .\\run_isaac.ps1 sim\\export_mesh.py tomato_soup_can
    -> models/foundationpose/meshes/tomato_soup_can/{mesh.obj, mesh.mtl, texture.png}

What this teaches: a USD asset is a tree; the visible geometry is one or more Mesh prims somewhere below the
prim you place (here /World/objects/target, whose pose Isaac publishes as the TF frame `target`). Each Mesh has
its own local transform. Exporting every Mesh's points through its transform RELATIVE TO the placed prim gives a
mesh whose frame is exactly that prim's frame, so a pose estimated against this mesh can be compared 1:1 with the
simulator's ground truth. Colors come from the material's texture (UsdUVTexture), positions-on-texture from the
mesh's "st" primvar (UV coordinates).
"""

import argparse

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import omni.client  # noqa: E402
from common import load_yaml, log, shutdown  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from pxr import Usd, UsdGeom, UsdShade  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("object", help="config/objects.yaml key, e.g. tomato_soup_can")
parser.add_argument("--out", default="models/foundationpose/meshes")
args = parser.parse_args()


def find_texture(stage, mesh_prim):
    """Path of the diffuse texture bound to this mesh (UsdPreviewSurface -> UsdUVTexture.file), or None."""
    mat, _ = UsdShade.MaterialBindingAPI(mesh_prim).ComputeBoundMaterial()
    if not mat:
        return None
    for prim in Usd.PrimRange(mat.GetPrim()):
        sh = UsdShade.Shader(prim)
        if not sh:
            continue
        for name in ("file", "diffuse_texture"):                 # UsdUVTexture / MDL OmniPBR
            inp = sh.GetInput(name)
            if inp and inp.Get():
                ap = inp.Get()
                return ap.resolvedPath or ap.path
    return None


def main() -> int:
    obj = load_yaml("config/objects.yaml")[args.object]
    stage = get_current_stage()
    root_path = "/World/export_root"
    add_reference_to_stage(usd_path=f"{get_assets_root_path()}/{obj['usd']}", prim_path=root_path)
    root = stage.GetPrimAtPath(root_path)
    cache = UsdGeom.XformCache()

    verts, uvs, faces, texture = [], [], [], None
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = np.asarray(mesh.GetPointsAttr().Get(), np.float64)
        M, _ = cache.ComputeRelativeTransform(prim, root)          # mesh -> placed prim (Gf: row vectors)
        M = np.asarray(M, np.float64)
        pts = np.c_[pts, np.ones(len(pts))] @ M
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        st = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
        st_vals = np.asarray(st.Get(), np.float64) if st and st.Get() is not None else None
        st_interp = st.GetInterpolation() if st else None
        st_idx = np.asarray(st.GetIndices()) if st and st.IsIndexed() else None
        v0, t0 = len(verts), len(uvs)
        verts += pts[:, :3].tolist()
        if st_vals is not None:
            uvs += st_vals.tolist()
        k = 0
        for c in counts:                                           # triangulate polygons as fans
            corner = list(range(k, k + c))
            for j in range(1, c - 1):
                tri = [corner[0], corner[j], corner[j + 1]]
                vi = [v0 + int(idx[t]) for t in tri]
                if st_vals is None:
                    ti = [None] * 3
                elif st_interp == "faceVarying":
                    ti = [t0 + int(st_idx[t] if st_idx is not None else t) for t in tri]
                else:                                              # vertex-interpolated UVs
                    ti = [t0 + int(st_idx[idx[t]] if st_idx is not None else idx[t]) for t in tri]
                faces.append((vi, ti))
            k += c
        texture = texture or find_texture(stage, prim)
        log(f"[export] {prim.GetPath()}: {len(pts)} points, {len(counts)} faces, uv={st_interp}")

    v = np.asarray(verts)
    log(f"[export] bounds min {v.min(0).round(4)} max {v.max(0).round(4)} size {(v.max(0) - v.min(0)).round(4)} m "
        f"(config dims_m {obj['dims_m']})")
    out = Path(args.out) / args.object
    out.mkdir(parents=True, exist_ok=True)
    if texture:
        res, _, content = omni.client.read_file(texture)           # works for local files and remote asset URLs
        if res == omni.client.Result.OK:
            (out / "texture.png").write_bytes(memoryview(content).tobytes())
            log(f"[export] texture {texture}")
        else:
            log(f"[export] WARNING could not read texture {texture}: {res}")
            texture = None
    with open(out / "mesh.mtl", "w") as f:
        f.write("newmtl material0\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\n" + ("map_Kd texture.png\n" if texture else ""))
    with open(out / "mesh.obj", "w") as f:
        f.write(f"# {args.object} from {obj['usd']} (frame = the placed prim = TF 'target' in the sim)\n")
        f.write("mtllib mesh.mtl\nusemtl material0\n")
        f.writelines(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in v)
        f.writelines(f"vt {a:.6f} {b:.6f}\n" for a, b in uvs)
        for vi, ti in faces:
            if ti[0] is None:
                f.write("f " + " ".join(str(a + 1) for a in vi) + "\n")
            else:
                f.write("f " + " ".join(f"{a + 1}/{b + 1}" for a, b in zip(vi, ti)) + "\n")
    log(f"[export] wrote {out / 'mesh.obj'}: {len(v)} vertices, {len(faces)} triangles, {len(uvs)} uvs")
    return 0


if __name__ == "__main__":
    shutdown(app, main())
