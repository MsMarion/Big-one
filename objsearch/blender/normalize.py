"""Runs INSIDE .venv-bpy (pip bpy):  python -m objsearch.blender.normalize IN OUT_DIR [--dims L W H] [--name N]

Import a mesh (glb/gltf/obj/stl/ply/fbx), bake all transforms into the mesh data, fix the unit scale against
the catalog dims (tries x1, x0.001, x0.0254, x0.01, x1000), lay it so the catalog height is vertical, put
its base on z=0 centred on the origin (metres, Z-up), then write:
    asset.usd  model.glb  mesh/mesh.obj (+.mtl)  thumb.png (Cycles CPU render)
and print one line:  OBJSEARCH_RESULT {...json...}
"""
import argparse
import json
import math
import os
import sys

import bpy
from mathutils import Matrix, Vector

UNIT_FACTORS = (1.0, 0.001, 0.0254, 0.01, 1000.0)


def log(*a):
    print("[normalize]", *a, flush=True)


def import_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    elif ext == ".stl":
        bpy.ops.wm.stl_import(filepath=path)
    elif ext == ".ply":
        bpy.ops.wm.ply_import(filepath=path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    else:
        raise ValueError(f"unsupported mesh type {ext}")


def bake_transforms():
    """World transforms -> mesh data; drop parents/empties so every mesh sits at identity."""
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    for o in meshes:
        mw = o.matrix_world.copy()
        if o.data.users > 1:
            o.data = o.data.copy()
        o.data.transform(mw)
        o.parent = None
        o.matrix_world = Matrix.Identity(4)
    for o in [o for o in bpy.context.scene.objects if o.type != "MESH"]:
        bpy.data.objects.remove(o, do_unlink=True)
    return meshes


def bbox(meshes):
    lo, hi = Vector((math.inf,) * 3), Vector((-math.inf,) * 3)
    for o in meshes:
        for v in o.data.vertices:
            for i in range(3):
                lo[i] = min(lo[i], v.co[i])
                hi[i] = max(hi[i], v.co[i])
    return lo, hi


def apply(meshes, m):
    for o in meshes:
        o.data.transform(m)
        o.data.update()


def fit_units(ext_sorted, dims_sorted_m):
    """Unit factor whose footprint (two largest sides) best matches the catalog; height is checked separately
    because pin headers and connectors vary between makers (same rule as acquire.size_fit)."""
    best = (1.0, math.inf)
    for f in UNIT_FACTORS:
        err = max(abs(a * f - b) / b for a, b in zip(ext_sorted[1:], dims_sorted_m[1:]))
        if err < best[1]:
            best = (f, err)
    return best


def save_textures(tex_dir):
    """Images packed in the source (glTF embeds them) -> PNG files next to the OBJ, so mesh.mtl can point at them."""
    for img in bpy.data.images:
        if img.source != "FILE" and not img.packed_file:
            continue
        if not img.packed_file and os.path.exists(bpy.path.abspath(img.filepath)):
            continue
        name = "".join(c if c.isalnum() else "_" for c in os.path.splitext(img.name)[0])[:40] or "texture"
        path = os.path.join(tex_dir, name + ".png")
        img.save(filepath=path)
        img.filepath = path
        if img.packed_file:
            img.unpack(method="REMOVE")


def default_material():
    m = bpy.data.materials.new("objsearch_default")
    m.diffuse_color = (0.6, 0.6, 0.62, 1)
    if hasattr(m, "use_nodes"):
        m.use_nodes = True
        bsdf = m.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (0.6, 0.6, 0.62, 1)
    return m


def use_gpu():
    """Enable a Cycles GPU backend if this machine/driver supports one (AMD HIP needs a recent Adrenalin driver)."""
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
    except KeyError:
        return False
    for backend in ("OPTIX", "CUDA", "HIP", "ONEAPI"):
        try:
            prefs.compute_device_type = backend
            prefs.refresh_devices()
        except TypeError:
            continue
        gpus = [d for d in prefs.devices if d.type == backend]
        if gpus:
            for d in prefs.devices:
                d.use = d.type == backend
            log("Cycles GPU:", backend, [d.name for d in gpus])
            return True
    log("Cycles: no usable GPU backend, rendering on CPU")
    return False


def srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def linearize_colors(meshes):
    """STEP/CAD colors are display (sRGB) values but arrive in glTF as if linear -> washed out. Fix base colors."""
    seen = set()
    for o in meshes:
        for m in o.data.materials:
            if not m or m.name in seen or not m.node_tree:
                continue
            seen.add(m.name)
            bsdf = next((n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
            if bsdf and not bsdf.inputs["Base Color"].is_linked:
                r, g, b, a = bsdf.inputs["Base Color"].default_value
                bsdf.inputs["Base Color"].default_value = (srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b), a)
    return len(seen)


# PCB copper finishes as metals (linear base colour, roughness); what KiCad's 3D viewer shows for each finish
FINISH_METAL = {"enig": ((0.66, 0.43, 0.10), 0.30), "enepig": ((0.66, 0.43, 0.10), 0.30),
                "hard gold": ((0.66, 0.43, 0.10), 0.25), "hal": ((0.62, 0.62, 0.62), 0.35),
                "hasl": ((0.62, 0.62, 0.62), 0.35), "immersion silver": ((0.80, 0.80, 0.80), 0.25),
                "immersion tin": ((0.62, 0.62, 0.62), 0.35), "osp": ((0.55, 0.22, 0.08), 0.35)}


def finish_pads(meshes, finish):
    """KiCad's GLB export paints pads flat grey whatever the stackup says; give 'board_pad' meshes the metal of
    the board's copper finish (ENIG -> gold, HASL -> silver, OSP/bare -> copper)."""
    key = next((k for k in FINISH_METAL if (finish or "").lower().startswith(k)), None)
    if key is None:
        return 0
    rgb, rough = FINISH_METAL[key]
    mat = bpy.data.materials.new(f"copper_{key.replace(' ', '_')}")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Metallic"].default_value = 1.0
    bsdf.inputs["Roughness"].default_value = rough
    n = 0
    for o in meshes:
        if o.name.startswith("board_pad") or o.data.name.startswith("board_pad"):  # glTF mesh name
            o.data.materials.clear()
            o.data.materials.append(mat)
            n += 1
    return n


def recolor_parts(meshes, spec):
    """spec {key: {"hex": "#RRGGBB" (sRGB), "roughness", "metallic", "size": [a, b, c] mm (optional)}}.
    A part is matched by its measured size (sorted extents, mm, within max(0.3 mm, 2 %)) when "size" is given -
    CAD exports often give every part the same name - otherwise by an exact, unique mesh/object name.
    Call after unit scaling. Returns {key: number of meshes recoloured}."""
    hits = {}
    for key, c in spec.items():
        h = c["hex"].lstrip("#")
        rgb = tuple(srgb_to_linear(int(h[i:i + 2], 16) / 255) for i in (0, 2, 4))
        # targets: (object, material slot index or None = whole object)
        if c.get("size"):
            want = sorted(float(x) for x in c["size"])
            fits = lambda got: all(abs(g - w) <= max(0.3, 0.02 * w) for g, w in zip(sorted(got), want))  # noqa: E731
            targets = []
            for o in meshes:
                # a CAD part may be its own object, or one material slot (glTF primitive) of a merged mesh
                slots = {}
                for poly in o.data.polygons:
                    slots.setdefault(poly.material_index, set()).update(poly.vertices)
                for idx, verts in slots.items():
                    co = [o.data.vertices[v].co for v in verts]
                    got = [(max(p[i] for p in co) - min(p[i] for p in co)) * 1000 for i in range(3)]
                    if fits(got):
                        targets.append((o, idx if len(slots) > 1 else None))
        else:
            targets = [(o, None) for o in meshes if o.data.name == key or o.name == key]
            if len(targets) > 1:  # ambiguous name: refuse rather than recolour the wrong parts
                targets = []
        if not targets:
            hits[key] = 0
            continue
        mat = bpy.data.materials.new(f"recolor_{key}")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
        bsdf.inputs["Roughness"].default_value = float(c.get("roughness", 0.5))
        bsdf.inputs["Metallic"].default_value = float(c.get("metallic", 0.0))
        for o, idx in targets:
            if idx is None:
                o.data.materials.clear()
                o.data.materials.append(mat)
            else:
                o.data.materials[idx] = mat
        hits[key] = len(targets)
    return hits


def render_ortho(meshes, path, view, size=(1280, 800)):
    """Orthographic top or bottom view (for side-by-side checks against straight-on photos). Top: looking down -Z
    with +X right / +Y up in the image; bottom: looking up +Z, mirrored like turning the board over left-right."""
    scene = bpy.context.scene
    lo, hi = bbox(meshes)
    center = (lo + hi) / 2
    cam_data = bpy.data.cameras.new("ortho")
    cam_data.type = "ORTHO"
    w, h = hi.x - lo.x, hi.y - lo.y
    cam_data.ortho_scale = max(w, h * size[0] / size[1]) * 1.06
    cam = bpy.data.objects.new("ortho", cam_data)
    scene.collection.objects.link(cam)
    dist = (hi - lo).length * 2
    if view == "top":
        cam.location = (center.x, center.y, hi.z + dist)
        cam.rotation_euler = (0, 0, 0)
    else:
        cam.location = (center.x, center.y, lo.z - dist)
        cam.rotation_euler = (math.radians(180), 0, 0)
    cam_data.clip_start, cam_data.clip_end = dist / 100, dist * 4
    scene.camera = cam
    sun = bpy.data.objects.new("sun_o", bpy.data.lights.new("sun_o", "SUN"))
    sun.data.energy = 2.5
    # light from 35/25 degrees off the view axis (like a product photo): no mirror glare on glossy screens
    rx, ry, rz = cam.rotation_euler
    sun.rotation_euler = (rx + math.radians(35), ry + math.radians(25), rz)
    scene.collection.objects.link(sun)
    scene.render.resolution_x, scene.render.resolution_y = size
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    for o in (cam, sun):
        bpy.data.objects.remove(o, do_unlink=True)


def render_thumb(meshes, path, size=(960, 720)):
    scene = bpy.context.scene
    lo, hi = bbox(meshes)
    center, radius = (lo + hi) / 2, max((hi - lo).length / 2, 1e-4)
    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = 50
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    direction = Vector((1.0, -1.25, 0.95)).normalized()
    fov = 2 * math.atan(cam_data.sensor_width / (2 * cam_data.lens))
    cam.location = center + direction * (radius / math.sin(fov / 2)) * 1.05
    cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
    cam_data.clip_start, cam_data.clip_end = radius / 100, radius * 100
    scene.camera = cam
    sun_data = bpy.data.lights.new("sun", "SUN")
    sun_data.energy = 3.0
    sun = bpy.data.objects.new("sun", sun_data)
    sun.rotation_euler = (math.radians(45), math.radians(10), math.radians(30))
    scene.collection.objects.link(sun)
    world = scene.world or bpy.data.worlds.new("world")
    scene.world = world
    if hasattr(world, "use_nodes"):
        world.use_nodes = True
        bg = world.node_tree.nodes.get("Background")
        if bg:
            bg.inputs["Color"].default_value = (0.92, 0.92, 0.94, 1)
            bg.inputs["Strength"].default_value = 0.9
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU" if use_gpu() else "CPU"
    scene.cycles.samples = 16          # + denoiser: clean previews in a fraction of the time
    scene.cycles.use_denoising = True
    scene.view_settings.view_transform = "Standard"  # AgX (default) desaturates; keep catalog colors faithful
    scene.render.resolution_x, scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    for o in (cam, sun):
        bpy.data.objects.remove(o, do_unlink=True)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out_dir")
    ap.add_argument("--dims", nargs=3, type=float)
    ap.add_argument("--name", default="asset")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--srgb-colors", action="store_true", help="material colors are display (sRGB) values: linearize")
    ap.add_argument("--pad-finish", help="PCB copper finish (from the board stackup) applied to KiCad's board_pad meshes")
    ap.add_argument("--recolor", help='JSON {part name: {"hex": "#RRGGBB", "roughness": r, "metallic": m}}')
    ap.add_argument("--views", default="", help="extra orthographic renders, e.g. top,bottom (view_<name>.png)")
    ap.add_argument("--source-zup", action="store_true",
                    help="GLB was converted from STEP with raw Z-up coordinates (cascadio): undo the glTF Y-up import")
    a = ap.parse_args(argv)
    os.makedirs(os.path.join(a.out_dir, "mesh"), exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    import_any(a.src)
    meshes = bake_transforms()
    if not meshes:
        raise SystemExit("no mesh geometry in the file")
    flags = []
    if a.source_zup:
        # Blender's glTF importer maps (x, y, z)_gltf -> (x, -z, y); the file was really Z-up already, so rotate
        # -90 deg about X to recover the CAD orientation exactly (pins down for a board on headers, etc.)
        apply(meshes, Matrix.Rotation(math.radians(-90), 4, "X"))
    if a.srgb_colors or a.source_zup:
        log("linearized", linearize_colors(meshes), "CAD materials")
    if a.pad_finish:
        log("pad finish", a.pad_finish, "->", finish_pads(meshes, a.pad_finish), "meshes")
    lo, hi = bbox(meshes)
    ext = hi - lo
    log("imported", len(meshes), "meshes, extents", tuple(round(x, 5) for x in ext))

    # 1) units: compare sorted extents with sorted catalog dims (metres)
    scale, fit_err = 1.0, None
    if a.dims:
        scale, fit_err = fit_units(sorted(ext), sorted(d / 1000 for d in a.dims))
        if fit_err > 0.15:
            flags.append(f"footprint off by {fit_err:.0%} vs catalog dims at best unit guess")
        h, want_h = sorted(ext)[0] * scale * 1000, sorted(a.dims)[0]
        if want_h > 0 and not 0.8 <= h / want_h <= 1.25:
            flags.append(f"height {h:.1f} mm vs catalog {want_h:.1f} mm (pins/headers, or a different variant)")
    elif max(ext) > 2.0:
        scale = 0.001
        flags.append("no catalog dims; assumed millimetres")
    if scale != 1.0:
        apply(meshes, Matrix.Scale(scale, 4))

    # 2) orientation: catalog height (3rd dim) vertical; for flat things that means smallest extent on Z.
    #    CAD (STEP) orientation is trusted as-is; only meshes of unknown orientation get turned.
    lo, hi = bbox(meshes)
    ext = hi - lo
    if a.dims and not a.source_zup and a.dims[2] <= min(a.dims[:2]):
        axis = min(range(3), key=lambda i: ext[i])
        if axis == 0:
            apply(meshes, Matrix.Rotation(math.radians(90), 4, "Y"))
        elif axis == 1:
            apply(meshes, Matrix.Rotation(math.radians(90), 4, "X"))

    # 3) base on z=0, centred in x/y
    lo, hi = bbox(meshes)
    apply(meshes, Matrix.Translation(Vector((-(lo.x + hi.x) / 2, -(lo.y + hi.y) / 2, -lo.z))))
    lo, hi = bbox(meshes)
    dims_mm = [round((hi[i] - lo[i]) * 1000, 2) for i in range(3)]

    # per-part recolour (after scaling: part sizes are in real mm now)
    if a.recolor:
        got = recolor_parts(meshes, json.loads(a.recolor))
        log("recolored", got)
        missing = sorted(k for k, n in got.items() if not n)
        if missing:
            flags.append(f"recolor: parts not found {missing}")

    # materials: give bare meshes something visible
    mat = None
    for o in meshes:
        if not o.data.materials:
            mat = mat or default_material()
            o.data.materials.append(mat)
    tris = sum(sum(len(p.vertices) - 2 for p in o.data.polygons) for o in meshes)
    if tris > 300_000:
        flags.append(f"heavy mesh ({tris} tris)")
    if tris < 100:  # a box or clearance envelope
        flags.append(f"very low detail ({tris} triangles): probably a placeholder box, not a realistic model")
    elif tris < 1000:
        flags.append(f"simple model ({tris} triangles)")

    # rename for tidy exports
    base = "".join(c if c.isalnum() else "_" for c in a.name)[:40] or "asset"
    for i, o in enumerate(meshes):
        o.name = base if len(meshes) == 1 else f"{base}_{i:02d}"

    files = {}
    usd = os.path.join(a.out_dir, "asset.usd")
    bpy.ops.wm.usd_export(filepath=usd, export_materials=True, meters_per_unit=1.0)
    files["usd"] = "asset.usd"
    glb = os.path.join(a.out_dir, "model.glb")
    bpy.ops.export_scene.gltf(filepath=glb, export_format="GLB")
    files["glb"] = "model.glb"
    obj = os.path.join(a.out_dir, "mesh", "mesh.obj")
    save_textures(os.path.join(a.out_dir, "mesh"))
    # same frame as asset.usd (Z-up, metres), so poses estimated on this mesh (FoundationPose) apply to the USD as-is
    bpy.ops.wm.obj_export(filepath=obj, export_materials=True, forward_axis="Y", up_axis="Z", path_mode="RELATIVE")
    files["obj"] = "mesh/mesh.obj"
    if not a.no_render:
        render_thumb(meshes, os.path.join(a.out_dir, "thumb.png"))
        files["thumb"] = "thumb.png"
    for v in [v for v in a.views.split(",") if v in ("top", "bottom")]:
        render_ortho(meshes, os.path.join(a.out_dir, f"view_{v}.png"), v)
        files[f"view_{v}"] = f"view_{v}.png"
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(a.out_dir, "scene.blend"))
    files["blend"] = "scene.blend"

    result = {"files": files, "dims_mm": dims_mm, "source_zup": a.source_zup, "scale_factor": scale, "dims_fit": fit_err, "tri_count": tris,
              "parts": len(meshes), "flags": flags, "blender": bpy.app.version_string,
              "obj_up": "Z"}  # mesh/mesh.obj shares asset.usd's frame (older builds wrote Y-up OBJs)
    print("OBJSEARCH_RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
