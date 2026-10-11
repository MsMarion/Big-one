"""Synthetic data generation (SDG) with Isaac Replicator: labeled images of our target objects for YOLO training.

What this teaches (see docs/learning/03_replicator_sdg.md):
  - Replicator = render products + annotators + writers. We use the built-in BasicWriter to save
    rgb / instance segmentation / 2D boxes / depth / camera params for every captured frame.
  - Domain randomization: everything the detector should IGNORE is randomized (lighting, table color,
    clutter, arm pose, camera jitter); what it should LEARN (the can's appearance) is kept real.
  - The randomization is a plain, seeded Python loop: easy to read/debug, and every frame's exact
    parameters are logged to frames.jsonl (reproducible, inspectable).
  - The SAME scene builder as the closed-loop sim (sim/scene.py) -> training images match the sim.

Run (Windows):  .\\run_isaac.ps1 sim\\sdg\\generate_dataset.py --config config\\sdg\\soup_can_v0.yaml
Output:         data/raw/<name>/  (BasicWriter files + frames.jsonl + sdg_config.yaml)
"""

import argparse
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--num-frames", type=int, default=None, help="override num_frames from the config")
args = parser.parse_args()

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # sim/ -> common, scene

import json
import shutil
import time

import numpy as np
import omni.replicator.core as rep
import yaml
from common import REPO, load_yaml, log, shutdown
from pnp_core.catalog import load_fixed_camera, load_robot, robot_name  # noqa: E402
from pnp_core.gripper import GripperModel  # noqa: E402
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.semantics import add_labels
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade
from scene import add_brick, build_scene, color_material, set_material_color, upright_quat_wxyz, xyzw_to_wxyz

from pnp_core.geometry import look_at_optical, matrix_to_quat

PARKED = np.array([0.0, 0.0, -5.0])     # unused objects are parked below the floor (and hidden)


# ---------------------------------------------------------------------------------------------- objects
class SdgObject:
    """A spawned prop we can place/hide each frame. Footprint/height come from its real bounding box."""

    def __init__(self, name, catalog_name, catalog, path, table_z, label=None):
        cfg = dict(catalog[catalog_name])
        self.up_axis = cfg.get("usd_up_axis", "Z")
        self.procedural = bool(cfg.get("brick") or cfg.get("box"))
        # Props are placed kinematically for the picture: plain referenced USD (or a procedural brick / box), no
        # rigid body.
        if cfg.get("brick"):
            add_brick(path, cfg)
        elif cfg.get("box"):
            UsdGeom.Xform.Define(get_current_stage(), path)
            cube = UsdGeom.Cube.Define(get_current_stage(), f"{path}/body")
            cube.CreateSizeAttr(1.0)
            cube.AddScaleOp().Set(Gf.Vec3f(*map(float, cfg["dims_m"])))
            UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(
                color_material(f"{path}/material", cfg.get("color", [0.3, 0.35, 0.6]), roughness=0.6))
        else:
            add_reference_to_stage(usd_path=f"{get_assets_root_path()}/{cfg['usd']}", prim_path=path)
        self.prim = get_current_stage().GetPrimAtPath(path)
        self.label = label
        if label:
            add_labels(self.prim, labels=[label], instance_name="class")
        self.xform = SingleXFormPrim(path, name=name)
        # Untransformed = in the asset's own frame (ComputeLocalBound would include the upright rotation
        # spawn_object already applied, mixing up height and footprint).
        rng = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
                                ).ComputeUntransformedBound(self.prim).ComputeAlignedRange()
        ext = np.array(rng.GetMax()) - np.array(rng.GetMin())
        self.extents = ext                                    # asset-frame x, y, z sizes (for catalog checks)
        # asset frame -> upright: for Y-up assets, height = y extent, footprint = x, z extents
        self.height = float(ext[1] if "Y" in self.up_axis.upper() else ext[2])
        foot = (ext[0], ext[2]) if "Y" in self.up_axis.upper() else (ext[0], ext[1])
        self.radius = 0.5 * float(np.hypot(*foot))          # conservative circle around the footprint
        self.table_z = table_z
        self.catalog_name = catalog_name

    def place(self, x, y, yaw):
        # Axis_Aligned assets have their origin at the bounding-box center -> center sits height/2 above the table
        self.xform.set_world_pose(position=np.array([x, y, self.table_z + self.height / 2 + 0.0005]),
                                  orientation=upright_quat_wxyz(self.up_axis, yaw))
        UsdGeom.Imageable(self.prim).MakeVisible()

    def recolor(self, rgb):
        """Procedural objects: a new color (bricks: the class is the SHAPE, colors vary)."""
        if self.procedural:
            set_material_color(f"{self.prim.GetPath()}/material", rgb)

    def park(self):
        self.xform.set_world_pose(position=PARKED)
        UsdGeom.Imageable(self.prim).MakeInvisible()


def sample_layout(rng, objs, region, gap, max_tries):
    """Non-overlapping XY positions (circle footprints). Objects that don't fit are left out."""
    placed = []
    for o in objs:
        for _ in range(max_tries):
            x, y = rng.uniform(*region["x"]), rng.uniform(*region["y"])
            if all(np.hypot(x - px, y - py) > o.radius + pr + gap for _, px, py, pr in placed):
                placed.append((o, x, y, o.radius))
                break
    return [(o, x, y, float(rng.uniform(-np.pi, np.pi))) for o, x, y, _ in placed]


# ----------------------------------------------------------------------------------------------- lights
def set_light(prim_path, intensity, kelvin=None):
    light = UsdLux.LightAPI(get_current_stage().GetPrimAtPath(prim_path))
    light.CreateIntensityAttr().Set(float(intensity))
    if kelvin is not None:
        light.CreateEnableColorTemperatureAttr().Set(True)
        light.CreateColorTemperatureAttr().Set(float(kelvin))


def load_sdg_config(path) -> dict:
    """SDG configs can inherit: `base: soup_can_v0.yaml` + only the keys that change (name, seed, ...)."""
    cfg = load_yaml(path)
    if "base" in cfg:
        base = load_sdg_config(Path(path).parent / cfg.pop("base"))

        def merge(a, b):
            return {k: merge(a[k], v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
                    for k, v in {**a, **b}.items()}
        cfg = merge(base, cfg)
    return cfg


def main() -> int:
    cfg = load_sdg_config(args.config)
    n_frames = args.num_frames or cfg["num_frames"]
    rng = np.random.default_rng(cfg["seed"])
    out = REPO / "data" / "raw" / cfg["name"]
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / "sdg_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    catalog = load_yaml("config/objects.yaml")
    scene_cfg = load_yaml(cfg["scene"])
    robot_cfg = load_robot(robot_name(scene_cfg))
    cam_cfg = load_fixed_camera(scene_cfg)
    gripper = GripperModel(robot_cfg["gripper"])

    # --- scene: same builder as the sim (table, FR3, camera with D435i intrinsics, lights) --------------
    h = build_scene(cfg["scene"], with_camera_rig=False)
    stage = get_current_stage()
    table_z = h.cfg["table"]["center"][2] + h.cfg["table"]["size"][2] / 2
    h.target.set_world_pose(position=PARKED)                 # the scene's single target is not used here
    UsdGeom.Imageable(stage.GetPrimAtPath(h.paths["target"])).MakeInvisible()
    # Labeled objects: `targets.objects` (a list, one YOLO class each) or the older single `targets.object`.
    # Each gets `count[1]` instances; a frame draws its targets from this pool, so classes mix in one image.
    target_names = cfg["targets"].get("objects") or [cfg["targets"]["object"]]
    cans = [SdgObject(f"{n}{i}", n, catalog, f"/World/sdg/{n}{i}", table_z, catalog[n]["yolo_class"])
            for n in target_names for i in range(cfg["targets"]["count"][1])]
    distractors = [SdgObject(f"distractor_{n}", n, catalog, f"/World/sdg/{n}", table_z)
                   for n in cfg["distractors"]["objects"]]
    for d in distractors:
        log(f"[sdg]   {d.catalog_name:16s} asset extents x/y/z = {np.round(d.extents * 1000).astype(int).tolist()} mm, "
            f"up={d.up_axis} -> height {d.height * 1000:.0f} mm")
    bad = [d.catalog_name for d in distractors if not (0.01 < d.height < 1.0)]
    if bad:
        raise RuntimeError(f"distractor assets with invalid bounds (missing/empty USD?): {bad}")
    spheres = []
    for i in range(cfg["lights"]["extra_spheres"][1]):
        s = UsdLux.SphereLight.Define(stage, Sdf.Path(f"/World/Lights/sdg_sphere{i}"))
        s.CreateRadiusAttr(0.1)
        s.AddTranslateOp()
        spheres.append(s)
    key = stage.GetPrimAtPath("/World/Lights/key")
    table = h.world.scene.get_object("table")
    log(f"[sdg] targets: {[(c.catalog_name, c.label, round(c.height, 3)) for c in cans[::cfg['targets']['count'][1]]]}; "
        f"distractors: {[(d.catalog_name, round(d.height, 3)) for d in distractors]}")

    # --- Replicator: writer on the camera's render product ----------------------------------------------
    rep.orchestrator.set_capture_on_play(False)               # we decide when to capture
    rp = rep.create.render_product(h.paths["camera"], (cam_cfg["width"], cam_cfg["height"]))
    writer = rep.writers.get("BasicWriter")
    writer.initialize(output_dir=str(out), semantic_types=["class"], colorize_instance_segmentation=False,
                      **cfg["writer"])
    writer.attach([rp])

    h.world.reset()                                            # physics on: lets us pose the arm
    names = h.robot.dof_names
    home = dict(zip(robot_cfg["arm"]["joints"], robot_cfg["arm"]["home"]))
    frames_log = open(out / "frames.jsonl", "w")
    t0 = time.perf_counter()

    for i in range(n_frames):
        p = {"frame": i}
        # 1. targets (sometimes none) + distractors, non-overlapping
        tc = cfg["targets"]
        n_cans = 0 if rng.random() < tc["p_zero"] else int(rng.integers(max(1, tc["count"][0]), tc["count"][1] + 1))
        n_dis = int(rng.integers(cfg["distractors"]["count"][0], cfg["distractors"]["count"][1] + 1))
        chosen = list(rng.permutation(cans))[:n_cans] + list(rng.permutation(distractors))[:n_dis]
        layout = sample_layout(rng, chosen, tc["region"], cfg["placement"]["min_gap_m"], cfg["placement"]["max_tries"])
        for o in cans + distractors:
            o.park()
        palette = cfg.get("colors")             # procedural objects: a random color per frame (from a palette)
        for o, x, y, yaw in layout:
            o.place(x, y, yaw)
            if palette:
                o.recolor(np.clip(np.array(palette[rng.integers(len(palette))]) + rng.uniform(-0.06, 0.06, 3), 0, 1))
        p["objects"] = [{"name": o.catalog_name, "label": o.label if o in cans else None,
                         "xy": [round(x, 4), round(y, 4)], "yaw": round(yaw, 4)} for o, x, y, yaw in layout]

        # 2. robot: home + noise (the arm is often in view), random gripper width
        q = np.array([home.get(n, 0.0) for n in names])
        noise = cfg["robot"]["joint_noise_rad"]
        q += np.array([rng.uniform(-noise, noise) if n in home else 0.0 for n in names])
        w = rng.uniform(*cfg["robot"]["gripper_width_m"])
        q = np.array([gripper.command(w) if n in robot_cfg["gripper"]["joints"] else v for n, v in zip(names, q)])
        h.robot.set_joint_positions(q)
        h.robot.set_joint_velocities(np.zeros_like(q))
        p["joints"] = [round(float(v), 4) for v in q]

        # 3. camera pose jitter around the configured mount
        c = cfg["camera"]
        eye = np.array(cam_cfg["mount"]["position"]) + rng.uniform(-1, 1, 3) * c["position_jitter_m"]
        tgt = np.array(cam_cfg["mount"]["look_at"]) + rng.uniform(-1, 1, 3) * c["look_at_jitter_m"]
        h.camera.set_world_pose(position=eye, orientation=xyzw_to_wxyz(matrix_to_quat(look_at_optical(eye, tgt))),
                                camera_axes="ros")
        p["camera"] = {"eye": eye.round(4).tolist(), "look_at": tgt.round(4).tolist()}

        # 4. lighting
        L = cfg["lights"]
        kelvin = rng.uniform(*L["color_temperature_k"])
        set_light("/World/Lights/dome", rng.uniform(*L["dome_intensity"]), kelvin)
        set_light("/World/Lights/key", rng.uniform(*L["key_intensity"]), rng.uniform(*L["color_temperature_k"]))
        elev, azim = rng.uniform(*L["key_elevation_deg"]), rng.uniform(0, 360)
        key.GetAttribute("xformOp:rotateXYZ").Set(Gf.Vec3f(-elev, 0.0, azim))
        n_sph = int(rng.integers(L["extra_spheres"][0], L["extra_spheres"][1] + 1))
        for k, s in enumerate(spheres):
            on = k < n_sph
            s.GetIntensityAttr().Set(float(rng.uniform(*L["sphere_intensity"])) if on else 0.0)
            s.GetPrim().GetAttribute("xformOp:translate").Set(
                Gf.Vec3d(rng.uniform(-0.3, 1.2), rng.uniform(-1.0, 1.0), rng.uniform(0.5, 1.8)))
        p["lights"] = {"kelvin": round(kelvin), "key_elev": round(elev, 1), "key_azim": round(azim, 1), "spheres": n_sph}

        # 5. table color
        T = cfg["table"]
        col = np.clip(np.array(T["colors"][rng.integers(len(T["colors"]))]) + rng.uniform(-1, 1, 3) * T["jitter"], 0, 1)
        table.get_applied_visual_material().set_color(col)
        p["table_rgb"] = col.round(3).tolist()

        # propagate the new poses through physics (2 steps = 17 ms: objects don't move noticeably), then capture
        for _ in range(2):
            h.world.step(render=False)
        rep.orchestrator.step(delta_time=0.0, rt_subframes=cfg["rt_subframes"], pause_timeline=False)
        frames_log.write(json.dumps(p) + "\n")
        if (i + 1) % 50 == 0 or i + 1 == n_frames:
            dt = time.perf_counter() - t0
            log(f"[sdg] {i + 1}/{n_frames} frames, {dt:.0f} s ({(i + 1) / dt:.1f} fps)")

    rep.orchestrator.wait_until_complete()
    frames_log.close()
    log(f"[sdg] done: {n_frames} frames -> {out}")
    return 0


if __name__ == "__main__":
    shutdown(app, main())
