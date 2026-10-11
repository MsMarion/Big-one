"""Builds the pick & place stage: table, FR3 robot, soup can, D435i-like camera, goal marker, lights.

What this teaches:
  - The World object owns physics (dt, solver) and the stage.
  - Assets are *referenced* into the stage (add_reference_to_stage), then physics APIs are applied.
  - Physics realism for grasping: friction materials, solver iterations, force-limited gripper drive.
  - Cameras: intrinsics (focal length / aperture) and pose, using the SAME config the ROS side uses.

Used by sim/run_sim.py (with ROS) and sim/sdg/generate_dataset.py (synthetic data, no ROS),
so training images and the closed-loop sim see the same scene.
Must be imported after SimulationApp is created.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root -> pnp_core importable

from isaacsim.core.api import World
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid, VisualCuboid, VisualCylinder
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
from isaacsim.core.utils.semantics import add_labels
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.sensors.camera import Camera
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

from pnp_core.geometry import camera_mount_T, make_transform, matrix_to_quat
from pnp_core.calibration import board_size_m, mount_T
from pnp_core.brick import brick_geometry
from pnp_core.catalog import load_fixed_camera, load_robot, robot_name
from pnp_core.gripper import GripperModel
from pnp_core.wrist import T_parent_link, T_parent_optical, housing_in_description, wrist_camera
from common import load_yaml, log

# USD camera optics (focalLength, horizontal/verticalAperture) are in MILLIMETERS by convention
# ("tenths of a scene unit" for cm stages), even when the stage is in meters. We set them directly:
# fx [px] = focalLength / horizontalAperture * width.  20.955 mm is USD's default aperture (35 mm film).
H_APERTURE_MM = 20.955


def set_pinhole(camera_prim, fx: float, fy: float, width: int, height: int, cx=None, cy=None) -> None:
    """Configure a USD camera so it renders with pinhole intrinsics fx, fy, cx, cy (default: principal point at the
    center). An off-center principal point (a real unit's calibration: the plate twin's D435i has cy = 256 of 480) is
    the aperture offset: USD shifts the film window, u0 = W/2 - hOffset/px, v0 = H/2 + vOffset/px."""
    px_mm = H_APERTURE_MM / width
    v_ap = px_mm * height * fx / fy
    camera_prim.GetAttribute("horizontalAperture").Set(H_APERTURE_MM)
    camera_prim.GetAttribute("verticalAperture").Set(v_ap)
    camera_prim.GetAttribute("focalLength").Set(fx * px_mm)
    if cx is not None and cy is not None:
        from pxr import Sdf
        for name, val in (("horizontalApertureOffset", (width / 2 - cx) * px_mm),
                          ("verticalApertureOffset", (cy - height / 2) * v_ap / height)):
            attr = camera_prim.GetAttribute(name) or camera_prim.CreateAttribute(name, Sdf.ValueTypeNames.Float)
            attr.Set(float(val))


def xyzw_to_wxyz(q) -> np.ndarray:
    return np.array([q[3], q[0], q[1], q[2]])


from scene_math import upright_quat_wxyz  # noqa: E402  (stand up, then yaw; see the Gf order gotcha there)


@dataclass
class SceneHandles:
    world: World
    robot: SingleArticulation
    target: SingleRigidPrim
    camera: Camera
    cfg: dict = field(default_factory=dict)
    paths: dict = field(default_factory=dict)
    obstacles: dict = field(default_factory=dict)     # id -> DynamicCuboid (Phase 5 box pool)
    extras: dict = field(default_factory=dict)        # name -> SingleRigidPrim: the scene's extra_objects


def _jaw_collision(stage, link_path: str, cc: dict) -> int:
    """Collision shapes that follow the jaws' real surfaces (robot profile sim.gripper_collision).

    The URDF importer gives every link a convex DECOMPOSITION with PhysX's defaults (up to 32 hulls at 10 % error).
    Fine for avoiding obstacles, but on the gripper it decides WHERE the object is touched: the SO-101's hulls stood
    ~1.9 mm off the fixed jaw, so it held a brick across a visible gap (tools/jaw_gap_check.py). Options:
      sdf                  PhysX signed-distance-field collision of the mesh itself (sdf_resolution voxels across it)
      convexDecomposition  tighter hulls: more of them, small error, shrink-wrapped onto the mesh
    The importer instances its meshes: collision prims are read-only instance proxies until de-instanced.
    -> how many collision meshes were changed."""
    root = stage.GetPrimAtPath(f"{link_path}/collisions")
    if not root:
        return 0
    while True:                                          # de-instance (nested instances show up level by level)
        inst = [p for p in Usd.PrimRange(root) if p.IsInstance()]
        if not inst:
            break
        for p in inst:
            p.SetInstanceable(False)
    approx, n = cc.get("approximation", "sdf"), 0
    for p in Usd.PrimRange(root):
        if not p.HasAPI(UsdPhysics.MeshCollisionAPI):
            continue
        UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Set(approx)
        if approx == "sdf":
            PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(p).CreateSdfResolutionAttr(int(cc.get("sdf_resolution", 256)))
        else:
            d = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(p)
            d.CreateMaxConvexHullsAttr(int(cc.get("max_hulls", 64)))
            d.CreateErrorPercentageAttr(float(cc.get("error_percentage", 1.0)))
            d.CreateVoxelResolutionAttr(int(cc.get("voxel_resolution", 1000000)))
            d.CreateShrinkWrapAttr(True)
        pc = PhysxSchema.PhysxCollisionAPI.Apply(p)       # contacts start 1 mm out, rest at the surface itself
        pc.CreateContactOffsetAttr(float(cc.get("contact_offset_m", 0.001)))
        pc.CreateRestOffsetAttr(0.0)
        n += 1
    return n


def _apply_friction(stage, prim_path: str, material_path: str):
    binding = UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(prim_path))
    binding.Bind(UsdShade.Material(stage.GetPrimAtPath(material_path)),
                 UsdShade.Tokens.weakerThanDescendants, "physics")


def add_brick(prim_path: str, obj_cfg: dict, color=None) -> None:
    """A LEGO-dimensioned brick (objects.yaml `brick: {studs: [nx, ny]}`): a box body + studs on top. Origin at the
    center of the whole bounding box incl. studs (= dims_m, like the YCB assets). Collision = the body only, with
    millimeter contact offsets: PhysX's default (~2 cm, scaled from the scene) is larger than the brick itself."""
    stage = get_current_stage()
    g = brick_geometry(obj_cfg)                  # pnp_core/brick.py: the same geometry FoundationPose's mesh has
    (L, W, H), stud_h = g.body_size, g.stud_height
    rgb = Gf.Vec3f(*map(float, color if color is not None else obj_cfg.get("color", [0.8, 0.1, 0.1])))
    UsdGeom.Xform.Define(stage, prim_path)
    mat = color_material(f"{prim_path}/material", rgb, roughness=0.35)    # glossy ABS plastic
    body = UsdGeom.Cube.Define(stage, f"{prim_path}/body")
    body.CreateSizeAttr(1.0)
    body.AddTranslateOp().Set(Gf.Vec3d(*map(float, g.body_center)))
    body.AddScaleOp().Set(Gf.Vec3f(L, W, H))
    body.CreateDisplayColorAttr([rgb])
    UsdPhysics.CollisionAPI.Apply(body.GetPrim())
    pc = PhysxSchema.PhysxCollisionAPI.Apply(body.GetPrim())
    pc.CreateContactOffsetAttr(0.001)
    pc.CreateRestOffsetAttr(0.0)
    ny = obj_cfg["brick"]["studs"][1]
    for k, c in enumerate(g.stud_centers):
        st = UsdGeom.Cylinder.Define(stage, f"{prim_path}/stud_{k // ny}_{k % ny}")
        st.CreateRadiusAttr(g.stud_radius)
        st.CreateHeightAttr(stud_h)
        st.CreateAxisAttr("Z")
        st.AddTranslateOp().Set(Gf.Vec3d(*map(float, c)))
        st.CreateDisplayColorAttr([rgb])
        UsdShade.MaterialBindingAPI.Apply(st.GetPrim()).Bind(mat)
    UsdShade.MaterialBindingAPI.Apply(body.GetPrim()).Bind(mat)


def color_material(path: str, rgb, roughness: float = 0.5) -> UsdShade.Material:
    """A plain colored material (UsdPreviewSurface). displayColor alone renders pale and flat in RTX.
    Recolor later with set_material_color(path, rgb)."""
    stage = get_current_stage()
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, rgb)))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def set_material_color(path: str, rgb) -> None:
    sh = UsdShade.Shader(get_current_stage().GetPrimAtPath(f"{path}/surface"))
    sh.GetInput("diffuseColor").Set(Gf.Vec3f(*map(float, rgb)))


def spawn_object(world: World, name: str, obj_cfg: dict, xy, yaw: float, prim_path: str,
                 table_z: float, physics_material_path: str | None, label: str | None) -> SingleRigidPrim:
    """Reference a YCB object (or build a procedural brick), make it a dynamic rigid body, stand it on the table."""
    stage = get_current_stage()
    if obj_cfg.get("brick"):
        add_brick(prim_path, obj_cfg)
    else:
        add_reference_to_stage(usd_path=f"{get_assets_root_path()}/{obj_cfg['usd']}", prim_path=prim_path)
    prim = stage.GetPrimAtPath(prim_path)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(obj_cfg["mass_kg"]))
    if physics_material_path:
        _apply_friction(stage, prim_path, physics_material_path)
    if label:
        add_labels(prim, labels=[label], instance_name="class")
    height = obj_cfg.get("dims_m", [0, 0, 0.1])[2]
    rigid = SingleRigidPrim(prim_path=prim_path, name=name)
    rigid.set_world_pose(position=np.array([xy[0], xy[1], table_z + height / 2 + 0.001]),
                         orientation=upright_quat_wxyz(obj_cfg.get("usd_up_axis", "Z"), yaw))
    return world.scene.add(rigid)


def build_scene(scene_cfg_path="config/scenes/basic.yaml", with_robot: bool = True,
                with_camera_rig: bool = True) -> SceneHandles:
    cfg = load_yaml(scene_cfg_path)
    robot_cfg = load_robot(robot_name(cfg))       # the scene names its robot (default fr3)
    cam_cfg = load_fixed_camera(cfg)              # config/camera_d435i.yaml + the scene's mount override
    objects = load_yaml("config/objects.yaml")
    phys = cfg["physics"]

    world = World(stage_units_in_meters=1.0, physics_dt=phys["dt"], rendering_dt=1.0 / cam_cfg["fps"])
    stage = get_current_stage()
    table_top = cfg["table"]["center"][2] + cfg["table"]["size"][2] / 2

    # --- environment: floor, light, table -------------------------------------------------------
    world.scene.add_default_ground_plane(z_position=table_top - cfg["table"]["height_above_floor"])
    lights = cfg.get("lights") or {}          # a twin of a real room (so101_plate.yaml) sets its own, dimmer
    UsdLux.DomeLight.Define(stage, Sdf.Path("/World/Lights/dome")).CreateIntensityAttr(float(lights.get("dome", 350.0)))
    key = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Lights/key"))
    key.CreateIntensityAttr(float(lights.get("key", 1200.0)))
    key.AddRotateXYZOp().Set(Gf.Vec3f(*map(float, lights.get("key_rotate_xyz_deg", [-35.0, 20.0, 0.0]))))

    fr = phys["friction"]
    grip_mat = PhysicsMaterial("/World/Physics/grip", static_friction=fr["static"],
                               dynamic_friction=fr["dynamic"], restitution=fr["restitution"])
    grid_png = TARGET_DIR / f"{cfg.get('name', '')}_floor_grid.png"
    if cfg.get("floor_grid") and grid_png.exists():
        # a grid painted on the board (tools/make_calibration_targets.py): origin at the base, 0.3 mm above the top
        # (under the AprilTags at 0.8 mm), visual only
        sx, sy = cfg["table"]["size"][0], cfg["table"]["size"][1]
        add_textured_plate("/World/floor_grid", _flat_on_table(cfg["table"]["center"][:2], 0.0, sx, sy, table_top - 0.0005),
                           sx, sy, grid_png)
    world.scene.add(FixedCuboid("/World/table", name="table", position=np.array(cfg["table"]["center"]),
                                scale=np.array(cfg["table"]["size"]), size=1.0,
                                color=np.array(cfg["table"].get("color", [0.55, 0.45, 0.35])), physics_material=grip_mat))

    # --- robot ------------------------------------------------------------------------------------
    robot = None
    rpath = robot_cfg["sim"]["prim_path"]
    art_path, link_root = rpath, rpath
    if with_robot:
        # Isaac's asset library (FR3) or a USD generated from the robot's URDF (sim/import_urdf.py, repo-relative)
        usd = robot_cfg["sim"]["usd"]
        local = REPO_ROOT / usd
        add_reference_to_stage(usd_path=str(local) if local.exists() else f"{get_assets_root_path()}/{usd}",
                               prim_path=rpath)
        # URDF-imported robots keep their links under a sub-prim and the articulation root on a fixed root joint
        link_root = f"{rpath}/{robot_cfg['sim']['link_root']}" if robot_cfg["sim"].get("link_root") else rpath
        art_path = f"{rpath}/{robot_cfg['sim']['articulation_root']}" if robot_cfg["sim"].get("articulation_root") else rpath
        art = PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(art_path))
        art.CreateSolverPositionIterationCountAttr(phys["solver_position_iterations"])
        art.CreateSolverVelocityIterationCountAttr(1)
        joints = {p.GetName(): p for p in Usd.PrimRange(stage.GetPrimAtPath(rpath)) if p.IsA(UsdPhysics.Joint)}
        drive_cfg = robot_cfg["sim"].get("drive")
        if drive_cfg:       # servo-like position drives for the arm + gripper (the URDF importer's are placeholders)
            for jn in list(robot_cfg["arm"]["joints"]) + [robot_cfg["gripper"]["driven_joint"]]:
                d = UsdPhysics.DriveAPI.Get(joints[jn], "angular")
                d.GetStiffnessAttr().Set(float(drive_cfg["stiffness"]))
                d.GetDampingAttr().Set(float(drive_cfg["damping"]))
                d.GetMaxForceAttr().Set(float(drive_cfg["max_force"]))
        # Force-limit the driven finger / jaw like the real gripper (Franka Hand ~70 N continuous)
        gj = joints.get(robot_cfg["gripper"]["driven_joint"])
        for kind in ("linear", "angular"):
            d = UsdPhysics.DriveAPI.Get(gj, kind) if gj else None
            if d and d.GetMaxForceAttr():
                limit = robot_cfg["sim"].get("gripper_max_force", robot_cfg["gripper"]["max_force_n"])
                d.GetMaxForceAttr().Set(float(limit))
        cc = robot_cfg["sim"].get("gripper_collision")
        for finger in robot_cfg["sim"].get("gripper_links", ["fr3_leftfinger", "fr3_rightfinger"]):
            _apply_friction(stage, f"{link_root}/{finger}", grip_mat.prim_path)
            if cc:
                n = _jaw_collision(stage, f"{link_root}/{finger}", cc)
                log(f"[scene] {finger}: {n} collision mesh(es) -> {cc.get('approximation', 'sdf')}")
        if cfg.get("robot_color"):            # the twin: the real arm's printed parts (the USD's are yellow)
            _tint(stage, rpath, cfg["robot_color"])
        robot = world.scene.add(SingleArticulation(prim_path=art_path, name=robot_cfg["name"]))

    # --- target object ------------------------------------------------------------------------------
    tcfg = cfg["target"]
    obj_cfg = objects[tcfg["object"]]
    target = spawn_object(world, "target", obj_cfg, tcfg["default_xy"], 0.0, "/World/objects/target",
                          table_top, grip_mat.prim_path, obj_cfg.get("yolo_class"))

    # --- extra pickable objects (scene key extra_objects: a second / third object on the table) -----------
    # each its own rigid body at /World/objects/<name>, labeled for the detector; its ground-truth TF frame is <name>
    extras = {}
    for e in cfg.get("extra_objects", []):
        ocfg = objects[e["object"]]
        extras[e["name"]] = spawn_object(world, e["name"], ocfg, e["default_xy"], float(e.get("yaw", 0.0)),
                                         f"/World/objects/{e['name']}", table_top, grip_mat.prim_path,
                                         ocfg.get("yolo_class"))

    # --- goal marker (visual only: no collision) ----------------------------------------------------
    g = cfg["goal"]
    world.scene.add(VisualCylinder("/World/goal_marker", name="goal_marker",
                                   position=np.array([g["xy"][0], g["xy"][1], table_top + 0.0005]),
                                   radius=g["marker_radius_m"], height=0.001, color=np.array([0.1, 0.8, 0.2])))

    # --- obstacle pool (Phase 5, Layer A): real rigid boxes, parked off the table until a layout uses them ---
    obstacles = {}
    ocfg = cfg.get("obstacles") or {}
    for i, spec in enumerate(ocfg.get("pool", [])):
        obstacles[spec["id"]] = world.scene.add(DynamicCuboid(
            f"/World/obstacles/{spec['id']}", name=f"obstacle_{spec['id']}", size=1.0,
            scale=np.array(spec["size"]), position=_park_position(cfg, i, spec["size"]),
            mass=float(ocfg["mass_kg"]), color=np.array(ocfg.get("color", [0.3, 0.3, 0.6]))))

    # --- camera: same intrinsics + mount as config/camera_d435i.yaml (the ROS side uses it too) -------
    R_opt = camera_mount_T(cam_cfg["mount"])[:3, :3]
    camera = Camera(prim_path="/World/camera_link/camera", name="d435i",
                    resolution=(cam_cfg["width"], cam_cfg["height"]), frequency=cam_cfg["fps"])
    camera.set_world_pose(position=np.array(cam_cfg["mount"]["position"]),
                          orientation=xyzw_to_wxyz(matrix_to_quat(R_opt)), camera_axes="ros")
    set_pinhole(camera.prim, cam_cfg["fx"], cam_cfg["fy"], cam_cfg["width"], cam_cfg["height"], cam_cfg.get("cx"),
                cam_cfg.get("cy"))
    camera.set_clipping_range(0.05, 10.0)

    if with_camera_rig:   # visual only; synthetic-data generation leaves it out (camera jitter would see it)
        if (cfg.get("camera_rig") or {}).get("urdf"):
            add_urdf_rig(cfg["camera_rig"], table_top)       # the real stand (a CAD export, tools/fit_camera_mount.py)
        else:
            add_camera_rig(cam_cfg, R_opt, table_top - cfg["table"]["height_above_floor"])

    paths = {"robot": art_path, "robot_links": link_root, "target": "/World/objects/target",
             "camera": "/World/camera_link/camera", "table": "/World/table"}
    if with_robot and robot_cfg.get("wrist_camera"):
        paths["wrist_camera"] = add_wrist_camera(wrist_camera(robot_cfg), link_root)
    # calibration boards / tags of this cell (synthetic data leaves them out: the detector shouldn't learn them)
    targets = cfg.get("calibration_targets", "config/calibration_targets.yaml" if robot_cfg["name"] == "fr3" else None)
    if with_robot and with_camera_rig and targets:
        paths.update(add_calibration_targets(link_root, table_top, targets))
    paths["extra_objects"] = {e["name"]: f"/World/objects/{e['name']}" for e in cfg.get("extra_objects", [])}
    return SceneHandles(world, robot, target, camera, cfg, paths, obstacles, extras)


def _tint(stage, root: str, rgb) -> None:
    """One matte material over everything under `root` (stronger than the asset's own bindings)."""
    mat = UsdShade.Material.Define(stage, Sdf.Path("/World/Looks/robot_tint"))
    sh = UsdShade.Shader.Define(stage, Sdf.Path("/World/Looks/robot_tint/shader"))
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, rgb)))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(root)).Bind(mat, UsdShade.Tokens.strongerThanDescendants)


def add_wrist_camera(wcfg: dict, robot_path: str) -> str:
    """Eye-in-hand camera: a camera prim UNDER the hand link, so it moves with the hand. Its local pose comes from
    pnp_core.wrist, the same transform the URDF gets (bringup.launch.py): sim image and ROS TF agree by
    construction. Plus a visual-only housing so you can see it (unless the robot model has the camera itself)."""
    parent = f"{robot_path}/{wcfg['mount']['parent']}"
    T = T_parent_optical(wcfg)
    cam_path = f"{parent}/wrist_camera"
    # A PLAIN USD camera, not isaacsim.sensors.camera.Camera: that class creates its own render product, so the
    # camera would be rendered twice (that one + the ROS graph's). Rendering cost showed up as camera images lagging
    # behind their timestamps (docs/learning/14_wrist_camera.md).
    # USD cameras look along -z with +y up; ROS optical: +z forward, +y down -> rotate 180 deg about x.
    R_usd = T[:3, :3] @ np.diag([1.0, -1.0, -1.0])
    cam = UsdGeom.Camera.Define(get_current_stage(), cam_path)
    xf = UsdGeom.Xformable(cam.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, T[:3, 3])))
    qw, qx, qy, qz = xyzw_to_wxyz(matrix_to_quat(R_usd))
    xf.AddOrientOp().Set(Gf.Quatf(float(qw), float(qx), float(qy), float(qz)))
    set_pinhole(cam.GetPrim(), wcfg["fx"], wcfg["fy"], wcfg["width"], wcfg["height"], wcfg.get("cx"), wcfg.get("cy"))
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.02, 2.0))
    if housing_in_description(wcfg):                       # the robot's USD already has the camera (from its URDF)
        return cam_path
    Tl = T_parent_link(wcfg)
    sx, sy, sz = wcfg["mount"]["body_size"]
    body = Tl[:3, 3] - Tl[:3, 0] * sx / 2                  # housing behind the lens (camera_link x = forward)
    VisualCuboid(f"{parent}/wrist_camera_body", name="wrist_camera_body", translation=body,
                 orientation=xyzw_to_wxyz(matrix_to_quat(Tl[:3, :3])), scale=np.array([sx, sy, sz]), size=1.0,
                 color=np.array([0.3, 0.3, 0.32]))
    return cam_path


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = REPO_ROOT / "models" / "calibration"


def add_textured_plate(path: str, T_parent_plate: np.ndarray, width: float, height: float, png: Path) -> None:
    """A flat, visual-only quad with an image on it (a printed calibration target). Plate frame = OpenCV's target
    frame: x along the image's columns, y along its rows (image top = y 0), z INTO the plate; the printed face is
    seen from -z. Double-sided so it renders from either side."""
    stage = get_current_stage()
    xf = UsdGeom.Xform.Define(stage, path)
    xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, T_parent_plate[:3, 3])))
    qw, qx, qy, qz = xyzw_to_wxyz(matrix_to_quat(T_parent_plate[:3, :3]))
    xf.AddOrientOp().Set(Gf.Quatf(float(qw), float(qx), float(qy), float(qz)))
    mesh = UsdGeom.Mesh.Define(stage, f"{path}/quad")
    mesh.CreatePointsAttr([Gf.Vec3f(0, 0, 0), Gf.Vec3f(width, 0, 0), Gf.Vec3f(width, height, 0), Gf.Vec3f(0, height, 0)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateDoubleSidedAttr(True)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set([Gf.Vec2f(0, 1), Gf.Vec2f(1, 1), Gf.Vec2f(1, 0), Gf.Vec2f(0, 0)])     # image top (row 0) at y = 0
    mat = UsdShade.Material.Define(stage, f"{path}/material")
    shader = UsdShade.Shader.Define(stage, f"{path}/material/surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    reader = UsdShade.Shader.Define(stage, f"{path}/material/st_reader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex = UsdShade.Shader.Define(stage, f"{path}/material/texture")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(png))
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)


def _flat_on_table(xy, yaw_deg, w, h, table_top):
    """Plate frame for a target lying face-up on the table, centered at xy: x along yaw, z down into the table."""
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    x, z = np.array([c, s, 0.0]), np.array([0.0, 0.0, -1.0])
    y = np.cross(z, x)
    center = np.array([xy[0], xy[1], table_top + 0.0008])
    return make_transform(np.column_stack([x, y, z]), center - x * w / 2 - y * h / 2)


def add_calibration_targets(robot_path: str, table_top: float, targets_cfg="config/calibration_targets.yaml") -> dict:
    """The calibration targets of config/calibration_targets.yaml (images: tools/make_calibration_targets.py):
    the hand board held by the gripper and the table board (both hidden until a calibration shows them: sim command
    calibration_targets), and the AprilTag(s) lying on the table (always there: the drift reference)."""
    t = load_yaml(targets_cfg)
    out = {}
    if not (TARGET_DIR / "hand_board.png").exists():
        return out                                   # images not generated yet
    hb = t["hand_board"]
    if not (TARGET_DIR / hb.get("image", "hand_board.png")).exists():
        return out                                   # this cell's images not generated yet
    w, h = board_size_m(hb)
    out["hand_board"] = f"{robot_path}/{hb['mount']['parent']}/calibration_board"
    add_textured_plate(out["hand_board"], mount_T(hb), w, h, TARGET_DIR / hb.get("image", "hand_board.png"))
    UsdGeom.Imageable(get_current_stage().GetPrimAtPath(out["hand_board"])).MakeInvisible()
    tb = t.get("table_board")
    w, h = board_size_m(tb) if tb else (0, 0)
    out["table_board"] = "/World/calibration/table_board" if tb else None      # hidden too: a real one is removed after calibrating
    if tb:
        add_textured_plate(out["table_board"],
                           _flat_on_table(tb["sim_pose"]["xy"], tb["sim_pose"]["yaw_deg"], w, h, table_top), w, h,
                           TARGET_DIR / tb.get("image", "table_board.png"))
        UsdGeom.Imageable(get_current_stage().GetPrimAtPath(out["table_board"])).MakeInvisible()
    a = t["apriltags"]
    side = a["size_m"] * 1.5                          # the image has a 25 % white border on each side
    for tag in a["tags"]:
        add_textured_plate(f"/World/calibration/apriltag_{tag['id']}",
                           _flat_on_table(tag["sim_xy"], tag["sim_yaw_deg"], side, side, table_top), side, side,
                           TARGET_DIR / f"{a.get('image_prefix', '')}apriltag_{tag['id']}.png")
    return {k: v for k, v in out.items() if v}


def rpy_matrix(r, p, y) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def camera_nominal_poses(h: SceneHandles) -> dict:
    """The cameras' BELIEVED optical poses (what ROS uses): fixed camera in the world (= robot base) frame, wrist
    camera relative to its hand link."""
    cam_cfg = load_fixed_camera()
    out = {"camera": camera_mount_T(cam_cfg["mount"])}
    if "wrist_camera" in h.paths:
        out["wrist_camera"] = T_parent_optical(wrist_camera(load_robot()))
    return out


def set_camera_pose(h: SceneHandles, key: str, T_optical: np.ndarray) -> None:
    """Move a simulated camera (optical frame pose: fixed camera in the world, wrist camera in its hand link)."""
    if key == "camera":
        h.camera.set_world_pose(position=T_optical[:3, 3], orientation=xyzw_to_wxyz(matrix_to_quat(T_optical[:3, :3])),
                                camera_axes="ros")
        return
    prim = get_current_stage().GetPrimAtPath(h.paths[key])
    qw, qx, qy, qz = xyzw_to_wxyz(matrix_to_quat(T_optical[:3, :3] @ np.diag([1.0, -1.0, -1.0])))   # ROS -> USD axes
    prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*map(float, T_optical[:3, 3])))
    prim.GetAttribute("xformOp:orient").Set(Gf.Quatf(float(qw), float(qx), float(qy), float(qz)))


def _sample_triangles(tri: np.ndarray, spacing: float, rng, max_points: int = 2_000_000) -> np.ndarray:
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    n = np.maximum(1, np.round(area / spacing ** 2)).astype(int)
    if n.sum() > max_points:                       # a guard: never let one request stall the sim loop
        n = np.maximum(1, (n * max_points / n.sum()).astype(int))
    idx = np.repeat(np.arange(len(tri)), n)
    u, v = rng.random(len(idx)), rng.random(len(idx))
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    return np.vstack([a[idx] + u[:, None] * (b - a)[idx] + v[:, None] * (c - a)[idx], a, b, c])


def _unit_shape_triangles(prim) -> np.ndarray:
    """Triangles of a USD Cube / Cylinder in its own frame (procedural shapes: the sim's LEGO bricks)."""
    if prim.IsA(UsdGeom.Cube):
        h = float(UsdGeom.Cube(prim).GetSizeAttr().Get() or 2.0) / 2
        c = np.array([[x, y, z] for z in (-h, h) for y in (-h, h) for x in (-h, h)])
        f = [[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6], [0, 1, 4], [1, 5, 4], [2, 6, 3], [3, 6, 7],
             [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5]]
        return c[np.array(f)]
    if prim.IsA(UsdGeom.Cylinder):
        cy = UsdGeom.Cylinder(prim)
        r, hh = float(cy.GetRadiusAttr().Get()), float(cy.GetHeightAttr().Get()) / 2
        a = np.linspace(0, 2 * np.pi, 33)
        ring = np.column_stack([np.cos(a), np.sin(a)]) * r
        tris = []
        for k in range(32):
            p0, p1 = ring[k], ring[k + 1]
            lo0, lo1, hi0, hi1 = [*p0, -hh], [*p1, -hh], [*p0, hh], [*p1, hh]
            tris += [[lo0, lo1, hi0], [lo1, hi1, hi0], [[0, 0, hh], hi0, hi1], [[0, 0, -hh], lo1, lo0]]
        return np.array(tris, float)
    return np.zeros((0, 3, 3))


def visual_surface_points(path: str, spacing: float = 0.0004, rng=None) -> np.ndarray:
    """Points spread over the VISIBLE surface of everything under a prim, world frame: render meshes and USD shape
    prims, skipping collision geometry (what physics touches, not what you see) and hidden / guide prims."""
    rng = rng or np.random.default_rng(0)
    stage = get_current_stage()
    cache = UsdGeom.XformCache()
    out = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(path), Usd.TraverseInstanceProxies()):
        if "collision" in str(prim.GetPath()).lower() or not prim.IsA(UsdGeom.Gprim):
            continue
        img = UsdGeom.Imageable(prim)
        if img.ComputeVisibility() == UsdGeom.Tokens.invisible or img.ComputePurpose() == UsdGeom.Tokens.guide:
            continue
        if prim.IsA(UsdGeom.Mesh):
            m = UsdGeom.Mesh(prim)
            pts = np.array(m.GetPointsAttr().Get() or [], float)
            counts = np.array(m.GetFaceVertexCountsAttr().Get() or [], int)
            idx = np.array(m.GetFaceVertexIndicesAttr().Get() or [], int)
            if len(pts) == 0 or len(counts) == 0:
                continue
            tris, k = [], 0
            for n in counts:                                     # fan-triangulate the polygons
                tris += [[idx[k], idx[k + i], idx[k + i + 1]] for i in range(1, n - 1)]
                k += n
            tri = pts[np.array(tris)]
        else:
            tri = _unit_shape_triangles(prim)
            if len(tri) == 0:
                continue
        # to the world FIRST (the transform includes scale: a YCB asset is in cm with a 0.01 scale), THEN sample: the
        # point count comes from the triangles' areas, which must be in m^2 (sampling in cm froze the sim: 1e6 points
        # per cm^2 triangle)
        M = np.array(cache.GetLocalToWorldTransform(prim), float).T          # (Gf: row vectors -> transpose)
        out.append(_sample_triangles(tri @ M[:3, :3].T + M[:3, 3], spacing, rng))
    return np.vstack(out) if out else np.zeros((0, 3))


def prim_world_T(path: str) -> np.ndarray:
    """A prim's pose in the world (= robot base) frame, scale removed."""
    M = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(get_current_stage().GetPrimAtPath(path)), float).T
    T = np.eye(4)                                                                # (Gf: row vectors -> transpose)
    T[:3, :3] = M[:3, :3] / np.linalg.norm(M[:3, :3], axis=0)
    T[:3, 3] = M[:3, 3]
    return T


def true_camera_pose(h: SceneHandles, key: str, relative_to_parent: bool = False) -> np.ndarray:
    """Where a simulated camera REALLY is: optical frame in the world (= robot base) frame, or in its parent prim's
    frame (the wrist camera: in fr3_hand, what an eye-in-hand calibration finds)."""
    T = prim_world_T(h.paths[key])
    T[:3, :3] = T[:3, :3] @ np.diag([1.0, -1.0, -1.0])                           # USD -> ROS camera axes
    if relative_to_parent:
        T = np.linalg.inv(prim_world_T(str(get_current_stage().GetPrimAtPath(h.paths[key]).GetParent().GetPath()))) @ T
    return T


def _park_position(cfg: dict, index: int, size) -> np.ndarray:
    """Where unused obstacle boxes wait: on the floor far behind the robot (outside the camera's view)."""
    floor = cfg["table"]["center"][2] + cfg["table"]["size"][2] / 2 - cfg["table"]["height_above_floor"]
    return np.array([-3.0, -1.0 + 0.6 * index, floor + size[2] / 2 + 0.001])


def set_obstacles(h: SceneHandles, layout: list[dict]) -> None:
    """Put the boxes in `layout` (pnp_core.obstacles dicts) on the table; park every other pool box."""
    placed = {b["id"]: b for b in layout}
    for i, spec in enumerate(h.cfg["obstacles"]["pool"]):
        box = h.obstacles[spec["id"]]
        if spec["id"] in placed:
            b = placed[spec["id"]]
            half = b["yaw"] / 2
            pos = np.array(b["center"]) + np.array([0.0, 0.0, 0.001])       # 1 mm drop: no initial overlap
            box.set_world_pose(position=pos, orientation=np.array([np.cos(half), 0.0, 0.0, np.sin(half)]))
        else:
            box.set_world_pose(position=_park_position(h.cfg, i, spec["size"]), orientation=np.array([1.0, 0, 0, 0]))
        box.set_linear_velocity(np.zeros(3))
        box.set_angular_velocity(np.zeros(3))


def add_urdf_rig(rig: dict, table_top: float, root: str = "/World/camera_rig") -> None:
    """A camera stand from its CAD export (scene camera_rig: urdf, xy, yaw_deg, tilt_rad), visual only: every mesh
    visual of the URDF where its link is, the mount's base flat on the board at xy / yaw, the tilt joint at tilt_rad
    (from the export pose). The sim camera itself stays where the calibration puts it; tools/fit_camera_mount.py placed
    this stand around it (its residual: how far the CAD's lens is from the calibrated one)."""
    from pnp_core.urdf_visuals import axis_R, link_T, load_urdf_visuals, read_stl_triangles

    base, joints, visuals = load_urdf_visuals(REPO_ROOT / rig["urdf"])
    q = {name: float(rig.get("tilt_rad", 0.0)) for (_p, _T, _a, typ, name) in joints.values()
         if typ in ("revolute", "continuous")}
    T_base_mount = make_transform(axis_R([0, 0, 1], np.radians(float(rig.get("yaw_deg", 0.0)))),
                                  [float(rig["xy"][0]), float(rig["xy"][1]), table_top])
    stage = get_current_stage()
    UsdGeom.Xform.Define(stage, root)
    allpts = []
    for link, items in visuals.items():
        T_link = T_base_mount @ link_T(link, joints, q)
        for i, (stl, T_vis, rgba) in enumerate(items):
            tri = read_stl_triangles(stl)
            T = T_link @ T_vis
            pts = tri.reshape(-1, 3) @ T[:3, :3].T + T[:3, 3]
            mesh = UsdGeom.Mesh.Define(stage, f"{root}/{link}_{i}")
            mesh.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in pts])
            mesh.CreateFaceVertexCountsAttr([3] * len(tri))
            mesh.CreateFaceVertexIndicesAttr(list(range(len(pts))))
            mesh.CreateDoubleSidedAttr(True)
            if rgba:
                mesh.CreateDisplayColorAttr([Gf.Vec3f(*map(float, rgba[:3]))])
            allpts.append(pts)
    allpts = np.vstack(allpts) if allpts else np.zeros((1, 3))
    print(f"[scene] camera stand {rig['urdf']}: {sum(len(v) for v in visuals.values())} meshes, base frame bbox "
          f"{np.round(allpts.min(0), 3).tolist()} .. {np.round(allpts.max(0), 3).tolist()} m", flush=True)


def add_camera_rig(cam_cfg: dict, R_opt: np.ndarray, floor_z: float) -> None:
    """Visual-only D435i body + floor stand at the camera mount, so you can SEE where the camera is.
    (Isaac ships only a D455 model; this box has the D435i's real size: 90 x 25 x 25 mm.)
    No collision, and everything sits behind the camera's 5 cm near clip, so it never appears in the images."""
    from pnp_core.geometry import optical_to_body

    pos = np.array(cam_cfg["mount"]["position"], float)
    R_link = optical_to_body(R_opt)                   # camera_link: x = viewing direction, z = up
    q_link = xyzw_to_wxyz(matrix_to_quat(R_link))
    rig = "/World/camera_rig"
    grey, dark, black = np.array([0.55, 0.57, 0.6]), np.array([0.08, 0.08, 0.1]), np.array([0.02, 0.02, 0.02])

    # body: 25 mm deep (x, behind the lens), 90 mm wide (y), 25 mm tall (z), in camera_link coordinates
    body_center = pos + R_link @ np.array([-0.0135, 0.0, 0.0])
    VisualCuboid(f"{rig}/body", name="d435i_body", position=body_center, orientation=q_link,
                 scale=np.array([0.025, 0.090, 0.025]), size=1.0, color=grey)
    # front glass strip + the two IR stereo lenses (50 mm baseline) and RGB lens
    VisualCuboid(f"{rig}/front", name="d435i_front", position=pos + R_link @ np.array([-0.0005, 0.0, 0.0]),
                 orientation=q_link, scale=np.array([0.001, 0.080, 0.018]), size=1.0, color=black)
    for name, y in (("ir_left", 0.025), ("ir_right", -0.025), ("rgb", -0.0125)):
        VisualCuboid(f"{rig}/{name}", name=f"d435i_{name}", position=pos + R_link @ np.array([0.0002, y, 0.0]),
                     orientation=q_link, scale=np.array([0.001, 0.008, 0.008]), size=1.0, color=dark)
    boom = cam_cfg["mount"].get("boom")
    if boom is None:
        # stand: vertical pole from the floor up to just under the camera, plus a small mount block
        pole_top = pos[2] - 0.03
        VisualCylinder(f"{rig}/pole", name="camera_pole",
                       position=np.array([pos[0], pos[1], (floor_z + pole_top) / 2]),
                       radius=0.012, height=pole_top - floor_z, color=dark)
        VisualCuboid(f"{rig}/mount", name="camera_mount", position=np.array([pos[0], pos[1], pole_top + 0.008]),
                     scale=np.array([0.03, 0.03, 0.016]), size=1.0, color=dark)
        return
    # a camera looking DOWN would see its own pole: a pole beside the table (at mount.boom xy) + a horizontal arm
    # ABOVE the camera body out to it (a desk-lamp / boom-arm mount)
    base_xy = np.array(boom, float)
    top = pos[2] + 0.03
    VisualCylinder(f"{rig}/pole", name="camera_pole", position=np.array([base_xy[0], base_xy[1], (floor_z + top) / 2]),
                   radius=0.012, height=top - floor_z, color=dark)
    d = pos[:2] - base_xy
    length = float(np.linalg.norm(d))
    yaw = float(np.arctan2(d[1], d[0]))
    VisualCuboid(f"{rig}/boom", name="camera_boom",
                 position=np.array([(pos[0] + base_xy[0]) / 2, (pos[1] + base_xy[1]) / 2, top]),
                 orientation=xyzw_to_wxyz(matrix_to_quat(rpy_matrix(0.0, 0.0, yaw))),
                 scale=np.array([length, 0.02, 0.02]), size=1.0, color=dark)
    VisualCuboid(f"{rig}/mount", name="camera_mount", position=np.array([pos[0], pos[1], (top + pos[2] + 0.0125) / 2]),
                 scale=np.array([0.015, 0.015, top - pos[2] - 0.0125]), size=1.0, color=dark)


def place_target(h: SceneHandles, xy, yaw: float = 0.0, z=None):
    """Teleport the target object (standing upright) to xy/yaw on the table, or to an explicit z."""
    objects = load_yaml("config/objects.yaml")
    obj = objects[h.cfg["target"]["object"]]
    table_top = h.cfg["table"]["center"][2] + h.cfg["table"]["size"][2] / 2
    z = table_top + obj["dims_m"][2] / 2 + 0.001 if z is None else z
    h.target.set_world_pose(position=np.array([xy[0], xy[1], z]),
                            orientation=upright_quat_wxyz(obj.get("usd_up_axis", "Z"), yaw))
    h.target.set_linear_velocity(np.zeros(3))
    h.target.set_angular_velocity(np.zeros(3))


def place_extra(h: SceneHandles, name: str, xy, yaw: float = 0.0, z=None):
    """Teleport one of the scene's extra_objects (standing upright) to xy/yaw on the table."""
    spec = next(e for e in h.cfg.get("extra_objects", []) if e["name"] == name)
    obj = load_yaml("config/objects.yaml")[spec["object"]]
    table_top = h.cfg["table"]["center"][2] + h.cfg["table"]["size"][2] / 2
    z = table_top + obj["dims_m"][2] / 2 + 0.001 if z is None else z
    rb = h.extras[name]
    rb.set_world_pose(position=np.array([xy[0], xy[1], z]), orientation=upright_quat_wxyz(obj.get("usd_up_axis", "Z"), yaw))
    rb.set_linear_velocity(np.zeros(3))
    rb.set_angular_velocity(np.zeros(3))


def reset_scene(h: SceneHandles, target_xy=None, target_yaw: float = 0.0):
    """Reset physics, put the robot at home and the object at target_xy (default from config)."""
    h.world.reset()
    robot_cfg = load_robot(robot_name(h.cfg))
    if h.robot is not None:
        names = h.robot.dof_names
        q = np.zeros(len(names))
        home = dict(zip(robot_cfg["arm"]["joints"], robot_cfg["arm"]["home"]))
        g = robot_cfg["gripper"]
        open_q = GripperModel(g).command(g["max_width_m"])
        for i, n in enumerate(names):
            q[i] = home.get(n, open_q if n in g["joints"] else 0.0)   # gripper open (a FR3 finger mimic: same value)
        h.robot.set_joint_positions(q)
        h.robot.set_joint_velocities(np.zeros_like(q))
        # Joint *state* and joint drive *targets* are separate: also set the targets, otherwise the
        # drives pull the arm toward their old targets (all zeros) as soon as physics runs.
        h.robot.apply_action(ArticulationAction(joint_positions=q))
    if target_xy is not None:
        place_target(h, target_xy, target_yaw)
        for e in h.cfg.get("extra_objects", []):
            place_extra(h, e["name"], e["default_xy"], float(e.get("yaw", 0.0)))
