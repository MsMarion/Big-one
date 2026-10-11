"""Joint offset calibration: find each joint's zero error from the arm accuracy check's views (kinematic calibration).

    python3 /repo/tools/fit_joint_offsets.py data/calibration/arm_accuracy_<time>.json            (report only)
    python3 /repo/tools/fit_joint_offsets.py <file> --write          -> config/calibration/<cell>/joint_offsets_<source>.yaml
        --reference truth | fixed_camera | configured_spot | free     where the board REALLY is (default: the best known)
        --with-hand-eye   also solve the wrist camera's mount (a correction of T_hand_cam) in the SAME fit: the joint
                          offsets and the wrist calibration depend on each other (each was measured through the other);
                          against an independent reference -- the tag-anchored fixed camera (calibrate.py locate on
                          the plate's AprilTags: no arm involved) -- both come out at once. --write then also writes
                          the corrected wrist calibration

What it solves. Each view k of the board, through the arm:
    T_base_board_k = FK(q_k + offset) * T_hand_cam * T_cam_board_k
    q_k        the joint angles the encoders REPORTED            T_hand_cam  the wrist camera calibration
    offset     what we want: true joint angle - encoder angle     T_cam_board the camera's view of the board
The board didn't move, so every view should give the same pose: the reference (where the board really is: the sim's
truth, the fixed camera's view, or the spot it was laid at) or, with `free`, a pose fitted along. A least-squares fit
finds the 5 offsets that make all views agree with it. A joint error moves the views in its own pattern (a shoulder
error tips the whole arm, a wrist error only the hand), so several different arm poses separate them; a small
prior (+-3 deg) keeps directions the views can't see near 0 (with `free`: the base rotation, which only turns the
whole picture). The fit uses the NOMINAL arm (the URDF as shipped): the result is the total offset, not a
correction of an earlier fit.

Gravity flex (links bending more when stretched out) is not a constant offset: the fit absorbs its average, the rest
stays in the residual. After --write, restart the stack (bringup loads it for that hardware) and run the accuracy
check again: the spread and the error against the reference should drop. docs/learning/17_so101.md "Joint offsets".
"""

import argparse
import datetime
import json
import sys

import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, "/repo")
from pnp_core import robot_description as rdesc  # noqa: E402
from pnp_core.calibration import board_center_T  # noqa: E402
from pnp_core.catalog import calibration_dir, calibration_targets_path, load_robot, load_yaml, repo_root  # noqa: E402
from pnp_core.urdf_kin import Chain  # noqa: E402

ROT_W = 0.1            # rotation residual weight: 1 rad counts like 0.1 m (1 deg ~ 1.7 mm)
PRIOR_DEG = 3.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("samples")
    ap.add_argument("--reference", choices=["truth", "fixed_camera", "configured_spot", "free"], default=None)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--with-hand-eye", action="store_true")
    a = ap.parse_args()
    path = a.samples if a.samples.startswith("/") else str(repo_root() / a.samples)
    rep = json.loads(open(path).read())
    cfg = load_robot(rep["robot"])
    joints = cfg["arm"]["joints"]
    hand = Chain(rdesc.urdf(cfg, repo_root(), "mock", offsets={}), cfg["frames"]["base"], rep["hand_link"])
    X = np.array(rep["T_hand_cam"])
    targets = load_yaml(calibration_targets_path())
    tb = dict(targets["table_board"])
    if rep.get("board", "table_board") != "table_board":         # another pattern on the desk (--board hand_board)
        tb.update({k: targets[rep["board"]][k] for k in ("dictionary", "squares_x", "squares_y", "square_m",
                                                          "marker_m", "first_id")})
    C = board_center_T(tb)
    S = rep["samples"]
    Q = np.array([[s["q_reported"][j] for j in hand.names] for s in S])
    Vb = [np.array(s["T_cam_board"]) @ C for s in S]                   # the board center, in the camera
    refs = rep.get("references", {})
    ref = a.reference or next((r for r in ("truth", "fixed_camera", "configured_spot") if r in refs), "free")
    if ref != "free" and ref not in refs:
        print(f"no '{ref}' reference in {path} (have: {list(refs)})")
        return 2
    B0 = np.array(refs[ref]) if ref != "free" else None
    idx = [hand.names.index(j) for j in joints]
    nj = len(joints)
    he = bool(a.with_hand_eye)
    if he and B0 is None:
        print("--with-hand-eye needs a reference (truth / fixed_camera / configured_spot): with `free` the board and "
              "the camera mount trade off")
        return 2

    def dX(p):                                       # the wrist mount correction: X = X0 @ dX (rotvec, mm)
        if not he:
            return np.eye(4)
        D = np.eye(4)
        D[:3, :3] = Rotation.from_rotvec(p[nj:nj + 3]).as_matrix()
        D[:3, 3] = p[nj + 3:nj + 6] / 1000.0
        return D

    def board_poses(off, D=np.eye(4)):
        out = []
        for q, V in zip(Q, Vb):
            qq = q.copy()
            qq[idx] += off
            out.append(hand.fk(qq) @ X @ D @ V)
        return out

    def residuals(p):
        off = p[:nj]
        if B0 is None:                                                  # free: the board pose is fitted too
            B = np.eye(4)
            B[:3, :3] = Rotation.from_rotvec(p[nj:nj + 3]).as_matrix()
            B[:3, 3] = p[nj + 3:nj + 6]
        else:
            B = B0
        r = []
        for T in board_poses(off, dX(p)):
            r += list((T[:3, 3] - B[:3, 3]) * 1000)                       # mm
            r += list(Rotation.from_matrix(B[:3, :3].T @ T[:3, :3]).as_rotvec() * ROT_W * 1000)
        r += list(np.degrees(off) / PRIOR_DEG)                          # prior: offsets near 0
        if he:                                                          # prior: the CAD mount is within ~3 deg / 5 mm
            r += list(np.degrees(p[nj:nj + 3]) / 3.0) + list(p[nj + 3:nj + 6] / 5.0)
        return np.array(r)

    p0 = np.zeros(nj + (6 if he else 0))
    if B0 is None:
        T_mean = board_poses(np.zeros(nj))[0]
        p0 = np.concatenate([p0, Rotation.from_matrix(T_mean[:3, :3]).as_rotvec(), T_mean[:3, 3]])
    before = residuals(p0)
    sol = least_squares(residuals, p0, method="lm")
    after = sol.fun
    npos = len(S) * 6

    def pos_rms(r):
        e = r[:npos].reshape(-1, 6)[:, :3]
        return float(np.sqrt(np.mean(np.sum(e ** 2, axis=1))))
    off_deg = np.degrees(sol.x[:nj])
    J = sol.jac[:, :nj]
    try:
        cov = np.linalg.inv(J.T @ J) * (np.sum(after ** 2) / max(1, len(after) - len(sol.x)))
        std = np.degrees(np.sqrt(np.diag(cov)))
    except np.linalg.LinAlgError:
        std = np.full(nj, np.nan)
    print(f"{len(S)} views, reference: {ref}")
    print(f"board position error over the views: rms {pos_rms(before):.2f} mm before -> {pos_rms(after):.2f} mm after")
    truth = (rep.get("servo_error") or {}).get("joints", {})
    print("joint            offset [deg]    +-   " + ("  sim truth (offset)" if truth else ""))
    for j, o, sd in zip(joints, off_deg, std):
        t = f"   {truth[j]['offset_deg']:+.2f}" if j in truth else ""
        print(f"  {j:16s} {o:+8.3f}    {sd:6.3f}  {t}")
    X_new = X @ dX(sol.x)
    if "truth" in refs:                    # sim: does the WHOLE corrected chain put the board where it really is?
        from pnp_core.calibration import mean_transform, pose_error
        Tt = np.array(refs["truth"])
        e0 = pose_error(mean_transform(board_poses(np.zeros(nj))), Tt)
        e1 = pose_error(mean_transform(board_poses(sol.x[:nj], dX(sol.x))), Tt)
        print(f"the arm + wrist chain vs the TRUE board: {e0[0]:.2f} mm / {e0[1]:.3f} deg before -> "
              f"{e1[0]:.2f} mm / {e1[1]:.3f} deg after")
    if he:
        d = dX(sol.x)
        print(f"wrist mount correction: {np.linalg.norm(d[:3, 3]) * 1000:.2f} mm, "
              f"{np.degrees(np.linalg.norm(Rotation.from_matrix(d[:3, :3]).as_rotvec())):.3f} deg")
        tw = rep.get("wrist_truth")                    # sim: the TRUE mount (if the samples carry it)
        if tw is not None:
            from pnp_core.calibration import pose_error
            e0, e1 = pose_error(X, np.array(tw)), pose_error(X_new, np.array(tw))
            print(f"  vs the sim's TRUE wrist mount: {e0[0]:.2f} mm / {e0[1]:.3f} deg before -> "
                  f"{e1[0]:.2f} mm / {e1[1]:.3f} deg after")
    if not a.write:
        print("(report only: --write saves it)")
        return 0
    out = calibration_dir() / f"joint_offsets_{rep['source']}.yaml"           # the active scene's cell
    entry = {"offsets_deg": {j: round(float(o), 4) for j, o in zip(joints, off_deg)},
             "std_deg": {j: round(float(s_), 4) for j, s_ in zip(joints, std)},
             "date": datetime.datetime.now().isoformat(timespec="seconds"), "reference": ref, "views": len(S),
             "board_rms_mm": {"before": round(pos_rms(before), 3), "after": round(pos_rms(after), 3)},
             "samples": str(path).replace(str(repo_root()) + "/", "")}
    out.write_text("# Joint offset calibration (tools/fit_joint_offsets.py): true joint angle - encoder angle, for the\n"
                   "# hardware this file is named after. Bringup folds it into the robot description (restart the stack).\n"
                   + yaml.safe_dump(entry, sort_keys=False))
    print(f"written: {out.relative_to(repo_root())}  -> restart the stack to use it")
    if he:
        write_wrist(rep, X_new, ref, len(S))
    return 0


def write_wrist(rep, X_new, ref, n):
    """The corrected wrist calibration, written like calibrate.py writes one (history kept) to the active set."""
    from pnp_core.geometry import matrix_to_quat
    from pnp_core.wrist import wrist_camera
    w = wrist_camera(load_robot(rep["robot"]))
    cal_set = rep.get("calibration_set") or "sim_realistic"
    path = calibration_dir() / f"{cal_set}_{w['name']}.yaml"
    old = yaml.safe_load(path.read_text()) if path.exists() else {}
    hist = old.pop("history", []) if old else []
    if old:
        hist = [{k: old[k] for k in ("date", "T_parent_optical", "method") if k in old}] + hist
    entry = {"camera": w["name"], "parent": rep["hand_link"], "optical_frame": w["frames"]["color_optical"],
             "T_parent_optical": {"xyz": [round(float(v), 6) for v in X_new[:3, 3]],
                                  "quat_xyzw": [round(float(v), 7) for v in matrix_to_quat(X_new[:3, :3])]},
             "date": datetime.datetime.now().isoformat(timespec="seconds"),
             "method": f"joint fit with the arm offsets (tools/fit_joint_offsets.py --with-hand-eye, reference {ref}, "
                       f"{n} views)", "history": hist[:10]}
    path.write_text("# Wrist camera calibration (tools/fit_joint_offsets.py --with-hand-eye). camera_tf_node publishes\n"
                    f"# T_parent_optical as {entry['parent']} -> {entry['optical_frame']}.\n"
                    + yaml.safe_dump(entry, sort_keys=False, default_flow_style=None, width=120))
    print(f"written: {path.relative_to(repo_root())}  (the wrist calibration; /calibration/reload applies it)")


if __name__ == "__main__":
    sys.exit(main())
