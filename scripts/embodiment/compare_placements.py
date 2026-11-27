"""Compare stances of the Vega under the IK the data is made with (ik_pyroki.py, fingertips mode).

  python scripts/embodiment/compare_placements.py --arm vega_1p [--heights 1.2 1.3 1.4 1.5] [--back 0 0.1 0.2]
      [--stride 6] [--workers 12]          (venv: $PROJECT_STORAGE_ROOT/venvs/arm; CPU only)
  -> $PROJECT_STORAGE_ROOT/runs/embodiment/placement/<arm>/stances.json

A stance is the right shoulder's position: its height above the floor and how far it is moved back from the current
placement, along the robot's heading (away from the table). The chassis follows and the torso's three joints take the
most upright posture that gives that height and keeps the robot off the table (place_base.py's torso table); the
heading and the hand's clocking stay as configured. Each stance is solved on every clip of the dataset's reference
motions at 60 / stride Hz with the arm's speed limits scaled to that rate, and measured on the exact meshes:

  usable frames        as ik_pyroki.py (fingertips within 3 mm, arm and hardware above the table, ...)
  elbow                the elbow joint's median angle and 5th percentile (0 = straight; the limit is -176 deg)
  near a limit         share of frames with any arm joint within 5 deg of its limit
  arm to itself        smallest distance between the upper arm (links 1-3) and the forearm, wrist and hand
  arm to body          smallest distance between the arm from the elbow outwards (or the hand) and the frozen body
  palm moved           how far the solver had to move the palm off the reference (median of the clips' maxima)
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import common as C  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402


def stance_candidate(cfg, joints, tab, floor_z, shoulder, heading_deg):
    """A place_base candidate (root pose + torso angles) putting the right shoulder joint at `shoulder`: the most
    upright torso posture of that height; the caller checks the table."""
    Rz = C.rot_z(np.radians(heading_deg))
    sh = joints[cfg["arm_joints"][0]]["xyz"]
    A = np.asarray(shoulder) - Rz @ sh
    ok = tab[np.abs(tab[:, 4] - (A[2] - floor_z)) < 0.008]
    ok = ok[np.argsort(np.abs(ok[:, 5]))]
    out = []
    for a, b, c, dx, dz, pitch in ok[:60]:
        root = A - Rz @ np.array([dx, 0, dz])
        out.append({"root": {"pos": [float(root[0]), float(root[1]), floor_z], "rpy_deg": [0, 0, heading_deg]},
                    "fixed": {"torso_j1": float(a), "torso_j2": float(b), "torso_j3": float(c)}, "lean_deg": float(np.degrees(pitch))})
    return out


def link_points(m, d, names, n=900):
    import spheres as S
    out = {}
    for nm in names:
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm)
        if b < 0:
            continue
        P = S.body_points(m, d, b, groups=(1, 0) if nm.startswith("right_") else (1,))
        if len(P):
            out[b] = P[np.random.default_rng(0).choice(len(P), min(n, len(P)), replace=False)]
    return out


def world(d, pts):
    return np.concatenate([P @ d.xmat[b].reshape(3, 3).T + d.xpos[b] for b, P in pts.items()]) if pts else np.zeros((0, 3))


def evaluate_stance(job):
    key, arm, dataset, stride, options, out_dir = job
    import ik_pyroki as IP
    import spheres as S
    from ik import ArmIK
    from place_base import apply_candidate
    from scene import Scene
    cfg = C.load_arm(arm)
    ik0 = ArmIK(arm)
    joints = C.read_urdf(C.VENDOR / cfg["urdf"])[1]
    m, d = ik0.model, ik0.data
    chosen = None
    for o in options:                                            # the most upright posture that clears the table
        apply_candidate(ik0, o, joints)
        ik0.set(ik0.home)
        if ik0.scene.points_to_box(ik0.scene.world_points(ik0.static_geoms), *ik0.scene.table_box()) >= 0.02:
            chosen = o
            break
    if chosen is None:
        return key, {"valid": False, "reason": "no torso posture of that height clears the table"}
    # this stance's URDF for PyRoki (same links, so the collision spheres carry over)
    cd = Path(out_dir) / key
    cd.mkdir(parents=True, exist_ok=True)
    info = json.load(open(C.model_dir(arm) / "spheres.json"))
    tips = {f"tip_{f}": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"track_hand_right_{f}_tip") for f in C.FINGERS}
    vel = dict(zip(info["arm_joints"], np.radians(cfg["velocity_deg_s"])))
    mujoco.mj_kinematics(m, d)
    S.write_urdf(m, d, cd / "ik_full.urdf", set(info["arm_joints"] + info["hand_joints"]), tips, vel)
    shutil.copyfile(C.model_dir(arm) / "spheres.json", cd / "spheres.json")
    fps = 60.0 / stride
    solver = IP.PyrokiIK(arm, "fingertips", fps=fps, model_dir=cd, verbose=False)
    hand = C.FloatingHand()
    floating = Scene("floating")
    G = info["groups"]
    upper = [f"R_arm_l{k}" for k in (1, 2, 3)]
    lower = [f"R_arm_l{k}" for k in (5, 6, 7, 8)] + ["hand_mount", C.PALM_BODY] + [n for n in G["fingers"]]
    # arm to body: from the elbow outwards (links 4-8 and the hand); the shoulder links sit against the torso's
    # shoulder housing by construction
    moving = [f"R_arm_l{k}" for k in (4, 5, 6, 7, 8)] + ["hand_mount", C.PALM_BODY] + list(G["fingers"])
    body = [n for n in G["fixed"] if not any(s in n for s in ("wheel", "lidar", "base_cam"))]
    P_up, P_lo, P_mv, P_body = (link_points(m, d, x) for x in (upper, lower, moving, body))
    mujoco.mj_kinematics(m, d)
    tree_body = cKDTree(world(d, P_body))
    res, Qall, self_d, body_d = {}, [], [], []
    for c in C.load_pack(dataset):
        Qf = c["joint_pos"][::stride]
        Ts = hand.palm_traj(Qf)
        Qh = hand.fingers(Qf)
        tipsT = np.zeros((len(Qf), 5, 4, 4))
        fing0, obj0 = np.zeros(len(Qf)), np.zeros(len(Qf))
        parts = IP.ObjectParts(c)
        OP, OR = c["obj_pos"][::stride], c["obj_rot"][::stride]
        for t, q in enumerate(Qf):
            floating.set_floating(q)
            floating.fk()
            for k, s in enumerate(floating.tip_sites):
                tipsT[t, k] = C.make_T(floating.data.site_xmat[s].reshape(3, 3), floating.data.site_xpos[s])
            fing0[t] = floating.table_clearance(floating.finger_geoms)
            obj0[t] = parts.depth(floating.world_points(floating.palm_geoms + floating.finger_geoms, 60), OP[t], OR[t])
        Q_ref, _ = ik0.convert(Ts, fps=fps)                      # None: the solver starts from the home posture
        arm_q, hand_q = solver.solve_clip(Ts, Qh, tipsT, Q_ref, fing0)
        ev = IP.evaluate(ik0.scene, ik0, arm_q, hand_q, Ts, tipsT[:, :, :3, 3], Qh, (parts, OP, OR), fps=fps)
        res[c["name"]] = IP.summarise(ev, fing0, obj0)
        Qall.append(arm_q)
        for t in range(len(arm_q)):
            ik0.scene.set_arm(arm_q[t], hand_q[t])
            ik0.scene.fk()
            self_d.append(cKDTree(world(d, P_up)).query(world(d, P_lo), k=1)[0].min())
            body_d.append(tree_body.query(world(d, P_mv), k=1)[0].min())
    Q = np.concatenate(Qall) if Qall else np.zeros((0, 7))
    lo, hi = ik0.lo, ik0.hi
    near = ((Q - lo) < np.radians(5)) | ((hi - Q) < np.radians(5))
    done = [v for v in res.values() if "frames" in v]
    n = sum(v["frames"] for v in done)
    self_d, body_d = np.array(self_d), np.array(body_d)
    mujoco.mj_kinematics(m, d)
    b1 = m.jnt_bodyid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "arm_j1")]
    return key, {
        "valid": True, "stance": chosen, "shoulder_xyz": d.xpos[b1].round(3).tolist(),
        "clips_fully_usable": int(sum(v.get("usable_pct", 0) >= 100 for v in res.values())),
        "frames_usable_pct": float(sum(v["usable_pct"] * v["frames"] for v in done) / max(1, n)),
        "elbow_median_deg": float(np.degrees(np.median(Q[:, 3]))) if len(Q) else None,
        "elbow_p5_deg": float(np.degrees(np.percentile(Q[:, 3], 5))) if len(Q) else None,
        "near_limit_pct": float(near.any(1).mean() * 100) if len(Q) else None,
        "arm_to_itself_min_mm": float(self_d.min() * 1000) if len(self_d) else None,
        "arm_to_itself_p5_mm": float(np.percentile(self_d, 5) * 1000) if len(self_d) else None,
        "arm_to_body_min_mm": float(body_d.min() * 1000) if len(body_d) else None,
        "arm_to_body_p5_mm": float(np.percentile(body_d, 5) * 1000) if len(body_d) else None,
        "palm_moved_max_mm_median": float(np.median([v["palm_dev_mm"]["max"] for v in done])) if done else None,
        "clips": res}


def main():
    from place_base import floor_offset, torso_table
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="vega_1p")
    ap.add_argument("--dataset", default=str(C.STORAGE / "runs" / "flashdex" / "data_r10_flat"))
    ap.add_argument("--heights", type=float, nargs="*", default=[1.2, 1.3, 1.4, 1.5])
    ap.add_argument("--back", type=float, nargs="*", default=[0.0, 0.1, 0.2])
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--tag", default="stances", help="output: placement/<arm>/<tag>.json")
    ap.add_argument("--origin", type=float, nargs=2, default=None,
                    help="x y of the shoulder that 'back' is measured from (default: the configured shoulder)")
    args = ap.parse_args()
    cfg = C.load_arm(args.arm)
    joints = C.read_urdf(C.VENDOR / cfg["urdf"])[1]
    from ik import ArmIK
    ik0 = ArmIK(args.arm)
    mujoco.mj_kinematics(ik0.model, ik0.data)
    b1 = ik0.model.jnt_bodyid[mujoco.mj_name2id(ik0.model, mujoco.mjtObj.mjOBJ_JOINT, "arm_j1")]
    sh0 = ik0.data.xpos[b1].copy()                               # the configured right shoulder
    if args.origin:
        sh0[:2] = args.origin
    heading = float(cfg["placement"]["root"]["rpy_deg"][2])
    back = -np.array([np.cos(np.radians(heading)), np.sin(np.radians(heading)), 0.0])
    tab = torso_table(cfg, joints)
    fz = floor_offset(args.arm)
    out = C.RUNS / "placement" / args.arm
    jobs = []
    for h in args.heights:
        for b in args.back:
            target = sh0 + back * b
            target[2] = h
            key = f"h{h:.2f}_back{b:.2f}"
            jobs.append((key, args.arm, args.dataset, args.stride, stance_candidate(cfg, joints, tab, fz, target, heading),
                         str(out / "stances")))
    with mp.get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
        rows = dict(pool.map(evaluate_stance, jobs, chunksize=1))
    json.dump({"arm": args.arm, "dataset": args.dataset, "stride": args.stride, "configured_shoulder": sh0.round(3).tolist(),
               "heading_deg": heading, "stances": rows}, open(out / f"{args.tag}.json", "w"), indent=1)
    print(f"configured shoulder {sh0.round(3)}, heading {heading:.0f} deg; solved at {60 / args.stride:.0f} Hz")
    print(f"{'stance':16s} {'lean':>5s} | {'clips':>5s} {'frames':>7s} | {'elbow med/p5':>13s} | {'near lim':>8s} | {'arm-arm min/p5':>14s} | {'arm-body min/p5':>15s} | palm moved")
    for k, r in rows.items():
        if not r.get("valid"):
            print(f"{k:16s} {r.get('reason')}")
            continue
        print(f"{k:16s} {r['stance']['lean_deg']:5.0f} | {r['clips_fully_usable']:5d} {r['frames_usable_pct']:6.1f}% | "
              f"{r['elbow_median_deg']:6.0f}/{r['elbow_p5_deg']:5.0f}  | {r['near_limit_pct']:7.1f}% | "
              f"{r['arm_to_itself_min_mm']:6.0f}/{r['arm_to_itself_p5_mm']:5.0f} mm | {r['arm_to_body_min_mm']:6.0f}/{r['arm_to_body_p5_mm']:5.0f} mm  | "
              f"{r['palm_moved_max_mm_median']:.0f} mm")


if __name__ == "__main__":
    main()
