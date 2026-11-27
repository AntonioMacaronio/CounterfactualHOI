"""Where does the arm stand? Search the arm's placement in the clip world so that it can follow every clip.

  python scripts/embodiment/place_base.py --arm marvin_m6ccs [--style pedestal shoulder] [--workers 32]
  python scripts/embodiment/place_base.py --arm vega_1p
      -> $PROJECT_STORAGE_ROOT/runs/embodiment/placement/<arm>/<style>.json, and the best one printed as the
         `placement:` (and, for the Vega, torso `fixed_joints:`) block for configs/embodiment/arms/<arm>.yaml

The clip world is FDR's: table at z = 0.8 m, the object's first-frame centre over the origin, +y the way the source
video's camera looked; the human's right arm reaches in from +x / -y. All ten clips share it, so one placement has
to serve all of them.

Styles (a candidate is the pose of the arm's root link plus, for the Vega, the frozen torso angles):
  pedestal  Marvin upright (z up) on the table or on a riser, anywhere on a grid; the heading does not matter (J1)
  shoulder  Marvin as on Tianji's dual-arm body: the base axis horizontal, pointing to the robot's right, at
            shoulder height; grid over shoulder position and the body's heading
  chassis   Vega: a grid over the right shoulder's position and the body's heading; the wheeled base stands on the
            floor and the three torso joints raise and lean the shoulders to get there

Each candidate is scored with the IK of ik.py on every clip at 10 Hz (best of several first-frame solutions):
clips fully followed, frames missed, frames where an arm link is within 1 cm of the table, the smallest margin to a
joint limit, and the fastest joint against its speed limit. Candidates whose fixed parts intersect the table, or
whose base stands within 15 cm of an object path, are dropped before any IK.
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp

import mujoco
import numpy as np

import common as C

_G = {}


def frozen_pose(j, value, snap=True):
    """Pose of a frozen joint's child link in its parent, at joint value `value`."""
    rpy = C.snap_rpy(j["rpy"]) if snap else j["rpy"]
    T = C.make_T(C.rpy_mat(rpy), j["xyz"])
    if j["type"] in ("revolute", "continuous"):
        return T @ C.make_T(C.axis_angle(j["axis"], value))
    return T @ C.make_T(None, j["axis"] / np.linalg.norm(j["axis"]) * value)


def apply_candidate(ik, cand, joints):
    """Move the compiled model's root link, and re-pose frozen joints that the candidate overrides."""
    m = ik.model
    root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ik.cfg["root_link"])
    T = C.cfg_T(cand["root"])
    m.body_pos[root] = T[:3, 3]
    m.body_quat[root] = C.mat2quat(T[:3, :3])
    for name, v in (cand.get("fixed") or {}).items():
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, joints[name]["child"])
        Tj = frozen_pose(joints[name], v)
        m.body_pos[b] = Tj[:3, 3]
        m.body_quat[b] = C.mat2quat(Tj[:3, :3])
    if "hand_yaw" in cand:                                       # the hand's clocking on the flange
        hm = dict(ik.cfg["hand_mount"], yaw_deg=cand["hand_yaw"])
        Th = C.cfg_T(ik.cfg["flange"]) @ C.make_T(C.rot_z(np.radians(hm["yaw_deg"])), [0, 0, hm.get("plate_mm", 0) / 1000])
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand_mount")
        m.body_pos[b] = Th[:3, 3]
        m.body_quat[b] = C.mat2quat(Th[:3, :3])


def _init(arm, dataset, stride):
    from ik import ArmIK
    ik = ArmIK(arm)
    hand = C.FloatingHand()
    clips = C.load_pack(dataset)
    _G.update(ik=ik, Ts=[hand.palm_traj(c["joint_pos"][::stride]) for c in clips], names=[c["name"] for c in clips],
              obj=np.concatenate([c["obj_pos"][::stride] for c in clips]), stride=stride,
              joints=C.read_urdf(C.VENDOR / ik.cfg["urdf"])[1])


def _eval(cand):
    ik, Ts = _G["ik"], _G["Ts"]
    options = cand.pop("options", None)
    if options is not None:                                      # several body postures give the same shoulder pose:
        chosen = None                                            # take the first that keeps clear of the table
        for o in options:
            apply_candidate(ik, dict(cand, **o), _G["joints"])
            ik.set(ik.home)
            if ik.scene.points_to_box(ik.scene.world_points(ik.static_geoms), *ik.scene.table_box()) >= cand.get("static_min", 0.0):
                chosen = o
                break
        if chosen is None:
            return dict(cand, valid=False, reason="no body posture reaches this shoulder pose clear of the table")
        cand = dict(cand, **chosen)
    apply_candidate(ik, cand, _G["joints"])
    ik.set(ik.home)
    res = dict(cand)
    # fixed parts against the table, and the base against the object paths
    static = ik.scene.points_to_box(ik.scene.world_points(ik.static_geoms), *ik.scene.table_box()) if ik.static_geoms else 1.0
    root = mujoco.mj_name2id(ik.model, mujoco.mjtObj.mjOBJ_BODY, ik.cfg["root_link"])
    b1 = ik.model.jnt_bodyid[mujoco.mj_name2id(ik.model, mujoco.mjtObj.mjOBJ_JOINT, "arm_j1")]
    base_xy = ik.data.xpos[b1][:2]
    res["static_table_mm"] = float(static * 1000)
    res["base_to_object_m"] = float(np.linalg.norm(_G["obj"][:, :2] - base_xy, axis=1).min())
    res["shoulder_xyz"] = ik.data.xpos[b1].round(3).tolist()
    if cand.get("needs_static_clear", True) and static < cand.get("static_min", 0.0):
        res.update(valid=False, reason="fixed parts intersect the table")
        return res
    if res["base_to_object_m"] < 0.15:
        res.update(valid=False, reason="base within 15 cm of an object path")
        return res
    fps = 60.0 / _G["stride"]
    solved, fails, hits, margin, speed, per = 0, 0, 0, np.inf, 0.0, {}
    for name, T in zip(_G["names"], Ts):
        q0, rows = ik.best_start(T, stride=1, n=48)
        if q0 is None:
            fails += len(T)
            per[name] = {"fail": len(T)}
            continue
        r = rows[0]
        Q, info = ik.track(T, q0, clearance=True)
        f = int(((info["pos_err"] > 1e-3) | (info["rot_err"] > np.radians(0.5))).sum())
        h = int((info["table_clearance"] < 0.0).sum())
        mg = float(min(info["margin_joint"].min(), info["margin_coupled"].min() / np.pi))
        sp = float((np.abs(np.diff(Q, axis=0)) * fps / ik.vmax).max()) if len(Q) > 1 else 0.0
        fails += f
        hits += h
        solved += int(f == 0)
        margin = min(margin, mg)
        speed = max(speed, sp)
        per[name] = {"fail": f, "table_hits": h, "margin": round(mg, 4), "speed": round(sp, 2)}
    res.update(valid=True, clips_solved=solved, frames_failed=fails, table_hits=hits,
               margin=float(margin) if np.isfinite(margin) else 0.0, speed=speed, per_clip=per)
    return res


def rank_key(r):
    if not r.get("valid"):
        return (1, 0, 0, 0, 0)
    return (0, -r["clips_solved"], r["frames_failed"], r["table_hits"], -round(r["margin"], 3))


# ---------------------------------------------------------------- candidate generators
def heading_to(xy, target=(0.15, 0.1)):
    return float(np.degrees(np.arctan2(target[1] - xy[1], target[0] - xy[0])))


def gen_pedestal(args):
    out = []
    for x, y, h, yaw in itertools.product(np.arange(0.2, 1.001, 0.1), np.arange(-0.9, 0.401, 0.1), (0.0, 0.15, 0.3, 0.45),
                                          args.hand_yaw):
        out.append({"style": "pedestal", "root": {"pos": [round(float(x), 3), round(float(y), 3), round(C.TABLE_Z + h, 3)],
                                                  "rpy_deg": [0, 0, round(heading_to((x, y)), 1)]},
                    "needs_static_clear": False, "hand_yaw": yaw})
    return out


def gen_shoulder(args):
    """Root frame: z = the body's right (horizontal), y = up, x = the body's forward. Heading psi turns the body
    about the vertical; psi = 90 faces +y like the video camera."""
    out = []
    for x, y, z, psi, yaw in itertools.product(np.arange(0.15, 0.751, 0.1), np.arange(-0.95, -0.249, 0.1),
                                               np.arange(1.0, 1.451, 0.15), (60.0, 90.0, 120.0), args.hand_yaw):
        f = np.array([np.cos(np.radians(psi)), np.sin(np.radians(psi)), 0.0])      # forward
        up = np.array([0, 0, 1.0])
        right = np.cross(f, up)
        R = np.stack([f, up, right], 1)
        out.append({"style": "shoulder", "root": {"pos": [round(float(x), 3), round(float(y), 3), round(float(z), 3)],
                                                  "quat": C.mat2quat(R).round(6).tolist()},
                    "needs_static_clear": False, "heading_deg": psi, "hand_yaw": yaw})
    return out


def torso_table(cfg, joints, step=7.5):
    """Every torso posture on a grid: (angles, arm_center's forward offset and height in the chassis frame, pitch)."""
    rows = []
    lim = {n: (joints[n]["lower"], joints[n]["upper"]) for n in ("torso_j1", "torso_j2", "torso_j3")}
    grids = [np.radians(np.arange(np.degrees(lo), np.degrees(hi) + 1e-6, step)) for lo, hi in lim.values()]
    fixed = dict(cfg.get("fixed_joints") or {})
    for a, b, c in itertools.product(*grids):
        fixed.update(torso_j1=a, torso_j2=b, torso_j3=c)
        T = C.urdf_fk(joints, fixed, "arm_center", cfg["root_link"])
        pitch = np.arctan2(T[0, 2], T[2, 2])                     # forward lean of the shoulder block
        rows.append((a, b, c, T[0, 3], T[2, 3], pitch))
    return np.array(rows)


def gen_chassis(args):
    """Vega: candidates are poses of the right shoulder (position, the body's heading). The chassis stands on the
    floor facing `heading`; the three torso joints raise and lean the shoulders. The arm's reach does not depend on
    the lean (its first joint turns about the same axis), so for each shoulder pose every torso posture of the right
    height is a body option; the most upright one that clears the table is used."""
    cfg = C.load_arm(args.arm)
    joints = C.read_urdf(C.VENDOR / cfg["urdf"])[1]
    tab = torso_table(cfg, joints)
    sh = joints[cfg["arm_joints"][0]]["xyz"]                     # shoulder joint in arm_center's frame
    out = []
    for x, y, z, psi, yaw in itertools.product((0.2, 0.3, 0.4), (-0.4, -0.3, -0.2), (1.2, 1.3, 1.4), (75.0, 90.0, 105.0), args.hand_yaw):
        Rz = C.rot_z(np.radians(psi))
        A = np.array([x, y, z]) - Rz @ sh                        # arm_center (sh is along the pitch axis: lean-free)
        ok = tab[np.abs(tab[:, 4] - (A[2] - args.floor_z)) < 0.008]
        ok = ok[np.argsort(np.abs(ok[:, 5]))][:60]
        options = []
        for a, b, c, dx, dz, pitch in ok:
            root = A - Rz @ np.array([dx, 0, dz])
            options.append({"root": {"pos": [round(float(root[0]), 4), round(float(root[1]), 4), args.floor_z], "rpy_deg": [0, 0, psi]},
                            "fixed": {"torso_j1": round(float(a), 4), "torso_j2": round(float(b), 4), "torso_j3": round(float(c), 4)},
                            "lean_deg": round(float(np.degrees(pitch)), 1)})
        if options:
            out.append({"style": "chassis", "target_shoulder": [x, y, z], "heading_deg": psi, "hand_yaw": yaw, "static_min": 0.02,
                        "options": options, "root": options[0]["root"]})
    return out


GENERATORS = {"pedestal": gen_pedestal, "shoulder": gen_shoulder, "chassis": gen_chassis}


def floor_offset(arm):
    """Height of the root link's origin above the floor: minus the lowest mesh vertex of the model at rest."""
    from scene import Scene
    sc = Scene(arm)
    m, d = sc.model, sc.data
    root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, C.load_arm(arm)["root_link"])
    m.body_pos[root] = 0
    m.body_quat[root] = [1, 0, 0, 0]
    mujoco.mj_kinematics(m, d)
    lo = np.inf
    for g in sc.arm_geoms:
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        i = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[i]: m.mesh_vertadr[i] + m.mesh_vertnum[i]]
        lo = min(lo, float((v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].min()))
    return round(-lo, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--dataset", default=str(C.STORAGE / "runs" / "flashdex" / "data_r10_flat"))
    ap.add_argument("--style", nargs="*", default=None)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--hand-yaw", type=float, nargs="*", default=None,
                    help="hand clockings on the flange to try [deg]; default: the arm config's hand_mount.yaw_deg")
    args = ap.parse_args()
    cfg = C.load_arm(args.arm)
    args.hand_yaw = args.hand_yaw or [cfg["hand_mount"]["yaw_deg"]]
    styles = args.style or ([cfg["placement"]["style"]] if cfg["placement"]["style"] != "pedestal" else ["pedestal", "shoulder"])
    args.floor_z = floor_offset(args.arm) if "chassis" in styles else 0.0
    out = C.RUNS / "placement" / args.arm
    out.mkdir(parents=True, exist_ok=True)
    with mp.get_context("fork").Pool(args.workers, initializer=_init, initargs=(args.arm, args.dataset, args.stride)) as pool:
        for style in styles:
            cands = GENERATORS[style](args)
            res = pool.map(_eval, cands, chunksize=2)
            res.sort(key=rank_key)
            json.dump({"arm": args.arm, "style": style, "dataset": args.dataset, "stride": args.stride,
                       "floor_z": args.floor_z, "candidates": res}, open(out / f"{style}.json", "w"), indent=1)
            valid = [r for r in res if r.get("valid")]
            full = [r for r in valid if r["clips_solved"] == len(_names(args.dataset)) and r["table_hits"] == 0]
            print(f"[{style}] {len(cands)} candidates, {len(valid)} valid, {len(full)} follow all clips without touching the table")
            for r in res[:5]:
                if not r.get("valid"):
                    continue
                extra = f" heading {r['heading_deg']}" if "heading_deg" in r else ""
                extra += f" lean {r['lean_deg']}" if "lean_deg" in r else ""
                extra += f" hand yaw {r['hand_yaw']:.0f}"
                print(f"   root {r['root']['pos']}{extra} | shoulder {r['shoulder_xyz']} | clips {r['clips_solved']} | failed frames "
                      f"{r['frames_failed']} | table hits {r['table_hits']} | margin {r['margin']:.3f} | speed {r['speed']:.2f}x")
            if res and res[0].get("valid"):
                b = res[0]
                print("   best -> placement:", json.dumps({"style": style, "root": b["root"]}), "fixed:", json.dumps(b.get("fixed", {})),
                      "hand yaw:", b["hand_yaw"])
            by_yaw = {}
            for r in valid:
                by_yaw.setdefault(r["hand_yaw"], []).append(r)
            for yw, rs in sorted(by_yaw.items()):
                b = rs[0]
                nfull = sum(1 for r in rs if r["clips_solved"] == len(b["per_clip"]) and r["table_hits"] == 0)
                print(f"   hand yaw {yw:5.0f}: best clips {b['clips_solved']}, failed frames {b['frames_failed']}, table hits "
                      f"{b['table_hits']}, margin {b['margin']:.3f}; {nfull} candidates follow all clips clear of the table")


def _names(dataset):
    return [c["name"] for c in C.load_pack(dataset)]


if __name__ == "__main__":
    main()
