"""Is each clip's reconstructed resting object pose physically stable? A MuJoCo drop test, no hand.

  python -I scripts/flashdex/rest_check.py DATASET [--seconds 0.5]       (flashdex env)

For every clip in DATASET/motion.pt: the object (its convex parts, FDR's objects/<id>/convex/, density 800 like the
env) starts at the reference's first-frame pose on the clip's table disk, and falls for --seconds. It reports how
far it moved and turned: a reconstructed rest pose that is tilted or inside the table settles somewhere else, which
the policy cannot undo (its object then starts off the reference). Also the same for the last frame (the placed
pose). Writes DATASET/rest_check.json.
"""

import argparse
import glob
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
import trimesh


def quat_angle_deg(q0, q1):
    d = abs(float(np.dot(q0, q1)))
    return float(np.degrees(2 * np.arccos(min(1.0, d))))


def drop(parts, pos, R, disk, seconds):
    spec = mujoco.MjSpec()
    spec.option.timestep = 1 / 480
    x, y, z_top, r, _ = disk
    tb = spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.5, 0.005, 0], pos=[x, y, z_top - 0.005])
    tb.friction = [1.0, 0.005, 0.0001]
    body = spec.worldbody.add_body(name="obj", pos=pos.tolist(), quat=mat2quat(R).tolist())
    body.add_freejoint()
    for k, (v, f) in enumerate(parts):
        spec.add_mesh(name=f"m{k}", uservert=v.reshape(-1).tolist(), userface=f.reshape(-1).tolist())
        g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"m{k}", density=800)
        g.friction = [1.0, 0.005, 0.0001]
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    q0, p0 = d.qpos[3:7].copy(), d.qpos[:3].copy()
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
    return float(100 * np.linalg.norm(d.qpos[:3] - p0)), quat_angle_deg(q0, d.qpos[3:7])


def mat2quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, np.float64).reshape(-1))
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--seconds", type=float, default=0.5)
    args = ap.parse_args()
    ds = Path(args.dataset)
    pt = torch.load(ds / "motion.pt", map_location="cpu", weights_only=False)
    names = [str(x).removeprefix("cfhoi_") for x in pt["motion_filename"]]
    out = {}
    for m, vid in enumerate(names):
        a0, n = int(pt["length_starts"][m]), int(pt["motion_num_frames"][m])
        d = ds / str(pt["right_object_mesh_dirs"][int(pt["motion_object_slot_right"][m])])
        parts = []
        for f in sorted(glob.glob(str(d / "convex" / "*.obj"))):
            mesh = trimesh.load(f, force="mesh", process=False)
            parts.append((np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int32)))
        disk = pt["support_disks"][m][0].numpy()
        res = {}
        for tag, t in (("start", a0), ("end", a0 + n - 1)):
            moved, turned = drop(parts, pt["obj_right_pos"][t].numpy().astype(np.float64),
                                 pt["obj_right_rotmat"][t].numpy().astype(np.float64), disk, args.seconds)
            res[tag] = {"moved_cm": round(moved, 2), "turned_deg": round(turned, 1)}
        out[vid] = res
        print(vid, res, flush=True)
    json.dump(out, open(ds / "rest_check.json", "w"), indent=1)


if __name__ == "__main__":
    main()
