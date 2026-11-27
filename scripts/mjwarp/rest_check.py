"""Drop test of every clip's reconstructed rest poses on MuJoCo Warp, under the nominal physics and under each of the
task's stored trials (scripts/mjwarp/trials.py): does the object stay where the reconstruction put it?

  python scripts/mjwarp/rest_check.py DATASET [--seconds 0.5]           (venvs/fdr-mjwarp; one GPU)

The MuJoCo Warp twin of scripts/flashdex/rest_check.py (CPU MuJoCo, one pose at a time, nominal physics only). Per
clip one model (the object's convex parts on a table: a plane at the clip's support-disk top) with one world per
(pose, physics): 2 poses (first and last reference frame) x (1 nominal + N trials). The trials' mass, inertia, centre of
mass, friction (sliding / torsional / rolling) and contact damping ratio (restitution) are written into the worlds'
model rows. Reports how far the object moved and turned in each world. Writes DATASET/rest_check_mjwarp.json.

--fix-trials: a trial whose object turns more than --max-extra-deg (5) beyond the nominal drop at the first or the
last pose is not a valid task (the reference has it at rest). Its centre-of-mass offset, the parameter that tips
marginal reconstructed rest poses (offsets of 0.8-1.4 cm did on the case, the mug and the bottle), is drawn again
within half the previous bound, up to 4 rounds, and trials.json is rewritten with "valid" per trial and "validated"
per task (all its trials valid). scripts/mjwarp/randomize.py draws only valid trials.
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch
import trimesh
import warp as wp

import mujoco_warp as mjw
from mjlab.sim.randomization import expand_model_fields

FIELDS = ["body_mass", "body_inertia", "body_ipos", "geom_friction", "geom_solref", "geom_condim"]


def build(parts, z_top, density=800.0):
    spec = mujoco.MjSpec()
    spec.option.timestep = 1 / 480
    spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    t = spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.01], pos=[0, 0, z_top])
    t.friction = [0.1, 0.0, 0.0]
    body = spec.worldbody.add_body(name="obj")
    body.add_freejoint()
    for k, (v, f) in enumerate(parts):
        spec.add_mesh(name=f"m{k}", uservert=v.reshape(-1).tolist(), userface=f.reshape(-1).tolist())
        g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"m{k}", density=density)
        g.friction, g.condim, g.solmix = [1.0, 0.005, 0.0001], 6, 1000.0
    return spec.compile()


def quat_angle_deg(q0, q1):
    d = np.abs(np.sum(q0 * q1, -1)).clip(max=1.0)
    return np.degrees(2 * np.arccos(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--seconds", type=float, default=0.5)
    ap.add_argument("--fix-trials", action="store_true")
    ap.add_argument("--max-extra-deg", type=float, default=5.0)
    a = ap.parse_args()
    D = Path(a.dataset)
    pack = D / "motion_bimanual.pt" if (D / "motion_bimanual.pt").exists() else D / "motion.pt"
    pt = torch.load(pack, map_location="cpu", weights_only=False)
    tfile = json.load(open(D / "trials.json")) if (D / "trials.json").exists() else {"tasks": {}}
    trials = tfile["tasks"]
    out = {}
    rng = np.random.default_rng(int(tfile.get("seed", 0)) + 1)
    for mi, name in enumerate(pt["motion_filename"]):
        for rnd in range(4 if a.fix_trials else 1):
            res = drop_clip(pt, D, trials, mi, name, a.seconds)
            if not a.fix_trials or name not in trials:
                break
            bad = [k for k in range(len(trials[name]["trials"])) if any(
                res[tag]["trials"][k]["turned_deg"] > res[tag]["nominal"]["turned_deg"] + a.max_extra_deg for tag in ("start", "end"))]
            for k, tr_k in enumerate(trials[name]["trials"]):
                tr_k["valid"] = k not in bad
            if not bad:
                trials[name]["validated"] = True
                break
            lim = np.minimum(0.10 * np.array(trials[name]["extent_m"]), 0.015) * 0.5 ** (rnd + 1)
            for k in bad:
                trials[name]["trials"][k]["com_offset_m"] = [float(x) for x in rng.uniform(-lim, lim)]
            print(f"   {name}: trials {bad} tip the object at rest; centre of mass drawn again within "
                  f"{[round(100 * x, 2) for x in lim]} cm", flush=True)
        else:
            trials[name]["validated"] = False
        out[name] = res
        tw = [x["turned_deg"] for x in res["start"]["trials"]], [x["turned_deg"] for x in res["end"]["trials"]]
        print(f"{name}: turned at start {res['start']['nominal']['turned_deg']:.1f} deg (trials {min(tw[0], default=0):.1f}-"
              f"{max(tw[0], default=0):.1f}), at end {res['end']['nominal']['turned_deg']:.1f} deg (trials "
              f"{min(tw[1], default=0):.1f}-{max(tw[1], default=0):.1f})", flush=True)
    json.dump(out, open(D / "rest_check_mjwarp.json", "w"), indent=1)
    if a.fix_trials:
        json.dump(tfile, open(D / "trials.json", "w"), indent=1)
        print(f"rewrote {D / 'trials.json'}: {sum(t.get('validated', False) for t in trials.values())}/{len(trials)} tasks validated")
    print(f"wrote {D / 'rest_check_mjwarp.json'}")


def drop_clip(pt, D, trials, mi, name, seconds):
    """Drop test of one clip's first and last pose under the nominal physics and each stored trial."""
    a0, n = int(pt["length_starts"][mi]), int(pt["motion_num_frames"][mi])
    slot = int(pt["motion_object_slot_right"][mi])
    parts = []
    for f in sorted(glob.glob(str(D / str(pt["right_object_mesh_dirs"][slot]) / "convex" / "*.obj"))):
        mesh = trimesh.load(f, force="mesh", process=False)
        parts.append((np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int32)))
    m = build(parts, float(pt["support_disks"][mi][0][2]))
    tr = trials.get(name, {}).get("trials", [])
    phys = [None] + tr
    W = 2 * len(phys)
    d = mujoco.MjData(m)
    mw = mjw.put_model(m)
    dw = mjw.put_data(m, d, nworld=W, nconmax=64 * W, njmax=512)
    if os.environ.get("RC_DEBUG"):
        free, total = torch.cuda.mem_get_info()
        print(f"   {name}: {m.ngeom} geoms, {W} worlds; GPU free {free / 2**30:.1f} GiB", flush=True)
    expand_model_fields(mw, W, FIELDS)
    tt = {f: wp.to_torch(getattr(mw, f)) for f in FIELDS}
    qpos = np.zeros((W, m.nq))
    for k, t in enumerate((a0, a0 + n - 1)):
        p = pt["obj_right_pos"][t].numpy().astype(np.float64)
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, pt["obj_right_rotmat"][t].numpy().astype(np.float64).reshape(-1))
        for j, ph in enumerate(phys):
            w = k * len(phys) + j
            qpos[w, :3], qpos[w, 3:7] = p, q
            if ph is None:
                continue
            base = trials[name]["mass_kg_base"] or float(m.body_mass[1])
            mass = base * ph["mass_scale"]
            tt["body_mass"][w, 1] = mass
            tt["body_inertia"][w, 1] = torch.as_tensor(m.body_inertia[1] * mass / m.body_mass[1], dtype=tt["body_inertia"].dtype)
            tt["body_ipos"][w, 1] = torch.as_tensor(m.body_ipos[1] + np.array(ph["com_offset_m"]), dtype=tt["body_ipos"].dtype)
            tt["geom_friction"][w, 1:] = torch.as_tensor(ph["friction"], dtype=tt["geom_friction"].dtype)
            tt["geom_solref"][w, 1:] = torch.as_tensor(ph["solref"], dtype=tt["geom_solref"].dtype)
    wp.to_torch(dw.qpos)[:] = torch.as_tensor(qpos, dtype=wp.to_torch(dw.qpos).dtype)
    mjw.set_const(mw, dw)
    mjw.forward(mw, dw)
    q0 = wp.to_torch(dw.qpos).cpu().numpy().copy()
    for _ in range(int(seconds / m.opt.timestep)):
        mjw.step(mw, dw)
    q1 = wp.to_torch(dw.qpos).cpu().numpy()
    del tt, mw, dw                                # free this clip's worlds before the next model
    import gc
    gc.collect()
    wp.synchronize()
    moved = 100 * np.linalg.norm(q1[:, :3] - q0[:, :3], axis=1)
    turned = quat_angle_deg(q0[:, 3:7], q1[:, 3:7])
    res = {}
    for k, tag in enumerate(("start", "end")):
        sl = slice(k * len(phys), (k + 1) * len(phys))
        res[tag] = {"nominal": {"moved_cm": round(float(moved[sl][0]), 2), "turned_deg": round(float(turned[sl][0]), 1)},
                    "trials": [{"moved_cm": round(float(x), 2), "turned_deg": round(float(y), 1)}
                               for x, y in zip(moved[sl][1:], turned[sl][1:])]}
    return res


if __name__ == "__main__":
    sys.exit(main())
