"""Flat-bottom collision meshes for objects whose reconstructed rest pose is not stable (rest_check.py).

  python -I scripts/flashdex/flat_base.py SRC DST --clips pp_anthony_real_1 ... [--band 0.01]     (flashdex env)

SAM3D hallucinates the object's unseen underside, so the mesh bottom is not flat where the real object sits flat on
the table: in simulation the object settles 15-23 deg off the reconstructed pose. DST is a copy of the dataset SRC
(human_demo, retargeted, objects) where, for each --clips object, the collision mesh gets a flat base: with the
object at the first frame of its reference (motion.pt), every vertex within --band above the table disk's top, or
below it, is projected onto the table plane, so the base is flush and the object rests at exactly that pose. CoACD
re-runs on that mesh (FDR's coacd settings) into objects/<id>/convex/; visual.obj (looks, mass properties, SDF)
is unchanged, and the mesh is kept as objects/<id>/collision_flat.obj. The clips' kinematic retargets are removed
from DST/retargeted so FDR's kinematic_retargeting.sh redoes only those (the rest are cached), and the objects' USD
is rebuilt from the newer convex parts.
"""

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import trimesh
from omegaconf import OmegaConf

FDR = Path(os.environ.get("FDR", "/data/sky/projects/CounterfactualHOI/build/FlashDexRetarget"))
sys.path.insert(0, str(FDR / "src"))


def flatten(obj_dir, pos, R, z_top, band):
    mesh = trimesh.load(obj_dir / "visual.obj", process=False, skip_materials=True, force="mesh")
    v = np.asarray(mesh.vertices, np.float64)
    w = v @ R.T + pos
    low = w[:, 2] < z_top + band
    w[low, 2] = z_top
    out = trimesh.Trimesh((w - pos) @ R, np.asarray(mesh.faces), process=False)
    out.export(obj_dir / "collision_flat.obj")
    return int(low.sum()), len(v), float(100 * (z_top - (v @ R.T + pos)[:, 2].min()))


def coacd_parts(obj_dir, kwargs):
    import coacd
    from retarget.object.obj_convex_decompose import _write
    t0 = time.time()
    mesh = trimesh.load(obj_dir / "collision_flat.obj", process=False, force="mesh")
    coacd.set_log_level("warn")
    parts = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), seed=1, **kwargs)
    _write(Path(obj_dir), [trimesh.Trimesh(vs, np.asarray(fs, dtype=int)) for vs, fs in parts])
    return obj_dir.name, len(parts), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--band", type=float, default=0.01)
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    for sub in ("human_demo", "retargeted", "objects"):
        if not (dst / sub).exists():
            shutil.copytree(src / sub, dst / sub, symlinks=True)
    pt = torch.load(src / "motion.pt", map_location="cpu", weights_only=False)
    names = [str(x).removeprefix("cfhoi_") for x in pt["motion_filename"]]
    cfg = OmegaConf.load(FDR / "config" / "retarget" / "convex" / "coacd.yaml")
    kwargs = {k: v for k, v in OmegaConf.to_container(cfg.coacd).items()}
    info, jobs = {}, []
    for vid in args.clips:
        m = names.index(vid)
        a0 = int(pt["length_starts"][m])
        oid = Path(str(pt["right_object_mesh_dirs"][int(pt["motion_object_slot_right"][m])])).name
        z_top = float(pt["support_disks"][m][0][2])
        n_low, n, depth = flatten(dst / "objects" / oid, pt["obj_right_pos"][a0].numpy().astype(np.float64),
                                  pt["obj_right_rotmat"][a0].numpy().astype(np.float64), z_top, args.band)
        info[vid] = {"object": oid, "table_top_z": round(z_top, 4), "flattened_vertices": n_low, "vertices": n,
                     "deepest_below_table_cm": round(depth, 2)}
        print(vid, info[vid], flush=True)
        jobs.append(dst / "objects" / oid)
        rt = dst / "retargeted" / f"cfhoi_{vid}.npz"
        if rt.exists():
            rt.unlink()
    with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
        for name, k, sec in ex.map(coacd_parts, jobs, [kwargs] * len(jobs)):
            info[name]["convex_parts"] = k
            print(f"[flat_base] {name}: {k} convex parts, {sec:.0f} s", flush=True)
    json.dump(info, open(dst / "flat_base.json", "w"), indent=1)


if __name__ == "__main__":
    main()
