"""Thinned copies of a model's visual meshes for the web viewer (view_embodiment.py), within a set error.

  python scripts/embodiment/viewer_meshes.py --arm vega_1p        (venv: $PROJECT_STORAGE_ROOT/venvs/arm, pymeshlab)
  -> $PROJECT_STORAGE_ROOT/build/embodiment/<arm>_sharpa_r/viewer_meshes/<mesh name>.npz  (vertices in the mesh
     file's frame, faces)
     and report.json (faces before / after and the measured error of every mesh)

The vendor meshes are CAD exports with up to 60k triangles per part; a viewer file carries every mesh, so they are
thinned. Each mesh is decimated with MeshLab's quadric edge collapse (pymeshlab; boundaries and normals preserved),
trying 6% to 60% of its triangles, and the smallest version is kept whose surface stays within the tolerance of the
original: the 99th percentile of the distance between points sampled on one surface and the other, both ways, and
the largest such distance within three times that. Tolerance: 1 mm for what moves (the arm, the hand, the mount),
2 mm for the robot's frozen body. A mesh that cannot be thinned within it is kept as it is.

The result is then split at sharp edges (dihedral angle over 30°), so that the browser's smooth shading keeps the
CAD creases instead of smearing shadows across them. The meshes of FlashDexRetarget's floating hand (the viewer's
ghost) have the same names as the robot's hand meshes and share the files.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp

import mujoco
import numpy as np
import trimesh

import common as C

FRACTIONS = (0.06, 0.09, 0.13, 0.19, 0.27, 0.4, 0.6)
TOL_MOVING, TOL_FIXED = 0.001, 0.002
CREASE_DEG = 30.0


def surface_error(a, b, n=2000, seed=0):
    """Distances [m] from n points sampled on mesh a to mesh b."""
    P, _ = trimesh.sample.sample_surface(a, n, seed=seed)
    _, d, _ = trimesh.proximity.closest_point(b, P)
    return d


def decimate(v, f, n):
    import pymeshlab
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(np.asarray(v, np.float64), np.asarray(f, np.int32)))
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int(n), preserveboundary=True, boundaryweight=2.0,
                                                preservenormal=True, preservetopology=False, optimalplacement=True,
                                                planarquadric=True, qualitythr=0.3, autoclean=True)
    mm = ms.current_mesh()
    return mm.vertex_matrix(), mm.face_matrix()


def thin(v, f, tol):
    """-> (vertices, faces, p99 error [m], max error [m], kept fraction of the faces)."""
    tm = trimesh.Trimesh(v, f, process=False)
    if len(f) <= 600:
        return v, f, 0.0, 0.0, 1.0
    for frac in FRACTIONS:
        try:
            dv, df = decimate(v, f, max(300, len(f) * frac))
        except Exception:
            continue
        if len(df) >= 0.9 * len(f) or len(df) == 0:
            break
        dec = trimesh.Trimesh(dv, df, process=False)
        e1 = surface_error(tm, dec, 2000, 0)
        if np.percentile(e1, 99) > tol or e1.max() > 3 * tol:
            continue
        e2 = surface_error(dec, tm, 1200, 1)
        if np.percentile(e2, 99) > tol or e2.max() > 3 * tol:
            continue
        return dv, df, float(max(np.percentile(e1, 99), np.percentile(e2, 99))), float(max(e1.max(), e2.max())), len(df) / len(f)
    return v, f, 0.0, 0.0, 1.0


def crease_split(v, f):
    tm = trimesh.Trimesh(v, f, process=True)
    try:
        sm = trimesh.graph.smooth_shade(tm, angle=np.radians(CREASE_DEG))
        return np.asarray(sm.vertices, np.float32), np.asarray(sm.faces, np.int32)
    except Exception:                                            # flat shading of the whole mesh is still correct
        return np.asarray(tm.vertices, np.float32), np.asarray(tm.faces, np.int32)


def _job(args):
    name, v, f, tol, path = args
    dv, df, p99, mx, frac = thin(v, f, tol)
    sv, sf = crease_split(dv, df)
    np.savez_compressed(path, vertices=sv, faces=sf)
    return name, {"faces": int(len(f)), "faces_thinned": int(len(df)), "kept": round(frac, 3), "tolerance_mm": tol * 1000,
                  "error_p99_mm": round(p99 * 1000, 3), "error_max_mm": round(mx * 1000, 3), "vertices_after_crease_split": int(len(sv))}


def meshes_of(model_path, moving_from_joint=None):
    """(mesh name, vertices, faces, moves) for every visible mesh geom of a model."""
    m = mujoco.MjModel.from_xml_path(str(model_path))
    b1 = m.jnt_bodyid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, moving_from_joint)] if moving_from_joint else 0

    def moves(b):
        if not moving_from_joint:
            return True
        while b > 0:
            if b == b1:
                return True
            b = m.body_parentid[b]
        return False
    out, seen = [], set()
    for g in range(m.ngeom):
        gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if (m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_group[g] not in (1, 2) or gname.startswith("collision_")
                or m.geom_rgba[g, 3] <= 0):
            continue
        i = m.geom_dataid[g]
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, i)
        if name in seen:
            continue
        seen.add(name)
        v = m.mesh_vert[m.mesh_vertadr[i]: m.mesh_vertadr[i] + m.mesh_vertnum[i]].astype(np.float64)
        f = m.mesh_face[m.mesh_faceadr[i]: m.mesh_faceadr[i] + m.mesh_facenum[i]].astype(np.int64)
        # MuJoCo re-centres a mesh on its inertia frame when it compiles; store the mesh file's own frame instead
        # (file = R v + p), so that any model compiled from the same file can use it (to_model_frame)
        v = v @ C.quat2mat(m.mesh_quat[i]).T + m.mesh_pos[i]
        out.append((name, v, f, moves(m.geom_bodyid[g])))
    return out


def to_model_frame(m, mesh_id, v_file):
    """Vertices in a mesh file's frame -> the frame of that mesh in compiled model m (the frame of geom_pos / quat)."""
    return (np.asarray(v_file, np.float64) - m.mesh_pos[mesh_id]) @ C.quat2mat(m.mesh_quat[mesh_id])


def export(arm, workers=8):
    md = C.model_dir(arm)
    out = md / "viewer_meshes"
    out.mkdir(exist_ok=True)
    jobs = []
    for name, v, f, moves in meshes_of(md / "scene.xml", "arm_j1"):
        jobs.append((name, v, f, TOL_MOVING if moves else TOL_FIXED, out / f"{name}.npz"))
    have = {j[0] for j in jobs}
    for name, v, f, _ in meshes_of(C.FDR / "assets" / "robot" / "sharpa" / "right.xml"):
        if name not in have:                                     # the floating hand's own meshes, if any differ
            jobs.append((name, v, f, TOL_MOVING, out / f"{name}.npz"))
    with mp.get_context("fork").Pool(workers) as pool:
        rows = dict(pool.map(_job, jobs, chunksize=1))
    n0 = sum(r["faces"] for r in rows.values())
    n1 = sum(r["faces_thinned"] for r in rows.values())
    json.dump({"arm": arm, "meshes": rows, "faces": n0, "faces_thinned": n1}, open(out / "report.json", "w"), indent=1)
    return rows, n0, n1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    rows, n0, n1 = export(args.arm, args.workers)
    worst = max(rows.items(), key=lambda kv: kv[1]["error_p99_mm"])
    kept = [k for k, r in rows.items() if r["kept"] >= 1.0 and r["faces"] > 600]
    print(f"[viewer meshes] {args.arm}: {len(rows)} meshes, {n0} -> {n1} triangles ({100 * n1 / n0:.0f}%); largest p99 error "
          f"{worst[1]['error_p99_mm']:.2f} mm ({worst[0]}); kept unthinned: {kept or 'none'}")


if __name__ == "__main__":
    main()
