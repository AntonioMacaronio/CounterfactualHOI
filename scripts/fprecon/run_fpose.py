"""FoundationPose object trajectory on hand-scaled Depth Pro 2 depth (step 3 of the FoundationPose reconstruction).

  python scripts/fprecon/run_fpose.py --video-dir $CHOIR_ROOT/output/VID --out $PROJECT_STORAGE_ROOT/runs/fprecon/VID
  (fpose env; needs OUT/depth_dp2.npy and OUT/hand_scale.json from run_depth.py / hand_scale.py)

Masks: SAM3 per-frame masks (sam3_masks.py) when present, else CHOIR's SAM2 tracks (see masks.py).
Depth: Depth Pro 2 x the MANO-hand scale, with the (dilated) hand pixels outside the object mask set to 0 = invalid,
so the refiner does not fit the object to the fingers. Mesh: the textured SAM3D mesh from sam3d_mesh.sh (its
layout.json scale and center depth, converted to meters with the metric depth under the frame-0 object mask), else
CHOIR's untextured glb (scale converted via CHOIR's frame-0 point map; then vertex-colored from frame 0). Either way
the size is refined by registering at frame 0 with a range of sizes (coarse grid, then +-5% around the best) and
keeping the one whose render best matches the object mask and the depth. The mesh is then tracked frame by frame
(FoundationPose track_one). Each frame is checked occlusion-aware (see verify_track.py):
coverage = share of the visible object mask under the render, spill = share of the render on neither object nor
hand. After 3 frames with coverage < 0.75 or spill > 0.25 (object at least partly visible), the object is
re-registered from that frame's mask, picking the best-fitting of FoundationPose's top 10 hypotheses by that
check; if that fits clearly better, it is kept, earlier frames are re-tracked
backwards from it (keeping whichever pose fits better, until 15 frames in a row are not), and forward tracking
resumes from it.
This catches e.g. a mug tracked with its handle on the wrong side after an in-hand rotation.
Writes OUT/object_mesh.obj (textured) or .ply (vertex-colored), metric, object frame, OUT/object_poses.npy ((N, 4, 4) object-to-camera,
OpenCV) and OUT/fpose_qc.json (size search, per-frame mask IoU / coverage / spill / depth residual, re-registered
frames).
"""

import argparse
import glob
import json
import logging
import os
import sys
import time

import cv2
import numpy as np
import torch
import trimesh

FP_DIR = os.environ.get("FP_DIR", "/data/sky/projects/CounterfactualHOI/build/FoundationPose")
sys.path.insert(0, FP_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "choir"))
from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor, dr  # noqa: E402
from Utils import make_mesh_tensors, nvdiffrast_render, set_seed  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from masks import hand_mask, mask_source, object_mask, textured_mesh  # noqa: E402
from pick_hand_side import read_mask  # noqa: E402


def load_frame(vd, run, depth, scale, i, hand_side):
    rgb = cv2.cvtColor(cv2.imread(os.path.join(vd, "inputs", "frames", "rgb", f"{i}.png")), cv2.COLOR_BGR2RGB)
    d = depth[i].astype(np.float32) * scale
    om = object_mask(vd, run, i)
    om = np.zeros(d.shape, bool) if om is None else om
    hm = hand_mask(vd, run, i, hand_side)
    if hm is None or hm.mean() >= 0.2:
        hm = np.zeros(d.shape, bool)
    # The object mask wins: a hand mask can swallow a held object (SAM2's took all of the mug in prp_4).
    hm = (cv2.dilate(hm.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0) & ~om
    d[hm] = 0
    return rgb, d, om, hm


def fit(glctx, mesh_tensors, K, pose, mask, depth, hand=None):
    """Mask IoU, coverage, spill and median |depth residual| (m) of the posed mesh against the observation."""
    H, W = depth.shape
    _, rd, _ = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=torch.as_tensor(pose[None], device="cuda", dtype=torch.float),
                                 glctx=glctx, mesh_tensors=mesh_tensors)
    rd = rd[0].cpu().numpy()
    sil = rd > 0
    hand = np.zeros_like(sil) if hand is None else hand
    both = sil & (depth > 0)
    return {"iou": float((sil & mask).sum() / max((sil | mask).sum(), 1)),
            "coverage": float((sil & mask).sum() / mask.sum()) if mask.sum() > 200 else float("nan"),
            "spill": float((sil & ~mask & ~hand).sum() / max(sil.sum(), 1)),
            "depth_res_m": float(np.median(np.abs(rd[both] - depth[both]))) if both.sum() > 20 else float("nan"),
            "visible": bool(mask.sum() > 500 and mask.sum() >= 0.3 * max(sil.sum(), 1))}, rd


def quality(q):
    return (1.0 if np.isnan(q["coverage"]) else q["coverage"]) - q["spill"]


def is_bad(q):
    return q["visible"] and ((not np.isnan(q["coverage"]) and q["coverage"] < 0.75) or q["spill"] > 0.25)


def sam3d_metric_scale(vd, depth0):
    """SAM3D object scale converted to meters: its frame-0 point map vs the metric depth on the background."""
    M = np.array(json.load(open(os.path.join(vd, "stage1", "object_init", "transform_0.json")))["transform"], float)
    s_sam3d = float(np.mean(np.linalg.norm(M[:3, :3], axis=1)))
    pc = np.asarray(trimesh.load(os.path.join(vd, "stage1", "depth", "env_depths", "pc_0.ply")).vertices)
    z = pc[:, 2].reshape(depth0.shape)
    bg = np.ones(depth0.shape, bool)
    for m in glob.glob(os.path.join(vd, "inputs", "masks", "*", "0.png")):
        m = read_mask(m)
        if m.mean() < 0.2:
            bg &= ~(cv2.dilate(m.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)
    ok = bg & (depth0 > 0.05) & np.isfinite(z) & (z > 0.05)
    k = float(np.median(depth0[ok] / z[ok]))
    return s_sam3d * k, s_sam3d, k


def textured_metric_scale(run, depth0, mask0):
    """SAM3D (textured mesh) scale in meters: its layout.json local->scene scale x (metric depth under the frame-0
    object mask / SAM3D's center depth of the object)."""
    o = json.load(open(os.path.join(run, "sam3d_tex", "masks", "layout.json")))["objects"][0]["local_to_scene"]
    s_sam3d, z_sam3d = float(np.mean(o["scale"])), float(o["translation"][2])
    k = float(np.median(depth0[mask0 & (depth0 > 0.05)])) / z_sam3d
    return s_sam3d * k, s_sam3d, k


def color_from_frame(mesh, pose, K, rgb, rd):
    """Vertex colors from the image for vertices visible in the rendered depth; the rest get the median color."""
    H, W = rd.shape
    v = mesh.vertices @ pose[:3, :3].T + pose[:3, 3]
    p = (K @ v.T).T
    uv = np.round(p[:, :2] / p[:, 2:3]).astype(int)
    inb = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    vis = np.zeros(len(v), bool)
    vis[inb] = np.abs(rd[uv[inb, 1], uv[inb, 0]] - v[inb, 2]) < 0.01
    cols = np.tile(np.median(rgb[uv[vis, 1], uv[vis, 0]], 0) if vis.any() else np.array([128, 128, 128]), (len(v), 1))
    cols[vis] = rgb[uv[vis, 1], uv[vis, 0]]
    m = mesh.copy()
    m.visual = trimesh.visual.ColorVisuals(m, vertex_colors=np.c_[cols, np.full(len(v), 255)].astype(np.uint8))
    return m, float(vis.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--est-refine-iter", type=int, default=5)
    ap.add_argument("--track-refine-iter", type=int, default=2)
    ap.add_argument("--size-factors", default="0.6,0.7,0.8,0.9,1.0,1.1,1.25,1.4")
    args = ap.parse_args()
    logging.disable(logging.INFO)            # FoundationPose logs every step at INFO
    set_seed(0)
    vd, out = args.video_dir, args.out
    hs = json.load(open(os.path.join(out, "hand_scale.json")))
    depth = np.load(os.path.join(out, "depth_dp2.npy"), mmap_mode="r")
    N = depth.shape[0]
    K = np.array(json.load(open(os.path.join(vd, "inputs", "intrinsics.json")))["intrinsics"], float)
    rgb0, d0, m0, h0 = load_frame(vd, out, depth, hs["scale"], 0, hs["side"])

    tex = textured_mesh(out)
    if tex:
        base, s_sam3d, k = textured_metric_scale(out, depth[0].astype(np.float32) * hs["scale"], m0)
        unit = trimesh.load(tex, force="mesh", process=False)
    else:
        base, s_sam3d, k = sam3d_metric_scale(vd, depth[0].astype(np.float32) * hs["scale"])
        unit = trimesh.load(os.path.join(vd, "stage1", "object_init", "glb_0.glb"), force="mesh", process=False)
    unit.vertices -= unit.bounding_box.centroid
    glctx = dr.RasterizeCudaContext()
    scorer, refiner = ScorePredictor(), PoseRefinePredictor()
    est = None
    search = []
    t0 = time.time()
    def try_size(f):
        nonlocal est
        mesh = unit.copy()
        mesh.vertices *= base * f
        if est is None:
            est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer,
                                 refiner=refiner, glctx=glctx, debug=0, debug_dir=os.path.join(out, "fpose_debug"))
        else:
            est.reset_object(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh)
        pose = est.register(K=K, rgb=rgb0, depth=d0, ob_mask=m0, iteration=args.est_refine_iter)
        q, _ = fit(glctx, est.mesh_tensors, K, pose, m0, d0)
        iou, res = q["iou"], q["depth_res_m"]
        search.append({"factor": f, "size_m": float(np.ptp(mesh.vertices, 0).max()), "iou": iou, "depth_res_m": res,
                       "score": iou - 10 * res})          # 1 cm of depth residual costs 0.1 IoU
        print(f"size x{f:.2f} ({search[-1]['size_m']:.3f} m): IoU {iou:.3f}, depth residual {1000 * res:.1f} mm")

    for f in [float(x) for x in args.size_factors.split(",")]:      # coarse grid, then +-5% around the best
        try_size(f)
    coarse = max(search, key=lambda s: s["score"])["factor"]
    for f in (coarse * 0.95, coarse * 1.05):
        try_size(f)
    best = max(search, key=lambda s: s["score"])
    mesh = unit.copy()
    mesh.vertices *= base * best["factor"]
    est.reset_object(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh)
    pose = est.register(K=K, rgb=rgb0, depth=d0, ob_mask=m0, iteration=args.est_refine_iter)
    vis_frac = None
    if not tex:                                   # CHOIR's glb has no colors: paint them from frame 0
        _, rd = fit(glctx, est.mesh_tensors, K, pose, m0, d0)
        mesh, vis_frac = color_from_frame(mesh, pose, K, rgb0, rd)
        est.reset_object(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh)
    mt = make_mesh_tensors(mesh)
    pose = est.register(K=K, rgb=rgb0, depth=d0, ob_mask=m0, iteration=args.est_refine_iter)

    def set_last(p):                               # continue tracking from pose p (original-mesh frame)
        est.pose_last = torch.as_tensor(p @ np.linalg.inv(est.get_tf_to_centered_mesh().cpu().numpy()),
                                        device="cuda", dtype=torch.float)

    poses = np.zeros((N, 4, 4))
    per_frame, rereg, attempts, bad_since, last_try = [], [], [], None, -10
    for i in range(N):
        rgb, d, m, hm = (rgb0, d0, m0, h0) if i == 0 else load_frame(vd, out, depth, hs["scale"], i, hs["side"])
        if i > 0:
            pose = est.track_one(rgb=rgb, depth=d, K=K, iteration=args.track_refine_iter)
        q, _ = fit(glctx, mt, K, pose, m, d, hm)
        bad_since = (i if bad_since is None else bad_since) if is_bad(q) else None
        if i > 0 and bad_since is not None and i - bad_since >= 2 and i - last_try >= 5:
            last_try = i
            est.register(K=K, rgb=rgb, depth=d, ob_mask=m, iteration=args.est_refine_iter)
            # FoundationPose's top-scored hypothesis can be the same wrong one (e.g. a mug's hidden handle on the
            # wrong side); let the mask check choose among its best 10 instead.
            tf = est.get_tf_to_centered_mesh()
            cands = [(est.poses[k] @ tf).cpu().numpy() for k in range(min(10, len(est.poses)))]
            scored = [(c, fit(glctx, mt, K, c, m, d, hm)[0]) for c in cands]
            p2, q2 = max(scored, key=lambda cq: quality(cq[1]))
            attempts.append({"frame": i, "before": quality(q), "best": quality(q2),
                             "candidates": [round(quality(cq[1]), 3) for cq in scored]})
            if quality(q2) > quality(q) + 0.05:
                pose, q = p2, q2
                rereg.append(i)
                # Re-track backwards from the new pose (the drift often starts well before it is detected, e.g.
                # while the hand hides a mug's handle), keeping better poses, until 15 frames in a row are not better.
                misses = 0
                for j in range(i - 1, (rereg[-2] if len(rereg) > 1 else 1) - 1, -1):
                    rj, dj, mj, hj = load_frame(vd, out, depth, hs["scale"], j, hs["side"])
                    pj = est.track_one(rgb=rj, depth=dj, K=K, iteration=args.track_refine_iter)
                    qj, _ = fit(glctx, mt, K, pj, mj, dj, hj)
                    if quality(qj) > quality(per_frame[j]) + 0.01:
                        poses[j], per_frame[j], misses = pj, dict(qj, retracked=True), 0
                    else:
                        misses += 1
                        if misses >= 15:
                            break
                bad_since = None
            set_last(pose)
        poses[i] = pose
        per_frame.append(q)

    np.save(os.path.join(out, "object_poses.npy"), poses)
    for stale in ("object_mesh.obj", "object_mesh.ply"):
        if os.path.exists(os.path.join(out, stale)):
            os.remove(os.path.join(out, stale))
    mesh.export(os.path.join(out, "object_mesh.obj" if tex else "object_mesh.ply"))
    vis = [p for p in per_frame if p["visible"]]
    qc = {"frames": N, "mask_source": mask_source(out), "mesh_source": "sam3d_textured" if tex else "choir_glb",
          "hand_scale": hs["scale"], "sam3d_scale": s_sam3d, "sam3d_to_metric": k, "base_size_scale": base,
          "size_search": search, "chosen_factor": best["factor"], "object_size_m": np.ptp(mesh.vertices, 0).tolist(),
          "frame0_visible_vertex_frac": vis_frac, "reregistered_frames": rereg, "reregister_attempts": attempts,
          "iou_visible_frames_mean": float(np.mean([p["iou"] for p in vis])) if vis else None,
          "coverage_median": float(np.nanmedian([p["coverage"] for p in vis])) if vis else None,
          "spill_median": float(np.median([p["spill"] for p in vis])) if vis else None,
          "flagged_frac": float(np.mean([is_bad(p) for p in per_frame])),
          "iou_first_last": [per_frame[0]["iou"], per_frame[-1]["iou"]], "seconds": time.time() - t0,
          "per_frame": per_frame}
    json.dump(qc, open(os.path.join(out, "fpose_qc.json"), "w"), indent=1)
    print(f"{os.path.basename(vd)}: size x{best['factor']:.3f} -> {np.ptp(mesh.vertices, 0).round(3).tolist()} m; "
          f"IoU first/last {per_frame[0]['iou']:.2f}/{per_frame[-1]['iou']:.2f}; coverage {qc['coverage_median']:.2f}, "
          f"spill {qc['spill_median']:.2f}, flagged {100 * qc['flagged_frac']:.0f}% of frames; "
          f"re-registered at {rereg}; {qc['seconds']:.0f}s")

if __name__ == "__main__":
    main()
