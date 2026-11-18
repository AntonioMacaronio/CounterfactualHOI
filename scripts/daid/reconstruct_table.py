"""Reconstruct the support table for a Do-as-I-Do run, in the same metric camera frame as hand + object.

  python scripts/daid/reconstruct_table.py --video-dir RUN/ --object-id red_apple   (sam3d env)

Do-as-I-Do reconstructs the hand (HaWoR, metric) and the object, but not the table. We fit it from the
MoGe pointmaps the pipeline already computed:
  1. rescale pointmaps to metric with the same per-frame factor DAID uses for the object
     (k = HaWoR hand depth / pointmap hand depth, saved per frame by optimize_translation_scale.py),
     dropping pixels under the (dilated) hand and object masks
  2. on the ref frame only (MoGe's scale drifts per frame; k is exact where the hand anchors it), RANSAC
     the support plane from points within --support-radius of the resting object, with the normal within
     --max-tilt-deg of the GeoCalib gravity axis (rejects walls and larger background planes)
  3. orient the normal toward the object (up), refine by SVD, and take the extent from coplanar points
     (ref frame + a few other frames, each with its own k) within --extent-radius of the object

Writes to RUN/table/: table.json (plane, frame basis, extent, QC), table_mesh.obj (a box whose top face is
the table surface), table_points.ply (colored inliers, for viewing).
"""

import argparse
import json
import os

import cv2
import numpy as np
import trimesh


def load_mask(video_dir, fidx, name, shape):
    p = os.path.join(video_dir, "video_segmentation", "masks", f"frame_{fidx:06d}_masks", f"{name}.png")
    if not os.path.exists(p):
        return np.zeros(shape, bool)
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE) > 127
    if m.shape != shape:
        m = cv2.resize(m.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    return m


def ransac_plane(pts, axis, max_tilt_deg, thresh, iters, rng):
    cos_min = np.cos(np.radians(max_tilt_deg))
    best_inl, best = None, None
    for _ in range(iters):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        if abs(n @ axis) < cos_min:
            continue
        inl = np.abs((pts - a) @ n) < thresh
        if best_inl is None or inl.sum() > best_inl.sum():
            best_inl, best = inl, (n, a)
    return best, best_inl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--object-id", required=True)
    ap.add_argument("--anchor-hand", default="right")
    ap.add_argument("--layout-json", default=None, help="default: <object> layout_camera_frame_optimized.json")
    ap.add_argument("--num-frames", type=int, default=8, help="extra frames sampled for the table extent")
    ap.add_argument("--support-radius", type=float, default=0.25,
                    help="fit the plane on points within this distance (m) of the object at the ref frame")
    ap.add_argument("--extent-radius", type=float, default=0.8,
                    help="table extent uses coplanar points within this in-plane distance (m) of the object")
    ap.add_argument("--max-tilt-deg", type=float, default=20.0)
    ap.add_argument("--inlier-thresh", type=float, default=0.01, help="metres")
    ap.add_argument("--thickness", type=float, default=0.03, help="box thickness below the surface, metres")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    vd = args.video_dir
    lay_path = args.layout_json or os.path.join(
        vd, "obj_tracking_out", args.object_id, "combined_visualization", "layout_camera_frame_optimized.json")
    opt = json.load(open(lay_path))["translation_scale_optimization"]
    k_by_frame = {f["frame_idx"]: f["pointmap_scale"] for f in opt["per_frame"]}
    if not k_by_frame:
        raise SystemExit("no per-frame pointmap scales in " + lay_path)
    k_med = float(np.median(list(k_by_frame.values())))
    gravity = np.array(json.load(open(os.path.join(vd, "gravity.json")))["vec3d"], float)
    gravity /= np.linalg.norm(gravity)

    frames = sorted(k_by_frame)
    per_frame = {f["frame_idx"]: f for f in opt["per_frame"]}
    ref = opt["ref_frame"] if opt["ref_frame"] in per_frame else frames[0]
    obj_ref = np.array(per_frame[ref]["obj_target"])  # object centre at the ref frame (resting on the table)
    rng = np.random.default_rng(args.seed)

    def frame_points(f):
        """Metric camera-frame points + colours of frame f, excluding (dilated) hand and object pixels."""
        pm = np.load(os.path.join(vd, "all_frames", f"{f:06d}_pointmap.npy"))
        h, w = pm.shape[:2]
        keep = np.isfinite(pm).all(-1) & (pm[..., 2] > 0)
        occl = load_mask(vd, f, args.object_id, (h, w)) | load_mask(vd, f, f"{args.anchor_hand}_hand_0", (h, w))
        keep &= ~(cv2.dilate(occl.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)
        img = cv2.cvtColor(cv2.imread(os.path.join(vd, "all_frames", f"{f:06d}.png")), cv2.COLOR_BGR2RGB)
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        idx = rng.choice(np.flatnonzero(keep), min(int(keep.sum()), 60000), replace=False)
        return pm.reshape(-1, 3)[idx] * k_by_frame[f], img.reshape(-1, 3)[idx]

    # Fit on the ref frame only: MoGe's scale drifts per frame, and k is exact only where the hand anchors it.
    # Restrict to points near the object, so we get the surface it rests on (not a larger background plane).
    pts_ref, rgb_ref = frame_points(ref)
    near = np.linalg.norm(pts_ref - obj_ref, axis=1) < args.support_radius
    if near.sum() < 200:
        raise SystemExit(f"only {near.sum()} points within {args.support_radius} m of the object at frame {ref}")
    (n, _), inl_near = ransac_plane(pts_ref[near], gravity, args.max_tilt_deg, args.inlier_thresh, 2000, rng)
    c = pts_ref[near][inl_near].mean(0)
    n = np.linalg.svd(pts_ref[near][inl_near] - c)[2][-1]   # least-squares refine
    if (obj_ref - c) @ n < 0:                                 # orient the normal up, toward the object
        n = -n
    d = -float(n @ c)                                         # plane: n.x + d = 0, n points up

    # Extent: inliers of this plane in the ref frame plus a few other frames (each rescaled by its own k),
    # so the table hidden behind the hand at the ref frame is still covered.
    others = [f for f in frames if f != ref]
    others = [others[i] for i in np.linspace(0, len(others) - 1, min(args.num_frames, len(others))).round().astype(int)] if others else []
    parts = [(pts_ref, rgb_ref)] + [frame_points(f) for f in others]
    pts, rgb = np.concatenate([p for p, _ in parts]), np.concatenate([r for _, r in parts])
    # In-plane basis: u = camera x projected onto the plane, v = n x u.
    u = np.array([1.0, 0, 0]) - n * n[0]
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    uv = np.stack([u, v], 1)
    q_all = (pts - c) @ uv
    q_obj = (obj_ref - c) @ uv
    # Coplanar points far from the object are usually other surfaces at the same height (shelves, counters).
    inl = (np.abs(pts @ n + d) < args.inlier_thresh) & (np.linalg.norm(q_all - q_obj, axis=1) < args.extent_radius)
    q = q_all[inl]
    lo, hi = np.percentile(q, 2, 0), np.percentile(q, 98, 0)
    center = c + u * (lo[0] + hi[0]) / 2 + v * (lo[1] + hi[1]) / 2
    size = hi - lo

    # Box mesh: top face on the plane, extruded down by --thickness.
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = u, v, n
    T[:3, 3] = center - n * args.thickness / 2
    box = trimesh.creation.box(extents=[size[0], size[1], args.thickness], transform=T)

    out = os.path.join(vd, "table")
    os.makedirs(out, exist_ok=True)
    box.export(os.path.join(out, "table_mesh.obj"))
    sel = rng.choice(np.flatnonzero(inl), min(int(inl.sum()), 60000), replace=False)
    trimesh.PointCloud(pts[sel], colors=np.c_[rgb[sel], np.full(len(sel), 255)]).export(
        os.path.join(out, "table_points.ply"))

    tilt = float(np.degrees(np.arccos(min(1.0, abs(n @ gravity)))))
    info = {
        "frame": "camera_frame (x right, y down, z forward), metres, same as DAID hand/object",
        "plane_normal_up": n.tolist(), "plane_d": d, "plane_point": c.tolist(),
        "surface_center": center.tolist(), "surface_basis_u": u.tolist(), "surface_basis_v": v.tolist(),
        "surface_size_uv_m": size.tolist(), "box_thickness_m": args.thickness,
        "qc": {
            "normal_vs_gravity_deg": tilt,
            "support_points_near_object": int(near.sum()),
            "support_inlier_fraction": float(inl_near.mean()),
            "extent_inliers": int(inl.sum()), "num_points": int(len(pts)),
            "inlier_rms_m": float(np.sqrt(np.mean((pts[inl] @ n + d) ** 2))),
            "object_ref_height_above_table_m": float(obj_ref @ n + d),
            "pointmap_scale_ref": float(k_by_frame[ref]), "pointmap_scale_median": k_med,
            "ref_frame": ref, "frames_used": [ref] + others,
        },
    }
    json.dump(info, open(os.path.join(out, "table.json"), "w"), indent=2)
    print(json.dumps(info["qc"], indent=2))
    print(f"table: {size[0]:.2f} x {size[1]:.2f} m -> {out}")


if __name__ == "__main__":
    main()
