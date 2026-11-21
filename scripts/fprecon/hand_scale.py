"""Metric scale for the Depth Pro 2 depth from the MANO hand (step 2 of the FoundationPose reconstruction).

  python scripts/fprecon/hand_scale.py --video-dir $CHOIR_ROOT/output/VID --out $PROJECT_STORAGE_ROOT/runs/fprecon/VID
  (sam3d env)

The Dyn-HaMR hand (CHOIR stage 1: stage1/hand/hand_meshes/[<side>/]hand_NNNNN.obj, OpenCV camera frame) is metric
through MANO's size prior, so its depth is the scale reference (masks: SAM3 when present, else CHOIR's SAM2; see masks.py). Per frame, the hand's camera-facing vertices
(z-buffered on a coarse pixel grid) that project inside the SAM2 hand mask (eroded, to stay off the silhouette
edge where depth mixes hand and table, and minus the object mask, which SAM2's hand mask sometimes swallows) are compared with the depth map at their pixels; the per-frame scale is
the median of z_hand / z_depth, and the clip scale is the median over frames, since the camera is static and Depth
Pro 2 is temporally consistent. Writes OUT/hand_scale.json (scale, per-frame scales and their spread).
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import cv2
import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "choir"))
sys.path.insert(0, HERE)
from masks import hand_mask, object_mask  # noqa: E402

def hand_side(vd):
    return subprocess.run([sys.executable, os.path.join(HERE, "..", "choir", "pick_hand_side.py"), vd],
                          capture_output=True, text=True, check=True).stdout.strip()


def hand_mesh_dir(vd, side):
    """Dyn-HaMR writes hand_meshes/<side>/ when it exported both hands ("dual"), else hand_meshes/ directly."""
    d = os.path.join(vd, "stage1", "hand", "hand_meshes")
    return os.path.join(d, side) if os.path.isdir(os.path.join(d, side)) else d


def visible_vertices(v, K, H, W, cell=6, tol=0.01):
    """Pixels and depths of the vertices nearest the camera in their cell (a cheap z-buffer)."""
    p = (K @ v.T).T
    uv = p[:, :2] / p[:, 2:3]
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H) & (v[:, 2] > 0.05)
    uv, z = uv[ok], v[ok, 2]
    cid = (uv[:, 1] // cell).astype(int) * (W // cell + 1) + (uv[:, 0] // cell).astype(int)
    zmin = np.full(cid.max() + 1, np.inf)
    np.minimum.at(zmin, cid, z)
    keep = z <= zmin[cid] + tol
    return uv[keep].astype(int), z[keep]


def corrected_hands(vd, out, smooth=9):
    """Dyn-HaMR hand vertices (N, 778, 3), faces and per-frame validity, each frame moved along the ray through
    its centroid onto the scaled depth map: frame i's z_hand / z_depth ratio s_i vs the clip scale s gives the
    depth factor s / s_i (median-filtered over `smooth` frames, interpolated where the hand mask was unusable).
    The translation keeps the hand's size and (nearly) its image position."""
    info = json.load(open(os.path.join(out, "hand_scale.json")))
    n = info["frames_total"]
    d = hand_mesh_dir(vd, info["side"])
    verts, valid, faces = np.zeros((n, 778, 3)), np.zeros(n, bool), None
    for i in range(n):
        f = os.path.join(d, f"hand_{i:05d}.obj")
        if os.path.exists(f):
            m = trimesh.load(f, process=False)
            verts[i], valid[i], faces = m.vertices, True, m.faces if faces is None else faces
    pf = {int(k): v for k, v in info["per_frame"].items()}
    idx = np.array(sorted(pf))
    s_i = np.interp(np.arange(n), idx, [pf[i] for i in idx])
    half = smooth // 2
    s_i = np.array([np.median(s_i[max(0, i - half):i + half + 1]) for i in range(n)])
    factor = info["scale"] / s_i
    c = verts.mean(1, keepdims=True)
    verts = verts + c * (factor[:, None, None] - 1)
    return verts, np.asarray(faces, np.uint32), valid, factor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--side", default=None, help="left/right (default: pick_hand_side.py)")
    args = ap.parse_args()
    vd = args.video_dir
    side = args.side or hand_side(vd)
    K = np.array(json.load(open(os.path.join(vd, "inputs", "intrinsics.json")))["intrinsics"], float)
    depth = np.load(os.path.join(args.out, "depth_dp2.npy"), mmap_mode="r")
    N, H, W = depth.shape
    kernel = np.ones((9, 9), np.uint8)
    per_frame = {}
    for f in sorted(glob.glob(os.path.join(hand_mesh_dir(vd, side), "hand_*.obj"))):
        i = int(os.path.basename(f)[5:10])
        m = hand_mask(vd, args.out, i, side)
        if i >= N or m is None or not 0.002 < m.mean() < 0.2:
            continue
        om = object_mask(vd, args.out, i)
        m = cv2.erode(m.astype(np.uint8), kernel) > 0
        if om is not None:                            # SAM2's hand mask can swallow a held object
            m &= ~(cv2.dilate(om.astype(np.uint8), kernel) > 0)
        uv, z = visible_vertices(np.asarray(trimesh.load(f, process=False).vertices), K, H, W)
        inside = m[uv[:, 1], uv[:, 0]]
        if inside.sum() < 40:
            continue
        zd = depth[i][uv[inside, 1], uv[inside, 0]].astype(np.float64)
        good = zd > 0.05
        if good.sum() >= 40:
            per_frame[i] = float(np.median(z[inside][good] / zd[good]))
    s = np.array(list(per_frame.values()))
    if len(s) < 5:
        sys.exit(f"only {len(s)} usable hand frames in {vd}")
    scale = float(np.median(s))
    info = {"side": side, "scale": scale, "frames_used": len(s), "frames_total": int(N),
            "per_frame_p10_p50_p90": np.percentile(s, [10, 50, 90]).tolist(),
            "mad_rel": float(np.median(np.abs(s - scale)) / scale), "per_frame": per_frame}
    json.dump(info, open(os.path.join(args.out, "hand_scale.json"), "w"), indent=2)
    print(f"{os.path.basename(vd)}: hand={side} scale={scale:.3f} from {len(s)}/{N} frames, "
          f"p10-p90 {info['per_frame_p10_p50_p90'][0]:.3f}-{info['per_frame_p10_p50_p90'][2]:.3f}, "
          f"MAD {100 * info['mad_rel']:.1f}%")


if __name__ == "__main__":
    main()
