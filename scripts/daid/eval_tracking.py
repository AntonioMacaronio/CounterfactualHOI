"""Quantify Do-as-I-Do object tracking quality for one run (no ground truth needed).

  python scripts/daid/eval_tracking.py --video-dir RUN/ --object-id red_apple [--fps 24]   (sam3d env)

Per frame:
  * visible silhouette IoU: convex hull of the projected tracked mesh vs the SAM3 object mask, ignoring
    pixels under the (dilated) hand mask (occluded by the hand, so unobservable). All four IMG_5710 objects
    are near-convex, so the hull is a fair silhouette.
  * tracked rotation speed (deg/frame, from consecutive quaternions) and its in-plane component (twist
    about the camera axis), which is comparable to TAPIR's 2D rotation estimate (motion_stats.json).
Writes RUN/qc/tracking_eval.{json,png}.
"""

import argparse
import glob
import json
import os

import cv2
import matplotlib
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_mask(vd, f, name, shape):
    p = os.path.join(vd, "video_segmentation", "masks", f"frame_{f:06d}_masks", f"{name}.png")
    if not os.path.exists(p):
        return np.zeros(shape, bool)
    m = cv2.imread(p, 0)
    return cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 127


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--object-id", required=True)
    ap.add_argument("--anchor-hand", default="right")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--layout", default="filled", choices=["filled", "optimized", "raw"])
    args = ap.parse_args()
    vd = args.video_dir
    cv_dir = os.path.join(vd, "obj_tracking_out", args.object_id, "combined_visualization")
    name = {"filled": "layout_camera_frame_filled.json", "optimized": "layout_camera_frame_optimized.json",
            "raw": "layout_camera_frame.json"}[args.layout]
    lay = json.load(open(os.path.join(cv_dir, name)))
    opt = lay.get("translation_scale_optimization", {})
    mesh_scale = opt.get("mesh_scale")
    poses = {}
    for o in lay["objects"]:
        f = o.get("frame_index", o.get("frame_idx"))
        if f is None:
            continue
        lts = o["local_to_scene"]
        q = lts["quat_wxyz_camera_frame"]
        poses[int(f)] = (R.from_quat([q[1], q[2], q[3], q[0]]), np.array(lts["translation_camera_frame"], float),
                         lts.get("scale"))
    frames = sorted(poses)
    mesh_frame = json.load(open(os.path.join(vd, "config.json")))["frame_number"]
    mesh = trimesh.load(os.path.join(vd, "video_segmentation", "masks", f"frame_{mesh_frame:06d}_masks",
                                     args.object_id, f"{args.object_id}.obj"), force="mesh")
    verts = np.asarray(mesh.vertices)
    verts = verts[np.random.default_rng(0).choice(len(verts), min(len(verts), 4000), replace=False)]
    K = np.load(os.path.join(vd, "all_frames", f"{frames[0]:06d}_intrinsics.npy"))
    H, W = cv2.imread(os.path.join(vd, "all_frames", f"{frames[0]:06d}.png")).shape[:2]
    shape = (H // 2, W // 2)                      # evaluate at half resolution
    Kh = K.copy(); Kh[:2] /= 2

    iou, rot_speed, twist = [], [], []
    prev = None
    for f in frames:
        rot, t, s = poses[f]
        sc = mesh_scale if mesh_scale is not None else (s[0] if isinstance(s, list) else 1.0)
        v = rot.apply(verts * sc) + t
        z = np.clip(v[:, 2], 1e-4, None)
        uv = np.stack([Kh[0, 0] * v[:, 0] / z + Kh[0, 2], Kh[1, 1] * v[:, 1] / z + Kh[1, 2]], 1)
        pred = np.zeros(shape, np.uint8)
        hull = cv2.convexHull(np.clip(uv, -1e4, 1e4).astype(np.int32))
        cv2.fillConvexPoly(pred, hull, 1)
        pred = pred > 0
        gt = load_mask(vd, f, args.object_id, shape)
        hand = cv2.dilate(load_mask(vd, f, f"{args.anchor_hand}_hand_0", shape).astype(np.uint8),
                          np.ones((7, 7), np.uint8)) > 0
        vis = ~(hand & ~gt)                       # pixels not hidden behind the hand
        inter, union = (pred & gt & vis).sum(), ((pred | gt) & vis).sum()
        iou.append(inter / union if union else np.nan)
        if prev is not None:
            rel = rot * prev.inv()                # rotation from previous frame, in the camera frame
            rv = rel.as_rotvec()
            rot_speed.append(np.degrees(np.linalg.norm(rv)))
            twist.append(np.degrees(rv[2]))       # component about the camera's optical axis
        else:
            rot_speed.append(0.0); twist.append(0.0)
        prev = rot
    iou, rot_speed, twist = map(np.array, (iou, rot_speed, twist))

    tapir = {}
    ms = glob.glob(os.path.join(vd, f"perframe_tracking_{args.object_id}", "motion_stats.json"))
    if ms:
        for e in json.load(open(ms[0])):
            if e.get("status") == "ok":
                tapir[e["frame"]] = e["rotation_deg"]     # 2D in-plane rotation frame n -> n+1
    tap = np.array([tapir.get(f - 1, np.nan) for f in frames])
    ok = np.isfinite(tap) & (np.arange(len(frames)) > 0)
    # sign convention of TAPIR's 2D angle is unknown: report the better-matching sign
    corr = max(np.corrcoef(twist[ok], tap[ok])[0, 1], np.corrcoef(-twist[ok], tap[ok])[0, 1]) if ok.sum() > 5 else np.nan
    hand_valid = np.load(glob.glob(os.path.join(vd, "*", "all_hand_meshes.npz"))[0])[f"{args.anchor_hand}_valid"]
    inhand = np.array([bool(hand_valid[min(f, len(hand_valid) - 1)]) for f in frames])

    summary = {
        "layout": name, "frames": len(frames),
        "visible_iou_mean": float(np.nanmean(iou)), "visible_iou_median": float(np.nanmedian(iou)),
        "visible_iou_mean_hand_present": float(np.nanmean(iou[inhand])),
        "frac_frames_iou_below_0.5": float(np.nanmean(iou < 0.5)),
        "tracked_rot_speed_mean_deg_per_frame": float(rot_speed.mean()),
        "tracked_total_rotation_deg": float(rot_speed.sum()),
        "tracked_inplane_total_deg": float(np.abs(twist).sum()),
        "tapir_inplane_total_deg": float(np.nansum(np.abs(tap))),
        "inplane_corr_tracked_vs_tapir": float(corr),
        "frac_steps_over_20deg": float(np.mean(rot_speed > 20)),
    }
    os.makedirs(os.path.join(vd, "qc"), exist_ok=True)
    json.dump({"summary": summary, "frame": frames, "visible_iou": iou.tolist(), "rot_speed_deg": rot_speed.tolist(),
               "twist_deg": twist.tolist(), "tapir_rot_deg": tap.tolist()},
              open(os.path.join(vd, "qc", "tracking_eval.json"), "w"), indent=1)

    ts = np.array(frames) / args.fps
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True)
    a1.plot(ts, iou, color="#d9822b"); a1.set_ylim(0, 1); a1.set_ylabel("visible IoU")
    a1.set_title(f"{os.path.basename(vd)}: tracked silhouette vs SAM3 mask")
    a2.plot(ts, np.abs(twist), color="#4c6ef5", label="tracked |in-plane| rotation")
    a2.plot(ts, np.abs(tap), color="#868e96", label="TAPIR |2D| rotation")
    a2.plot(ts, rot_speed, color="#2b8a3e", alpha=0.6, label="tracked |3D| rotation")
    a2.set_ylabel("deg / frame"); a2.set_xlabel("time (s)"); a2.legend(fontsize=7, frameon=False, loc="upper left")
    for a in (a1, a2):
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(vd, "qc", "tracking_eval.png"), dpi=120)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
