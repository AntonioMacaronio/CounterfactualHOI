"""QC for one Do-as-I-Do run (+ table): reprojection overlays and height-above-table over time.

  python scripts/daid/qc_recon.py --video-dir RUN/ --object-id red_apple   (sam3d env)

Writes RUN/qc/: overlay.jpg (object = orange, hand = blue, table = green, projected into the video
frames), heights.png and qc.json, with per-frame heights above the table plane of the object's and
hand's lowest points (negative = below the table surface, i.e. penetration) and summary statistics.
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


def project(pts, K):
    z = np.clip(pts[:, 2], 1e-6, None)
    return np.stack([K[0, 0] * pts[:, 0] / z + K[0, 2], K[1, 1] * pts[:, 1] / z + K[1, 2]], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--object-id", required=True)
    ap.add_argument("--anchor-hand", default="right")
    ap.add_argument("--fps", type=float, default=24.0)
    args = ap.parse_args()
    vd = args.video_dir

    cv_dir = os.path.join(vd, "obj_tracking_out", args.object_id, "combined_visualization")
    lay_path = os.path.join(cv_dir, "layout_camera_frame_filled.json")      # fill_unanchored.py output
    if not os.path.exists(lay_path):
        lay_path = os.path.join(cv_dir, "layout_camera_frame_optimized.json")
    lay = json.load(open(lay_path))
    opt = lay["translation_scale_optimization"]
    poses = {}
    for o in lay["objects"]:
        f = o.get("frame_index", o.get("frame_idx"))
        if f is not None:
            lts = o["local_to_scene"]
            q = lts["quat_wxyz_camera_frame"]
            poses[int(f)] = (R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix(),
                             np.array(lts["translation_camera_frame"], float))
    frames = sorted(poses)
    mesh = trimesh.load(os.path.join(vd, "video_segmentation", "masks", f"frame_{json.load(open(os.path.join(vd, 'config.json')))['frame_number']:06d}_masks",
                                     args.object_id, f"{args.object_id}.obj"), force="mesh")
    ov = np.asarray(mesh.vertices) * opt["mesh_scale"]
    hands = np.load(glob.glob(os.path.join(vd, "*", "all_hand_meshes.npz"))[0])
    hv = hands[f"{args.anchor_hand}_vertices"]
    hvalid = hands[f"{args.anchor_hand}_valid"]
    table = json.load(open(os.path.join(vd, "table", "table.json")))
    n, d = np.array(table["plane_normal_up"]), table["plane_d"]
    tbox = trimesh.load(os.path.join(vd, "table", "table_mesh.obj"), force="mesh")

    obj_low, hand_low, obj_center = [], [], []
    for f in frames:
        Rm, t = poses[f]
        v = ov @ Rm.T + t
        obj_low.append(float((v @ n + d).min()))
        obj_center.append(float(v.mean(0) @ n + d))
        hi = min(f, len(hv) - 1)
        hand_low.append(float((hv[hi] @ n + d).min()) if hvalid[hi] else np.nan)
    obj_low, hand_low, obj_center = map(np.array, (obj_low, hand_low, obj_center))
    t_s = np.array(frames) / args.fps

    size = np.ptp(ov @ poses[opt["ref_frame"]][0].T if opt["ref_frame"] in poses else ov, axis=0)
    rest = obj_low[: max(1, int(np.searchsorted(frames, opt["ref_frame"])) + 1)]
    qc = {
        "frames": len(frames),
        "object_extent_m": size.round(4).tolist(),
        "object_rest_gap_m_at_start": float(np.median(rest)),
        "object_min_height_m": float(np.nanmin(obj_low)),
        "object_max_lift_m": float(np.nanmax(obj_low)),
        "object_penetration_frames(>1cm)": int((obj_low < -0.01).sum()),
        "hand_min_height_m": float(np.nanmin(hand_low)),
        "hand_penetration_frames(>1cm)": int((hand_low < -0.01).sum()),
        "table_size_m": table["surface_size_uv_m"],
        "table_normal_vs_gravity_deg": table["qc"]["normal_vs_gravity_deg"],
        "scale_fit_error_mean_m_after": float(np.mean([p["error_after"] for p in opt["per_frame"]])),
    }
    os.makedirs(os.path.join(vd, "qc"), exist_ok=True)
    json.dump({"summary": qc, "frame": frames, "object_lowest_height_m": obj_low.tolist(),
               "object_center_height_m": obj_center.tolist(), "hand_lowest_height_m": hand_low.tolist()},
              open(os.path.join(vd, "qc", "qc.json"), "w"), indent=1)

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.axhline(0, color="#3a7d44", lw=1.5, label="table surface")
    ax.plot(t_s, obj_low * 100, color="#d9822b", label=f"{args.object_id}: lowest point")
    ax.plot(t_s, hand_low * 100, color="#4c6ef5", label="hand: lowest point")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("height above table (cm)")
    ax.set_title(os.path.basename(vd))
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(vd, "qc", "heights.png"), dpi=120)

    # Overlays: object / hand vertices and the table outline projected into a few frames.
    K = np.load(os.path.join(vd, "all_frames", f"{frames[0]:06d}_intrinsics.npy"))
    tiles = []
    for f in [frames[int(i)] for i in np.linspace(0, len(frames) - 1, 6)]:
        img = cv2.imread(os.path.join(vd, "all_frames", f"{f:06d}.png"))
        Rm, t = poses[f]
        for pts, col in ((ov @ Rm.T + t, (43, 130, 217)), (hv[min(f, len(hv) - 1)], (245, 110, 76))):
            if col == (245, 110, 76) and not hvalid[min(f, len(hv) - 1)]:
                continue
            uv = project(pts[:: max(1, len(pts) // 1500)], K)
            for x, y in uv[np.isfinite(uv).all(1)].astype(int):
                if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                    cv2.circle(img, (x, y), 2, col, -1)
        top = tbox.vertices[(tbox.vertices @ n + d) > -1e-3]           # 4 top corners of the table box
        if len(top) == 4:
            q = (top - top.mean(0)) @ np.stack([table["surface_basis_u"], table["surface_basis_v"]], 1)
            order = np.argsort(np.arctan2(q[:, 1], q[:, 0]))         # corners in order around the rectangle
            cv2.polylines(img, [project(top[order], K).astype(np.int32)], True, (68, 125, 58), 3)
        cv2.putText(img, f"frame {f}", (20, 50), 0, 1.5, (255, 255, 255), 3)
        tiles.append(cv2.resize(img, (480, int(480 * img.shape[0] / img.shape[1]))))
    cv2.imwrite(os.path.join(vd, "qc", "overlay.jpg"), np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]))
    print(json.dumps(qc, indent=2))


if __name__ == "__main__":
    main()
