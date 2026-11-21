"""Build a Do-as-I-Do retargeting input dir (retargeting/launch.py --raw-dir) from one of our reconstructions.

  python scripts/retarget/prepare_daid_input.py --video-dir RUN/ --object-id white_case --out IN/   (sam3d env)

Retargeting stage 1 (retargeting/pipeline/process_dataset.py) reads, relative to --raw-dir:
  config.json (object_names[0], anchor_hand), <task>/all_hand_meshes.npz, gravity.json (vec3d = up in the
  camera frame), obj_tracking_out/<obj>/combined_visualization/layout_camera_frame_optimized.json and
  video_segmentation/masks/frame_*_masks/<obj>/<obj>.obj.
Differences from our run dirs that this fixes:
  * object_names holds the text prompt ("white case") but the folders use the id ("white_case");
  * HaWoR leaves NaN vertices/joints on some frames it marks valid: those frames are marked invalid;
  * the object layout is our anchor-repaired one (layout_camera_frame_filled.json) when present;
  * GeoCalib gravity is replaced by the table normal when they disagree by more than --max-gravity-deg
    (GeoCalib was 26-33 deg off on plain table+wall scenes; retargeting aligns the scene and floor to it).
Large files are symlinked; only the small JSON/NPZ are written.
"""

import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--object-id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-gravity-deg", type=float, default=10.0)
    args = ap.parse_args()
    vd, obj, out = os.path.abspath(args.video_dir), args.object_id, os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    cfg = json.load(open(os.path.join(vd, "config.json")))
    cfg["object_names"] = [obj]
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)

    hand_npz = glob.glob(os.path.join(vd, "*", "all_hand_meshes.npz"))[0]
    task = os.path.basename(os.path.dirname(hand_npz))
    hands = dict(np.load(hand_npz))
    for side in ("left", "right"):
        bad = ~np.isfinite(hands[f"{side}_vertices"]).all(axis=(1, 2)) | ~np.isfinite(hands[f"{side}_joints"]).all(axis=(1, 2))
        print(f"{side}: {int((bad & hands[f'{side}_valid']).sum())} valid frames with NaN -> invalid")
        hands[f"{side}_valid"] = hands[f"{side}_valid"] & ~bad
        for k in ("vertices", "joints", "rot", "trans", "hand_pose", "betas"):
            if f"{side}_{k}" in hands:
                hands[f"{side}_{k}"] = np.nan_to_num(hands[f"{side}_{k}"])
    os.makedirs(os.path.join(out, task), exist_ok=True)
    np.savez(os.path.join(out, task, "all_hand_meshes.npz"), **hands)

    cv_src = os.path.join(vd, "obj_tracking_out", obj, "combined_visualization")
    layout = os.path.join(cv_src, "layout_camera_frame_filled.json")
    if not os.path.exists(layout):
        layout = os.path.join(cv_src, "layout_camera_frame_optimized.json")
    cv_dst = os.path.join(out, "obj_tracking_out", obj, "combined_visualization")
    os.makedirs(cv_dst, exist_ok=True)
    dst = os.path.join(cv_dst, "layout_camera_frame_optimized.json")
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(layout, dst)

    mesh_frame = cfg["frame_number"]
    mdir = f"frame_{mesh_frame:06d}_masks"
    os.makedirs(os.path.join(out, "video_segmentation", "masks", mdir), exist_ok=True)
    link = os.path.join(out, "video_segmentation", "masks", mdir, obj)
    if not os.path.lexists(link):
        os.symlink(os.path.join(vd, "video_segmentation", "masks", mdir, obj), link)

    grav = json.load(open(os.path.join(vd, "gravity.json")))
    g = np.array(grav["vec3d"], float); g /= np.linalg.norm(g)
    table = os.path.join(vd, "table", "table.json")
    if os.path.exists(table):
        n = np.array(json.load(open(table))["plane_normal_up"], float)
        ang = float(np.degrees(np.arccos(np.clip(abs(g @ n), -1, 1))))
        if ang > args.max_gravity_deg:
            grav = dict(grav, vec3d=n.tolist(), source=f"table_normal (GeoCalib off by {ang:.1f} deg)")
        else:
            grav = dict(grav, source=f"geocalib ({ang:.1f} deg from table normal)")
    json.dump(grav, open(os.path.join(out, "gravity.json"), "w"), indent=2)
    print(f"task={task} gravity={grav.get('source', 'geocalib')} -> {out}")


if __name__ == "__main__":
    main()
