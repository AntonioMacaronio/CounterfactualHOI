"""Fix object translations on frames where Do-as-I-Do's hand anchor is missing or unreliable.

  python scripts/daid/fill_unanchored.py --video-dir RUN/ --object-id red_apple                (sam3d env)
  python scripts/daid/fill_unanchored.py --video-dir RUN/ --object-id red_apple --suggest-ref  # prints a frame

optimize_translation_scale.py fits the object's metric translation per frame from a pointmap-to-metric
factor k = (HaWoR hand depth) / (pointmap hand depth). Two failure modes:
  * no hand mask (before the hand enters / after it leaves): the frame is skipped and keeps only the
    ref-frame scaling, which puts the resting object far off along the camera ray;
  * hand cut off at the image border (entering / leaving): k is an outlier (e.g. 0.95 or 0.17 against
    ~0.41 for the rest of the clip) and the object jumps toward or away from the camera.
k varies smoothly, so a frame is "reliable" when its k is within --max-k-dev of the median k of its
anchored neighbours (+-window frames). Unreliable frames hold the translation of the nearest reliable
frame (the object is at rest or barely moving there); rotations from tracking are kept. Writes
layout_camera_frame_filled.json with "translation_source" per frame.

--suggest-ref prints the nearest reliable frame to the current ref frame (the ref frame's k sets the
mesh's metric scale, so finish_recon.sh re-runs the optimization from there if the ref is unreliable).
"""

import argparse
import copy
import json
import os

import numpy as np


def reliable_frames(per_frame, window, max_dev):
    k = {p["frame_idx"]: p["pointmap_scale"] for p in per_frame}
    frames = np.array(sorted(k))
    ks = np.array([k[f] for f in frames])
    ok = []
    for f, kf in zip(frames, ks):
        nb = ks[np.abs(frames - f) <= window]
        if abs(kf / np.median(nb) - 1) <= max_dev:
            ok.append(int(f))
    return np.array(ok), frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--object-id", required=True)
    ap.add_argument("--window", type=int, default=8, help="+- frames for the local median of k")
    ap.add_argument("--max-k-dev", type=float, default=0.15, help="max relative deviation from the local median")
    ap.add_argument("--suggest-ref", action="store_true")
    args = ap.parse_args()
    cv = os.path.join(args.video_dir, "obj_tracking_out", args.object_id, "combined_visualization")
    src = os.path.join(cv, "layout_camera_frame_optimized.json")
    lay = json.load(open(src))
    opt = lay["translation_scale_optimization"]
    good, anchored = reliable_frames(opt["per_frame"], args.window, args.max_k_dev)
    if len(good) == 0:
        raise SystemExit("no reliable hand-anchored frames in " + src)

    if args.suggest_ref:
        ref = opt["ref_frame"]
        print(ref if ref in set(good.tolist()) else int(good[np.argmin(np.abs(good - ref))]))
        return

    out = copy.deepcopy(lay)
    src_t = {int(o.get("frame_index", o.get("frame_idx"))): o["local_to_scene"]["translation_camera_frame"]
             for o in lay["objects"] if o.get("frame_index", o.get("frame_idx")) is not None}
    good_set, held = set(good.tolist()), 0
    for o in out["objects"]:
        f = o.get("frame_index", o.get("frame_idx"))
        if f is None:
            continue
        lts = o["local_to_scene"]
        if int(f) in good_set:
            lts["translation_source"] = "anchored"
            continue
        nearest = int(good[np.argmin(np.abs(good - int(f)))])
        lts["translation_camera_frame"] = src_t[nearest]
        lts["translation_source"] = f"held_from_{nearest}"
        held += 1
    out["translation_fill"] = {
        "method": "hold_nearest_reliable_anchor", "window": args.window, "max_k_dev": args.max_k_dev,
        "held_frames": held, "anchored_frames": int(len(anchored)), "reliable_frames": int(len(good)),
        "rejected_anchored_frames": sorted(set(anchored.tolist()) - good_set),
    }
    dst = os.path.join(cv, "layout_camera_frame_filled.json")
    json.dump(out, open(dst, "w"), indent=2)
    print(f"held {held} frames ({len(anchored)} anchored, {len(anchored) - len(good)} rejected as k outliers: "
          f"{out['translation_fill']['rejected_anchored_frames'][:12]}) -> {dst}")


if __name__ == "__main__":
    main()
