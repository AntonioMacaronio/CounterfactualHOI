"""Check a FoundationPose object track against the video on every frame, occlusion-aware.

  python scripts/fprecon/verify_track.py --video-dir $CHOIR_ROOT/output/VID --out $PROJECT_STORAGE_ROOT/runs/fprecon/VID
  (fpose env)

Mask IoU is unfair while the hand covers the object (the mask shrinks, the render does not), so per frame:
  coverage = |render & object mask| / |object mask|        visible object pixels explained by the render
  spill    = |render - object mask - hand mask| / |render|  render on pixels that are neither object nor hand
A correct pose has coverage ~1 and spill ~0 whatever the occlusion. Frames without an object mask are not assessed. Frames with coverage < 0.7 or spill > 0.3
are flagged; OUT/verify.json has the per-frame numbers and flagged runs, OUT/verify.jpg shows the worst frames
(object mask green outline, render orange, hand mask blue outline).
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import trimesh

FP_DIR = os.environ.get("FP_DIR", "/data/sky/projects/CounterfactualHOI/build/FoundationPose")
sys.path.insert(0, FP_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "choir"))
import nvdiffrast.torch as dr  # noqa: E402
from Utils import make_mesh_tensors, nvdiffrast_render  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from masks import hand_mask, object_mask, tracked_mesh  # noqa: E402


def runs_of(flags):
    out, start = [], None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = i
        if not f and start is not None:
            out.append([start, i - 1])
            start = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--show", type=int, default=6, help="worst frames to draw")
    args = ap.parse_args()
    vd, out = args.video_dir, args.out
    K = np.array(json.load(open(os.path.join(vd, "inputs", "intrinsics.json")))["intrinsics"], float)
    poses = np.load(os.path.join(out, "object_poses.npy"))
    mt = make_mesh_tensors(trimesh.load(tracked_mesh(out), force="mesh", process=False))
    side = json.load(open(os.path.join(out, "hand_scale.json")))["side"]
    glctx = dr.RasterizeCudaContext()
    H, W = cv2.imread(os.path.join(vd, "inputs", "frames", "rgb", "0.png")).shape[:2]
    cov, spill, sils = [], [], {}
    for i in range(len(poses)):
        _, rd, _ = nvdiffrast_render(K=K, H=H, W=W, glctx=glctx, mesh_tensors=mt,
                                     ob_in_cams=torch.as_tensor(poses[i][None], device="cuda", dtype=torch.float))
        sil = rd[0].cpu().numpy() > 0
        om = object_mask(vd, out, i)
        hm = hand_mask(vd, out, i, side)
        om = np.zeros_like(sil) if om is None else om
        hm = np.zeros_like(sil) if hm is None or hm.mean() > 0.2 else cv2.dilate(hm.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        hm &= ~om
        cov.append(float((sil & om).sum() / max(om.sum(), 1)) if om.sum() > 200 else float("nan"))
        # no object mask (hidden, or the segmenter missed it): nothing to check the render against
        spill.append(float((sil & ~om & ~hm).sum() / max(sil.sum(), 1)) if om.sum() > 200 else float("nan"))
    cov, spill = np.array(cov), np.array(spill)
    bad = (np.nan_to_num(cov, nan=1.0) < 0.7) | (np.nan_to_num(spill, nan=0.0) > 0.3)
    flagged = [r for r in runs_of(bad) if r[1] - r[0] >= 1]
    info = {"frames": len(poses), "coverage_median": float(np.nanmedian(cov)), "spill_median": float(np.nanmedian(spill)),
            "coverage_p10": float(np.nanpercentile(cov, 10)), "spill_p90": float(np.nanpercentile(spill, 90)),
            "unassessed_frames": int(np.isnan(spill).sum()),
            "flagged_frac": float(bad.mean()), "flagged_runs": flagged,
            "per_frame": [{"coverage": None if np.isnan(c) else float(c), "spill": None if np.isnan(s) else float(s)}
                          for c, s in zip(cov, spill)]}
    json.dump(info, open(os.path.join(out, "verify.json"), "w"), indent=1)

    score = np.nan_to_num(1 - cov, nan=0) + np.nan_to_num(spill, nan=0)
    worst = sorted(np.argsort(-score)[:args.show].tolist())
    tiles = []
    for i in worst:
        img = cv2.imread(os.path.join(vd, "inputs", "frames", "rgb", f"{i}.png"))
        _, rd, _ = nvdiffrast_render(K=K, H=H, W=W, glctx=glctx, mesh_tensors=mt,
                                     ob_in_cams=torch.as_tensor(poses[i][None], device="cuda", dtype=torch.float))
        sil = rd[0].cpu().numpy() > 0
        img[sil] = (0.45 * img[sil] + 0.55 * np.array([40, 150, 240])).astype(np.uint8)
        for m, col in ((object_mask(vd, out, i), (0, 255, 0)), (hand_mask(vd, out, i, side), (255, 120, 0))):
            if m is not None and m.mean() < 0.2:
                cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cs, -1, col, 3)
        c = "--" if np.isnan(cov[i]) else f"{cov[i]:.2f}"
        sp = "--" if np.isnan(spill[i]) else f"{spill[i]:.2f}"
        cv2.putText(img, f"{i} cov {c} spill {sp}", (25, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 6)
        tiles.append(cv2.resize(img, (640, 360)))
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(os.path.join(out, "verify.jpg"), np.vstack([np.hstack(tiles[j:j + 3]) for j in range(0, len(tiles), 3)]))
    print(f"{os.path.basename(vd)}: coverage median {info['coverage_median']:.2f} (p10 {info['coverage_p10']:.2f}), "
          f"spill median {info['spill_median']:.2f} (p90 {info['spill_p90']:.2f}), flagged {100 * info['flagged_frac']:.0f}% "
          f"of frames, runs {flagged}")


if __name__ == "__main__":
    main()
