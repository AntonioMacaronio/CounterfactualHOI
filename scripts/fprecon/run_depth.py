"""Depth Pro 2 metric video depth for one CHOIR clip (step 1 of the FoundationPose reconstruction).

  python scripts/fprecon/run_depth.py --video-dir $CHOIR_ROOT/output/VID --out $PROJECT_STORAGE_ROOT/runs/fprecon/VID
  (depthpro2 venv)

Reads the clip's frames (inputs/frames/rgb/<i>.png, CHOIR's interaction-cropped video) and runs Depth Pro 2 H+ in
its temporal (buffered, 16-frame window) mode with the focal length fixed to CHOIR's intrinsics, so that depth,
the Dyn-HaMR hands and FoundationPose share one camera. Writes OUT/depth_dp2.npy, (N, H, W) float16 z-depth in
meters (memory-mapped by the later steps), and OUT/depth_dp2.json with the focal used and Depth Pro 2's own focal
estimate for reference.
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import torch
from PIL import Image

from depth_pro_2 import create_model_and_transforms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="depthpro2-h")
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--overlap", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rgb_dir = os.path.join(args.video_dir, "inputs", "frames", "rgb")
    paths = sorted(glob.glob(os.path.join(rgb_dir, "*.png")), key=lambda p: int(os.path.basename(p)[:-4]))
    K = np.array(json.load(open(os.path.join(args.video_dir, "inputs", "intrinsics.json")))["intrinsics"], float)
    model, transform = create_model_and_transforms(args.model)
    load = lambda p: transform(np.asarray(Image.open(p).convert("RGB")))

    with torch.no_grad():
        own_f = float(model.infer(load(paths[0]).to(next(model.parameters()).device))["focallength_px"])
    W, H = Image.open(paths[0]).size
    depth = np.lib.format.open_memmap(os.path.join(args.out, "depth_dp2.npy"), mode="w+", dtype=np.float16,
                                      shape=(len(paths), H, W))
    t0 = time.time()
    for i, pred in enumerate(model.infer_video((load(p) for p in paths), window=args.window, overlap=args.overlap,
                                               f_px=float(K[0, 0]))):
        depth[i] = np.clip(pred["depth"].float().cpu().numpy(), 0, 60).astype(np.float16)
    depth.flush()
    info = {"frames": len(paths), "height": H, "width": W, "f_px_used": float(K[0, 0]), "f_px_depthpro2": own_f,
            "model": args.model, "window": args.window, "overlap": args.overlap, "seconds": time.time() - t0,
            "median_depth_m": float(np.median(depth[0][depth[0] > 0]))}
    json.dump(info, open(os.path.join(args.out, "depth_dp2.json"), "w"), indent=2)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
