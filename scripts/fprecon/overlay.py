"""QC overlay for one FoundationPose reconstruction: object render (orange), its mask (green outline) and the hand
(blue vertices) on a few frames, written to OUT/overlay.jpg.

  python scripts/fprecon/overlay.py --video-dir $CHOIR_ROOT/output/VID --out $PROJECT_STORAGE_ROOT/runs/fprecon/VID
  (fpose env)
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch
import trimesh

FP_DIR = os.environ.get("FP_DIR", "/data/sky/projects/CounterfactualHOI/build/FoundationPose")
sys.path.insert(0, FP_DIR)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "choir"))
sys.path.insert(0, HERE)
import nvdiffrast.torch as dr  # noqa: E402
from Utils import make_mesh_tensors, nvdiffrast_render  # noqa: E402
from hand_scale import hand_mesh_dir  # noqa: E402
from masks import object_mask, tracked_mesh  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=6)
    args = ap.parse_args()
    vd, out = args.video_dir, args.out
    K = np.array(json.load(open(os.path.join(vd, "inputs", "intrinsics.json")))["intrinsics"], float)
    poses = np.load(os.path.join(out, "object_poses.npy"))
    mt = make_mesh_tensors(trimesh.load(tracked_mesh(out), force="mesh", process=False))
    side = json.load(open(os.path.join(out, "hand_scale.json")))["side"]
    glctx = dr.RasterizeCudaContext()
    tiles = []
    for i in np.linspace(0, len(poses) - 1, args.frames).astype(int):
        img = cv2.imread(os.path.join(vd, "inputs", "frames", "rgb", f"{i}.png"))
        H, W = img.shape[:2]
        _, rd, _ = nvdiffrast_render(K=K, H=H, W=W, glctx=glctx, mesh_tensors=mt,
                                     ob_in_cams=torch.as_tensor(poses[i][None], device="cuda", dtype=torch.float))
        sil = rd[0].cpu().numpy() > 0
        img[sil] = (0.45 * img[sil] + 0.55 * np.array([40, 150, 240])).astype(np.uint8)
        m = object_mask(vd, out, i)
        if m is not None:
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cs, -1, (0, 255, 0), 3)
        hf = os.path.join(hand_mesh_dir(vd, side), f"hand_{i:05d}.obj")
        if os.path.exists(hf):
            v = np.asarray(trimesh.load(hf, process=False).vertices)
            p = (K @ v.T).T
            for x, y in (p[:, :2] / p[:, 2:3])[::3].astype(int):
                if 0 <= x < W and 0 <= y < H:
                    cv2.circle(img, (x, y), 3, (220, 90, 40), -1)
        cv2.putText(img, f"{i}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
        tiles.append(cv2.resize(img, (640, 360)))
    rows = [np.hstack(tiles[j:j + 3]) for j in range(0, len(tiles), 3)]
    cv2.imwrite(os.path.join(out, "overlay.jpg"), np.vstack(rows))
    print(os.path.join(out, "overlay.jpg"))


if __name__ == "__main__":
    main()
