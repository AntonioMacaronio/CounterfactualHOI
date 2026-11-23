"""Give every imported clip its table as one support disk (FlashDexRetarget's desk), instead of fitting disks.

  python -I scripts/flashdex/add_table.py DATASET [--radius 0.3]      (the flashdex env; FDR=... to override)

FlashDexRetarget puts the table under the objects as up to 4 thin disks, fitted where an object stands still
(desk.mode=fit). On our clips that fit fails: FoundationPose's frame-to-frame jitter breaks most still segments,
so 6 of 10 clips lose the disk under the placed object (rejected as obj_offdisk_untouched) and 4 get no disk at
all (their objects would start 0.8 m above the floor). Our clips are all on one flat table, so each gets one disk
centred under the object's path (radius --radius; the env adds radius_margin 0.1 and rounds up to its 0.4 m bucket),
its top at the object's resting height on the first frames: the median lowest hull point over the first 10 frames
(the reconstruction's table plane is 0.4-2.5 cm off the object's bottom; the simulated object must start resting on
the disk). Writes support_disks into DATASET/human_demo/*.npz; run kinematic retargeting with desk.mode=keep.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

FDR = os.environ.get("FDR", "/data/sky/projects/CounterfactualHOI/build/FlashDexRetarget")
sys.path.insert(0, os.path.join(FDR, "src"))
from retarget.human_demo import clips as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--radius", type=float, default=0.3)
    ap.add_argument("--rest-frames", type=int, default=10)
    args = ap.parse_args()
    ds = Path(args.dataset)
    meshes = C.ObjectMeshes(ds)
    rows = {}
    for path in C.list_clips(ds / "human_demo"):
        clip = C.load_clip(path)
        (side,) = C.objects(clip)
        mesh = meshes.get(C.obj_id(clip, side), C.obj_scale(clip, side))
        pose = C.obj_pose(clip, side).astype(np.float64)
        hull = np.einsum("tij,vj->tvi", pose[:, :3, :3], mesh["hull"]) + pose[:, None, :3, 3]
        lo, hi = hull[..., :2].reshape(-1, 2).min(0), hull[..., :2].reshape(-1, 2).max(0)
        c = (lo + hi) / 2
        reach = float(np.linalg.norm(hull[..., :2].reshape(-1, 2) - c, axis=1).max())
        z_top = float(np.median(hull[: args.rest_frames, :, 2].min(1)))
        z_end = float(np.median(hull[-args.rest_frames:, :, 2].min(1)))
        clip["support_disks"] = np.array([[c[0], c[1], z_top, args.radius, 0.01]], np.float32)
        C.save_clip(path, clip)
        rows[path.stem] = {"centre": [round(float(x), 3) for x in c], "z_top": round(z_top, 4),
                           "end_bottom_above_top_cm": round(100 * (z_end - z_top), 2),
                           "object_reach_m": round(reach, 3), "covered": reach < args.radius + 0.1}
        print(path.stem, rows[path.stem])
    json.dump(rows, open(ds / "tables.json", "w"), indent=1)


if __name__ == "__main__":
    main()
