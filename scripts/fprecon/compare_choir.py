"""Compare Open-CHOIR and the FoundationPose + Depth Pro 2 pipeline on the real clips (markdown table + JSON).

  python scripts/fprecon/compare_choir.py [--out ../outputs/fprecon_viser/compare.json]   (sam3d env)

Per clip: object silhouette IoU against the object mask (SAM3 when present, the same for both methods) on the first and last frame (the hand is off the
object there, so the mask is the whole object), and the object's height above the table at the last frame.
CHOIR: its final posed meshes; its table height comes from the viewer JSON (scene rescaled so the object starts
on the table, outputs/choir_viser/<VID>.json). New pipeline: FoundationPose's mesh + poses, table from the same
hand-scaled depth (outputs/fprecon_viser/<VID>.json).
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "choir"))
sys.path.insert(0, HERE)
from masks import object_mask  # noqa: E402

STORAGE = os.environ.get("PROJECT_STORAGE_ROOT", "/data/sky/projects/CounterfactualHOI")
CHOIR_ROOT = os.environ.get("CHOIR_ROOT", os.path.join(STORAGE, "build", "CHOIR"))
OUTPUTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "outputs")


def sil_iou(verts, faces, K, mask):
    p = (K @ verts.T).T
    p = (p[:, :2] / p[:, 2:3]).astype(np.int32)
    sil = np.zeros(mask.shape, np.uint8)
    cv2.fillPoly(sil, list(p[faces]), 1)
    sil = sil > 0
    return float((sil & mask).sum() / max((sil | mask).sum(), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(OUTPUTS, "fprecon_viser", "compare.json"))
    args = ap.parse_args()
    rows = []
    for s in ("pp", "prp"):
        for k in range(1, 6):
            vid = f"{s}_anthony_real_{k}"
            vd = os.path.join(CHOIR_ROOT, "output", vid)
            K = np.array(json.load(open(os.path.join(vd, "inputs", "intrinsics.json")))["intrinsics"], float)
            row = {"clip": vid}
            fin = os.path.join(vd, "stage3", "final", "optimized_meshes", "object")
            n = len(os.listdir(fin)) if os.path.isdir(fin) else 0
            if n:
                ious = []
                for i in (0, n - 1):
                    m = trimesh.load(os.path.join(fin, f"{i:04d}_mesh.ply"), process=False)
                    ious.append(sil_iou(np.asarray(m.vertices), m.faces, K, object_mask(vd, os.path.join(STORAGE, "runs", "fprecon", vid), i)))
                cj = json.load(open(os.path.join(OUTPUTS, "choir_viser", f"{vid}.json")))
                row["choir"] = {"iou_first_last": ious, "end_height_m": cj["object_height_above_table_first_last_m"][1]}
            fj = os.path.join(OUTPUTS, "fprecon_viser", f"{vid}.json")
            if os.path.exists(fj):
                f = json.load(open(fj))
                h = f["object_height_above_table_first_last_m"]
                row["fprecon"] = {"iou_first_last": f["iou_first_last"], "start_height_m": h[0], "end_height_m": h[1],
                                  "depth_scale_from_hand": f["depth_scale_from_hand"], "object_size_m": f["object_size_m"],
                                  "table_tilt_vs_gravity_deg": f["table_tilt_vs_gravity_deg"]}
            rows.append(row)
    json.dump(rows, open(args.out, "w"), indent=2)
    print("| clip | CHOIR IoU first/last | new IoU first/last | CHOIR end height | new start/end height | hand depth scale |")
    print("|---|---|---|---|---|---|")
    def iou(r, key):
        return f"{r[key]['iou_first_last'][0]:.2f} / {r[key]['iou_first_last'][1]:.2f}" if key in r else "—"
    for r in rows:
        c, f = r.get("choir"), r.get("fprecon")
        ch = "—" if not c else "%+.1f cm" % (100 * c["end_height_m"])
        fh = "—" if not f else "%+.1f / %+.1f cm" % (100 * f["start_height_m"], 100 * f["end_height_m"])
        sc = "—" if not f else "%.3f" % f["depth_scale_from_hand"]
        print(f"| {r['clip']} | {iou(r, 'choir')} | {iou(r, 'fprecon')} | {ch} | {fh} | {sc} |")


if __name__ == "__main__":
    main()
