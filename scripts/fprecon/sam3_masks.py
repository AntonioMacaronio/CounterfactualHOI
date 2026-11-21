"""Per-frame SAM3 (image model, text prompts) object and hand masks for one clip, replacing CHOIR's SAM2 tracks.

  python scripts/fprecon/sam3_masks.py --video-dir $CHOIR_ROOT/output/VID --out $PROJECT_STORAGE_ROOT/runs/fprecon/VID \
      --object-prompt "mug"          (sam3 env)

CHOIR's stage-1 SAM2 tracks drift: a spurious "hand" on the whole table (pp_2), a hand mask that swallows the held
mug (prp_4), an object mask that loses the camera's black front panel (pp_1, prp_1). SAM3 segments every frame from
the text prompt(s) instead, so errors do not accumulate; several prompts can be pooled (the RealSense camera is a
"camera" lying down but only a "device" standing on its end). Instance choice per frame:
  object: the instance overlapping the last seen object mask most (dilated by 40 px, +20 px per frame unseen;
          frame 0: CHOIR's inputs/annotations/first_mask.png), ignoring other matches such as the hot-sauce bottle
          behind the soap; with none there, a detection scoring >= 0.6 anywhere (the object was set down elsewhere
          while hidden); else (hidden in the hand) an empty mask;
  hand:   the instance containing most of the projected Dyn-HaMR hand of the chosen side (its pixels inside the
          instance), else the one overlapping the previous hand mask; pixels of the chosen object are removed.
Writes OUT/masks_sam3/{object,hand}/<i>.png (0/255) and OUT/masks_sam3/masks.json (per-frame scores and areas).
"""

import argparse
import glob
import json
import os

import cv2
import numpy as np
import PIL.Image as Image
import torch
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


def hand_mesh_dir(vd, side):     # as hand_scale.hand_mesh_dir (the sam3 env has no trimesh to import that module)
    d = os.path.join(vd, "stage1", "hand", "hand_meshes")
    return os.path.join(d, side) if os.path.isdir(os.path.join(d, side)) else d


def obj_vertices(path):
    return np.array([[float(x) for x in ln.split()[1:4]] for ln in open(path) if ln.startswith("v ")])


def instances(proc, state, prompt):
    state = proc.set_text_prompt(prompt, state)
    if len(state["scores"]) == 0:
        return [], []
    return list(state["masks"][:, 0].cpu().numpy()), list(state["scores"].float().cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--object-prompt", required=True, help="comma-separated text prompts; candidates are pooled")
    ap.add_argument("--hand-prompt", default="hand")
    ap.add_argument("--side", required=True, help="left/right (scripts/choir/pick_hand_side.py)")
    ap.add_argument("--threshold", type=float, default=0.3)
    args = ap.parse_args()
    vd, out = args.video_dir, args.out
    side = args.side
    K = np.array(json.load(open(os.path.join(vd, "inputs", "intrinsics.json")))["intrinsics"], float)
    frames = sorted(glob.glob(os.path.join(vd, "inputs", "frames", "rgb", "*.png")), key=lambda p: int(os.path.basename(p)[:-4]))
    od, hd = os.path.join(out, "masks_sam3", "object"), os.path.join(out, "masks_sam3", "hand")
    os.makedirs(od, exist_ok=True)
    os.makedirs(hd, exist_ok=True)
    proc = Sam3Processor(build_sam3_image_model(), confidence_threshold=args.threshold)
    ref_obj = cv2.imread(os.path.join(vd, "inputs", "annotations", "first_mask.png"), 0) > 127
    ref_hand = None
    missed = 0
    log = []
    for i, path in enumerate(frames):
        img = np.asarray(Image.open(path).convert("RGB"))
        H, W = img.shape[:2]
        state = proc.set_image(Image.fromarray(img))

        ms, ss = [], []
        for prompt in args.object_prompt.split(","):     # pool candidates over the prompts
            m_, s_ = instances(proc, state, prompt.strip())
            ms, ss = ms + m_, ss + s_
        r = min(40 + 20 * missed, 400)              # search wider the longer the object has been unseen
        near = cv2.dilate(ref_obj.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
        ov = [float((m & near).sum()) / max(m.sum(), 1) for m in ms]
        obj, oscore = np.zeros((H, W), bool), None
        k = None
        if ms and max(ov) > 0.3:
            k = int(np.argmax([o + 0.1 * s for o, s in zip(ov, ss)]))
        elif ms and max(ss) >= 0.6:                 # re-acquire a confident detection elsewhere (e.g. set down)
            k = int(np.argmax(ss))
        if k is not None:
            obj, oscore, ref_obj, missed = ms[k], float(ss[k]), ms[k], 0
        else:
            missed += 1

        ms, ss = instances(proc, state, args.hand_prompt)
        hf = os.path.join(hand_mesh_dir(vd, side), f"hand_{i:05d}.obj")
        hand, hscore, how = np.zeros((H, W), bool), None, None
        if ms:
            if os.path.exists(hf):
                v = obj_vertices(hf)
                p = (K @ v.T).T
                uv = np.round(p[:, :2] / p[:, 2:3]).astype(int)
                uv = uv[(uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)]
                inside = [float(m[uv[:, 1], uv[:, 0]].mean()) if len(uv) else 0.0 for m in ms]
                k, how = int(np.argmax(inside)), "dynhamr"
                ok = inside[k] > 0.2
            elif ref_hand is not None:
                iou = [float((m & ref_hand).sum()) / max((m | ref_hand).sum(), 1) for m in ms]
                k, how = int(np.argmax(iou)), "previous"
                ok = iou[k] > 0.2
            else:
                ok = False
            if ok:
                hand, hscore, ref_hand = ms[k] & ~obj, float(ss[k]), ms[k]
        cv2.imwrite(os.path.join(od, f"{i}.png"), obj.astype(np.uint8) * 255)
        cv2.imwrite(os.path.join(hd, f"{i}.png"), hand.astype(np.uint8) * 255)
        log.append({"object_score": oscore, "object_area": float(obj.mean()), "hand_score": hscore,
                    "hand_area": float(hand.mean()), "hand_by": how})
    json.dump({"object_prompt": args.object_prompt, "hand_prompt": args.hand_prompt, "side": side, "per_frame": log},
              open(os.path.join(out, "masks_sam3", "masks.json"), "w"), indent=1)
    no_obj = sum(f["object_score"] is None for f in log)
    no_hand = sum(f["hand_score"] is None for f in log)
    print(f"{os.path.basename(vd)}: SAM3 '{args.object_prompt}' / '{args.hand_prompt}' ({side}) on {len(log)} frames; "
          f"object missing on {no_obj}, hand missing on {no_hand}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
