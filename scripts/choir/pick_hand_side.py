"""Print which hand (left/right) interacts with the object in a CHOIR clip, for stage 3's --hand_side.

  python scripts/choir/pick_hand_side.py $CHOIR_ROOT/output/VID      (choir env)

Dyn-HaMR exports both hands ("dual" MANO params) whenever it sees two, and stage 3 then needs a side. Stage 1
tracks a mask per detected hand (inputs/masks/hand_{left,right}); with one mask that side is used, with two the
side whose mask (dilated by 25 px; masks are RGBA cut-outs, the alpha channel is the mask) overlaps the object mask on the most pixels summed over all frames. Frames
where a "hand" mask covers more than 6% of the image are skipped: SAM2 sometimes tracks a spurious second-hand
detection onto the whole table (pp_anthony_real_2), which then overlaps the object everywhere.
"""

import glob
import os
import sys

import cv2
import numpy as np


def read_mask(path):
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    return (m[..., 3] if m.ndim == 3 and m.shape[2] == 4 else m if m.ndim == 2 else m.max(-1)) > 127


def main():
    vd = sys.argv[1]
    sides = [s for s in ("left", "right") if os.path.isdir(os.path.join(vd, "inputs", "masks", f"hand_{s}"))]
    if len(sides) == 1:
        print(sides[0])
        return
    if not sides:
        sys.exit(f"no hand masks in {vd}/inputs/masks")
    kernel = np.ones((51, 51), np.uint8)
    score = dict.fromkeys(sides, 0)
    for om in glob.glob(os.path.join(vd, "inputs", "masks", "object", "*.png")):
        obj = read_mask(om)
        for s in sides:
            hm = read_mask(os.path.join(vd, "inputs", "masks", f"hand_{s}", os.path.basename(om)))
            if hm is None or hm.mean() > 0.06:
                continue
            score[s] += int((cv2.dilate(hm.astype(np.uint8), kernel) > 0)[obj].sum())
    print(f"hand-object overlap: {score}", file=sys.stderr)
    print(max(score, key=score.get))


if __name__ == "__main__":
    main()
