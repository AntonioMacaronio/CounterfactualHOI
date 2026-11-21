#!/usr/bin/env bash
# Make CHOIR's Stage-1 seed mask (output/<VID>/inputs/annotations/first_mask.png) from a text prompt,
# replacing CHOIR's click-based labeling.py. Runs SAM3 (Do-as-I-Do's sam3 env) on the clip's first second
# and keeps the frame-0 object mask, plus a red overlay for checking.
#
#   bash scripts/choir/first_mask.sh VID "object text"        # expects $CHOIR_ROOT/data/VID.mp4
#   POINTS="775,830;980,950" bash scripts/choir/first_mask.sh VID "spatula"   # point prompt instead
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
VID="$1"; TEXT="$2"
SRC="$CHOIR_ROOT/data/$VID.mp4"
ANN="$CHOIR_ROOT/output/$VID/inputs/annotations"
WORK="$TMPDIR/first_mask/$VID"
rm -rf "$WORK"; mkdir -p "$WORK" "$ANN"
# POINTS="x,y;x,y" (frame-0 pixels, all positive) replaces the text prompt when text detection finds nothing.
if [ -n "${POINTS:-}" ]; then
  labels="$(echo "$POINTS" | sed 's/[^;]*/1/g')"   # one positive label per point
  PROMPT=(--points "$POINTS" --point_labels "$labels")
else
  PROMPT=(--text "$TEXT")
fi
ffmpeg -v error -y -i "$SRC" -t 1 -c:v libx264 -crf 12 -pix_fmt yuv420p "$WORK/clip.mp4"
(cd "$DAID_ROOT/reconstruction/scripts" && CUDA_VISIBLE_DEVICES="${GPU:-0}" "$CONDA_ENVS_PATH/sam3/bin/python" \
  run_sam3_video.py --video "$WORK/clip.mp4" "${PROMPT[@]}" --obj_id obj --frame_idx 0 > "$WORK/sam3.log" 2>&1)
"$CONDA_ENVS_PATH/sam3d/bin/python" - "$WORK" "$ANN" "$SRC" <<'EOF'
import sys, cv2, numpy as np
work, ann, src = sys.argv[1:]
m = cv2.imread(f"{work}/video_segmentation/masks/frame_000000_masks/obj.png", 0)
ok, frame = cv2.VideoCapture(src).read()
if m is None or (m > 127).sum() == 0:
    sys.exit(f"empty frame-0 mask (see {work}/sam3.log); try another prompt")
m = cv2.resize(m, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
cv2.imwrite(f"{ann}/first_mask.png", (m > 127).astype(np.uint8) * 255)
cv2.imwrite(f"{ann}/frame_000000.jpg", frame)
over = frame.copy(); sel = m > 127
over[sel] = (0.45 * over[sel] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
cv2.imwrite(f"{ann}/first_mask_overlay.jpg", over)
print(f"first_mask.png: {sel.sum()} px -> {ann}")
EOF
