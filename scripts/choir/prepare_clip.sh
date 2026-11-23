#!/usr/bin/env bash
# Bring one recorded clip into the pipeline: $CHOIR_ROOT/data/VID.mp4 (30 fps, 1920x1080, H.264, no audio; the
# input of first_mask.sh / run_choir.sh) and its GeoCalib gravity, runs/choir_gravity/VID/gravity.json (camera-frame
# up vector; the camera is static), which the viewers, fprecon, the refinement and retargeting all use as "up".
#
#   bash scripts/choir/prepare_clip.sh SRC.MOV VID [GPU=0]
#   e.g. bash scripts/choir/prepare_clip.sh ../realvideos/pickplace/pp_anthony_real_1.MOV pp_anthony_real_1
#
# Gravity: Do-as-I-Do's predict_video_gravity.py (sam3d env, which has GeoCalib) on every 5th frame. Each part is
# skipped when its output exists.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
SRC="$1"; VID="$2"; GPU="${3:-0}"
FF="$CONDA_ENVS_PATH/sam3d/bin/ffmpeg"
OUT="$CHOIR_ROOT/data/$VID.mp4"
G="$PROJECT_STORAGE_ROOT/runs/choir_gravity/$VID"
mkdir -p "$CHOIR_ROOT/data" "$G/frames"
[ -f "$OUT" ] || "$FF" -v error -y -i "$SRC" -an -vf "fps=30,scale=1920:1080" -c:v libx264 -crf 16 -preset medium \
  -pix_fmt yuv420p "$OUT"
[ -n "$(ls "$G/frames")" ] || "$FF" -v error -i "$OUT" -vf "select=not(mod(n\,5))" -fps_mode passthrough \
  -start_number 0 "$G/frames/%06d.png"
if [ ! -f "$G/gravity.json" ]; then
  (cd "$DAID_ROOT/reconstruction/scripts" && CUDA_VISIBLE_DEVICES="$GPU" "$CONDA_ENVS_PATH/sam3d/bin/python" \
    predict_video_gravity.py "$G/frames" --output_path "$G/gravity.json" > "$G/gravity.log" 2>&1) \
    || { echo "$VID: gravity FAILED (see $G/gravity.log)" >&2; exit 1; }
fi
python3 -c "import json; g = json.load(open('$G/gravity.json')); print('$VID: roll %+.1f pitch %+.1f (%d/%d frames)' % (g['roll_deg'], g['pitch_deg'], g['n_inliers'], g['n_frames']))"
