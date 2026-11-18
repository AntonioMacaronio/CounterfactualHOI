#!/usr/bin/env bash
# Headless Do-as-I-Do reconstruction of one clip, with outputs on /data.
#
#   bash scripts/daid/run_recon.sh VIDEO "OBJECT TEXT" [FRAME_N=10] [HAND=right] [RUN_NAME]
#   e.g. bash scripts/daid/run_recon.sh ../outputs/minimax_h3/sources/IMG_5710.mp4 "airpods case" 10 right
#
# Runs upstream reconstruction/run_pipeline.sh unchanged except for one swap: SAM3 segments the object
# from a text prompt (OBJECT TEXT, e.g. the <CLS> of a generated clip) instead of the click GUI, which
# needs an X display. The upstream pipeline writes everything next to the video, so the clip is first
# copied into $PROJECT_STORAGE_ROOT/runs/recon/<RUN_NAME>/ (pointmaps alone are a few GB per clip).
# FRAME_N is the reference frame for the SAM3D mesh: pick one where the object is unoccluded and the
# anchor hand is visible (translation/scale is fit against the hand at this frame).
# RECON_GPU=N selects the GPU (upstream config/paths.sh hard-codes 0).
# HAND_FRAME=N prompts SAM3's hand ("right hand") at frame N instead of FRAME_N. SAM3 often drops a
# just-entered hand for ~10 frames around the prompt frame; prompting where the hand is large and clear
# (e.g. mid-manipulation) and letting it propagate back gives a hand mask at FRAME_N, which the
# translation/scale fit requires.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

VIDEO="$(realpath "${1:?usage: run_recon.sh VIDEO \"OBJECT TEXT\" [FRAME_N] [HAND] [RUN_NAME]}")"
OBJECT="${2:?object text, e.g. \"red apple\"}"
FRAME_N="${3:-10}"
HAND="${4:-right}"
RUN_NAME="${5:-$(basename "${VIDEO%.*}")__${OBJECT// /_}}"

R="$DAID_ROOT/reconstruction"
WORK="$PROJECT_STORAGE_ROOT/runs/recon/$RUN_NAME"
mkdir -p "$WORK"
# Upstream assumes <dir>/<dir>.mp4 (e.g. whisking/whisking.mp4): HaWoR writes <dir>/<video name>/ and
# optimize_translation_scale.py looks for hand meshes in <dir>/<dir name>/, so name the clip after the run.
CLIP="$WORK/$RUN_NAME.mp4"
[ -f "$CLIP" ] || cp "$VIDEO" "$CLIP"
# Upstream's frame extraction has no `ffmpeg -y`, so a rerun would stop at the overwrite prompt.
# Frames (and their pointmaps) are derived from the clip, so clear them for a fresh run.
rm -rf "$WORK/all_frames"

# Build the headless pipeline from upstream; fail loudly if upstream changed and the swap no longer applies.
PIPE="$WORK/run_pipeline_headless.sh"
sed -e "s|^HERE=.*|HERE=\"$R\"|" \
    -e 's|^source "\$HERE/config/paths.sh"$|&\nexport CUDA_VISIBLE_DEVICES="${RECON_GPU:-0}"   # paths.sh pins GPU 0|' \
    -e 's|^\(\s*\)DISPLAY="\$SAM3_DISPLAY" python run_sam3_video.py|\1python run_sam3_video.py|' \
    -e 's|^\(\s*\)--click \\$|\1--text "$OBJ_NAME" \\|' \
    -e '/--text "\$HAND_NAME"/,/--frame_idx/ s|--frame_idx "\$n"|--frame_idx "${HAND_FRAME:-$n}"|' \
    "$R/run_pipeline.sh" > "$PIPE"
grep -q -- '--text "$OBJ_NAME"' "$PIPE" && ! grep -q -- '--click' "$PIPE" && grep -q RECON_GPU "$PIPE" && grep -q HAND_FRAME "$PIPE" \
  || { echo "could not patch the click prompt in $R/run_pipeline.sh" >&2; exit 1; }

export PATH="$HOME/miniconda3/bin:$PATH"   # run_pipeline.sh calls `conda info --base` / `conda activate`
start=$(date +%s)
echo "[recon] $RUN_NAME: object='$OBJECT' frame=$FRAME_N hand=$HAND hand_prompt_frame=${HAND_FRAME:-$FRAME_N} -> $WORK"
set +e
bash "$PIPE" "$CLIP" "$FRAME_N" "$OBJECT" "$HAND" < /dev/null 2>&1 | tee "$WORK/pipeline.log"
rc=${PIPESTATUS[0]}
set -e
echo "[recon] $RUN_NAME: exit=$rc after $(( $(date +%s) - start ))s; outputs in $WORK"
exit "$rc"
