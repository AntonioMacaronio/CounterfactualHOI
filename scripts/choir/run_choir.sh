#!/usr/bin/env bash
# Run Open-CHOIR end to end on one clip ($CHOIR_ROOT/data/VID.mp4, seeded by first_mask.sh).
#
#   bash scripts/choir/run_choir.sh VID [GPU=0]
#   CHOIR_STAGE1_ONLY=1 bash scripts/choir/run_choir.sh VID [GPU]   # steps 1-3 only (all that fprecon needs)
#
# Steps (each in its env, skipped when its outputs exist, logs in runs/logs/choir/VID/):
#   1 stage1 masks     Yolov8/run_with_manual_object_mask.py      (choir)
#   2 stage1 object    sam-3d-objects/run_reconstruction.py      (choir)  -> also writes inputs/intrinsics.json
#   3 stage1 hand      Dyn-HaMR run_mano_sequence.py (+ VIPE)     (dynhamr, VIPE in $CHOIR_VIPE_ENV)
#   4 stage2+3 HOI     diffusion-vas/run_temporal_optimization.py (choir)
# Dyn-HaMR sets CUDA_VISIBLE_DEVICES itself from --gpus, so the physical GPU id is passed there; it renders with
# pyrender, on EGL here (setup.sh workdir lets PYOPENGL_PLATFORM override its hard-coded OSMesa).
# Its HaMeR/ViTPose checkpoints predate torch 2.6's weights_only=True default (pickled configs), hence the override.
# Stage 3's GraspFlowMatching torchrun gets a per-GPU port (CHOIR_GFM_MASTER_PORT, patched in by setup.sh workdir).
# VIPE runs offline (HF_HUB_OFFLINE) with caches inside its source dir; the first run here warms them online.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
VID="$1"; GPU="${2:-0}"
C="$CHOIR_ROOT"; O="$C/output/$VID"
LOGD="$PROJECT_STORAGE_ROOT/runs/logs/choir/$VID"; mkdir -p "$LOGD"
export CHOIR_VIPE_ENV="$CONDA_ENVS_PATH/vipe"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
[ -f "$O/inputs/annotations/first_mask.png" ] || { echo "missing first_mask.png; run first_mask.sh" >&2; exit 1; }
exec 8> "$O/.run_choir.lock"    # one run per clip: several GPU queues may list the same clip
flock -n 8 || { echo "[choir] $VID: already being processed by another run_choir.sh, skipping"; exit 0; }

step() {  # step NAME DONE_TEST ENV DIR CMD...
  local name="$1" done_test="$2" env="$3" dir="$4"; shift 4
  if eval "$done_test"; then echo "[choir] $VID $name: done, skipping"; return; fi
  local t0; t0=$(date +%s)
  echo "[choir] $VID $name: running on GPU $GPU"
  ( set +u; conda activate "$CONDA_ENVS_PATH/$env" && export PATH="$CONDA_ENVS_PATH/$env/bin:$PATH" && cd "$dir" && "$@" ) > "$LOGD/$name.log" 2>&1 \
    || { echo "[choir] $VID $name: FAILED after $(( $(date +%s) - t0 ))s (see $LOGD/$name.log)"; tail -5 "$LOGD/$name.log" >&2; exit 1; }
  eval "$done_test" || { echo "[choir] $VID $name: finished but outputs missing (see $LOGD/$name.log)"; exit 1; }
  echo "[choir] $VID $name: ok in $(( $(date +%s) - t0 ))s"
}

step 1_masks "[ -d $O/inputs/masks/object ] && [ -f $O/inputs/video.mp4 ]" choir "$C/stage1_preprocess/Yolov8" \
  env CUDA_VISIBLE_DEVICES="$GPU" python run_with_manual_object_mask.py --video_id "$VID" \
  --data_dir "$C/data" --output_dir "$C/output" --num_workers 1

step 2_object "[ -f $O/stage1/object_init/glb_0.glb ] && [ -f $O/inputs/intrinsics.json ]" choir "$C/stage1_preprocess/sam-3d-objects" \
  env CUDA_VISIBLE_DEVICES="$GPU" python run_reconstruction.py --video_id "$VID" --data_dir "$C/output" --output_dir "$C/output"

V="$C/stage1_preprocess/Dyn_HaMR_new/third-party/vipe"
exec 9> "$V/.warmup.lock"; flock 9    # parallel clips: only one warms the caches
if [ -z "$(ls -A "$V/torch_cache" 2>/dev/null)" ]; then
  echo "[choir] warming VIPE caches online (one-time) with $VID"
  ( set +u; conda activate "$CHOIR_VIPE_ENV" && export PATH="$CHOIR_VIPE_ENV/bin:$PATH" && cd "$V" && CUDA_VISIBLE_DEVICES="$GPU" TORCH_HOME="$V/torch_cache" \
    HF_HOME="$V/hf_cache" vipe infer "$O/inputs/video.mp4" --output "$TMPDIR/vipe_warmup_$VID" ) \
    > "$LOGD/vipe_warmup.log" 2>&1 || { echo "VIPE warm-up failed (see $LOGD/vipe_warmup.log)"; exit 1; }
fi
flock -u 9

step 3_hand "[ -d $O/stage1/hand/mano_params ] && [ -n \"\$(ls -A $O/stage1/hand/mano_params 2>/dev/null)\" ]" dynhamr \
  "$C/stage1_preprocess/Dyn_HaMR_new/dyn-hamr" env PYOPENGL_PLATFORM=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python run_mano_sequence.py --video_id "$VID" --video_dir "$C/output" --gpus "$GPU"

# Stage 1 (steps 1-3) is all scripts/fprecon/run_fprecon.sh needs; step 4 is CHOIR's own HOI result (the baseline).
if [ -n "${CHOIR_STAGE1_ONLY:-}" ]; then echo "[choir] $VID: stage 1 done (CHOIR_STAGE1_ONLY: skipping step 4)"; exit 0; fi

# Dyn-HaMR exports both hands when it sees two ("dual"); stage 3 then needs the interacting side (HAND_SIDE overrides).
SIDE="${HAND_SIDE:-$("$CONDA_ENVS_PATH/choir/bin/python" "$(dirname "${BASH_SOURCE[0]}")/pick_hand_side.py" "$O" 2>> "$LOGD/hand_side.log")}"
echo "[choir] $VID hand side: $SIDE"
step 4_hoi "[ -d $O/stage3/final/optimized_meshes/object ] && [ -n \"\$(ls -A $O/stage3/final/optimized_meshes/object 2>/dev/null)\" ]" choir \
  "$C/stage3_temporal_optimization/diffusion-vas" env CUDA_VISIBLE_DEVICES="$GPU" CHOIR_GFM_MASTER_PORT=$((29600 + GPU)) \
  python run_temporal_optimization.py \
  --video_id "$VID" --data_path "$C/output" --data_output_path "$C/output" --hand_side "$SIDE"
echo "[choir] $VID: all stages done"
