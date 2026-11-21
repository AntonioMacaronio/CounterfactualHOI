#!/usr/bin/env bash
# FoundationPose + Depth Pro 2 hand-object reconstruction of one clip, on top of CHOIR's stage 1
# ($CHOIR_ROOT/output/VID: frames, intrinsics, first-frame object mask, Dyn-HaMR hands; run_choir.sh steps 1-3).
#
#   bash scripts/fprecon/run_fprecon.sh VID [GPU=0]      OBJECT_PROMPT="mug,cup" overrides the prompt table below
#
# Steps (each skipped when its output exists; outputs in $PROJECT_STORAGE_ROOT/runs/fprecon/VID):
#   0 masks     sam3_masks.py    per-frame SAM3 object + hand masks (text prompts)   -> masks_sam3/     (sam3 env)
#   0 mesh      sam3d_mesh.sh    textured SAM3D mesh from frame 0 + its SAM3 mask     -> sam3d_tex/      (sam3d env)
#   1 depth     run_depth.py     Depth Pro 2 H+ video depth, CHOIR's focal          -> depth_dp2.npy   (depthpro2 venv)
#   2 scale     hand_scale.py    metric scale from the Dyn-HaMR MANO hand            -> hand_scale.json (sam3d env)
#   3 object    run_fpose.py     SAM3D mesh sizing + FoundationPose tracking         -> object_poses.npy (fpose env)
#   4 verify    verify_track.py  per-frame coverage / spill check vs the masks       -> verify.json     (fpose env)
#   5 overlay   overlay.py       QC image (12 frames)                                -> overlay.jpg     (fpose env)
#   6 viewer    view_fprecon.py  offline viser HTML -> ../outputs/fprecon_viser/VID.html (sam3d env)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
VID="$1"; GPU="${2:-0}"
VD="$CHOIR_ROOT/output/$VID"; O="$PROJECT_STORAGE_ROOT/runs/fprecon/$VID"
HTML="$(dirname "$PROJECT_ROOT")/outputs/fprecon_viser/$VID.html"
S="$(dirname "${BASH_SOURCE[0]}")"
mkdir -p "$O" "$(dirname "$HTML")"
export TORCH_HOME HF_HOME CUDA_VISIBLE_DEVICES="$GPU"
[ -d "$VD/stage1/hand/hand_meshes" ] || { echo "$VID: no CHOIR stage 1 (run run_choir.sh first)" >&2; exit 1; }

run() {  # run NAME OUTPUT CMD...
  local name="$1" output="$2"; shift 2
  if [ -e "$output" ]; then echo "[fprecon] $VID $name: done, skipping"; return; fi
  local t0; t0=$(date +%s)
  "$@" > "$O/$name.log" 2>&1 || { echo "[fprecon] $VID $name: FAILED (see $O/$name.log)"; tail -3 "$O/$name.log" >&2; exit 1; }
  echo "[fprecon] $VID $name: ok in $(( $(date +%s) - t0 ))s"
}

ENVF="$(dirname "$PROJECT_ROOT")/.env"                 # HF token for the gated SAM3 / SAM3D weights
if [ -f "$ENVF" ]; then set -a; source "$ENVF"; set +a; fi
case "${VID##*_}" in   # real-video objects (antzhang-notes/project.md); prompts are pooled per frame
  1) P="camera,device";; 2) P="tape roll";; 3) P="black spatula,spatula";; 4) P="mug,cup";; 5) P="dish soap bottle";; *) P="";;
esac
P="${OBJECT_PROMPT:-$P}"
[ -n "$P" ] || { echo "$VID: set OBJECT_PROMPT" >&2; exit 1; }
SIDE="$("$CONDA_ENVS_PATH/sam3d/bin/python" "$S/../choir/pick_hand_side.py" "$VD" 2>/dev/null)"
run masks "$O/masks_sam3/masks.json" "$CONDA_ENVS_PATH/sam3/bin/python" "$S/sam3_masks.py" --video-dir "$VD" --out "$O" \
  --object-prompt "$P" --side "$SIDE"
run mesh "$O/sam3d_tex/masks/object/object.obj" bash "$S/sam3d_mesh.sh" "$VID" "$GPU"
run depth "$O/depth_dp2.json" "$PROJECT_STORAGE_ROOT/venvs/depthpro2/bin/python" "$S/run_depth.py" --video-dir "$VD" --out "$O"
run scale "$O/hand_scale.json" "$CONDA_ENVS_PATH/sam3d/bin/python" "$S/hand_scale.py" --video-dir "$VD" --out "$O"
run object "$O/fpose_qc.json" "$CONDA_ENVS_PATH/fpose/bin/python" "$S/run_fpose.py" --video-dir "$VD" --out "$O"
run verify "$O/verify.json" "$CONDA_ENVS_PATH/fpose/bin/python" "$S/verify_track.py" --video-dir "$VD" --out "$O"
run overlay "$O/overlay.jpg" "$CONDA_ENVS_PATH/fpose/bin/python" "$S/overlay.py" --video-dir "$VD" --out "$O" --frames 12
run viewer "$HTML" "$CONDA_ENVS_PATH/sam3d/bin/python" "$S/view_fprecon.py" --video-dir "$VD" --run-dir "$O" --export-html "$HTML"
grep -h "^$VID\|^wrote" "$O/masks.log" "$O/scale.log" "$O/object.log" "$O/verify.log" "$O/viewer.log" 2>/dev/null | cut -c1-260 || true
