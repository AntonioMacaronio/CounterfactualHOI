#!/usr/bin/env bash
# Post-process a Do-as-I-Do run: (re-)run the translation/scale optimization, reject unreliable hand
# anchors, reconstruct the table, and write QC. Run after run_recon.sh; also usable on its own when stage 4
# failed but tracking (hours) succeeded.
#
#   bash scripts/daid/finish_recon.sh RUN_DIR OBJECT_ID MESH_FRAME [HAND=right]
#
# MESH_FRAME is run_recon.sh's FRAME_N (where SAM3D built the mesh). The optimization's ref frame sets the
# mesh's metric scale from k at that frame; if k there is an outlier (hand cut off at the image border),
# it is re-run from the nearest reliable frame (see fill_unanchored.py), keeping the same mesh.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
VD="$(realpath "$1")"; OBJ="$2"; MESH_FRAME="$3"; HAND="${4:-right}"
PY="$CONDA_ENVS_PATH/sam3d/bin/python"
CV="$VD/obj_tracking_out/$OBJ/combined_visualization"
MESH="$VD/video_segmentation/masks/$(printf "frame_%06d_masks" "$MESH_FRAME")/$OBJ/$OBJ.obj"
HANDS="$(ls "$VD"/*/all_hand_meshes.npz | head -1)"   # HaWoR output dir is named after the clip file

optimize() {
  (cd "$DAID_ROOT/reconstruction/scripts" && CUDA_VISIBLE_DEVICES="${RECON_GPU:-0}" "$PY" optimize_translation_scale.py \
    --video-dir "$VD" --layout-json "$CV/layout_camera_frame.json" --anchor-hand "$HAND" --ref-frame "$1" \
    --mesh "$MESH" --hand-meshes "$HANDS")
}
optimize "$MESH_FRAME"
REF="$("$PY" "$PROJECT_ROOT/scripts/daid/fill_unanchored.py" --video-dir "$VD" --object-id "$OBJ" --suggest-ref)"
if [ "$REF" != "$MESH_FRAME" ]; then
  echo "[finish] k at frame $MESH_FRAME is an outlier; re-running the optimization with ref frame $REF"
  optimize "$REF"
fi
"$PY" "$PROJECT_ROOT/scripts/daid/fill_unanchored.py" --video-dir "$VD" --object-id "$OBJ"
"$PY" "$PROJECT_ROOT/scripts/daid/reconstruct_table.py" --video-dir "$VD" --object-id "$OBJ" --anchor-hand "$HAND"
# fps of the input clip (skip the pipeline's own output_*/tracked_* visualisation videos)
FPS="$("$PY" -c "import cv2,glob,os; v=[p for p in sorted(glob.glob('$VD/*.mp4')) if not os.path.basename(p).startswith(('output_','tracked_'))][0]; print(round(cv2.VideoCapture(v).get(cv2.CAP_PROP_FPS)))")"
"$PY" "$PROJECT_ROOT/scripts/daid/qc_recon.py" --video-dir "$VD" --object-id "$OBJ" --anchor-hand "$HAND" --fps "$FPS"
