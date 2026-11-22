#!/usr/bin/env bash
# Agentic refinement of one clip's FoundationPose + Depth Pro 2 reconstruction (scripts/refine/agent.py, Claude on
# Amazon Bedrock), then its offline viewer.
#
#   bash scripts/refine/run_refine.sh VID [GPU=0] [extra agent.py args, e.g. --effort xhigh]
#   REFINE_TAG=r2 bash scripts/refine/run_refine.sh ...   (writes runs/refine_r2/VID and outputs/refine_viser_r2/ instead)
#
# Needs the clip's fprecon run ($PROJECT_STORAGE_ROOT/runs/fprecon/VID, scripts/fprecon/run_fprecon.sh). Writes
# $PROJECT_STORAGE_ROOT/runs/refine/VID (report.md, transcript.jsonl, images/, versions/, final.npz, export/) and
# ../outputs/refine_viser/VID.html (+ .json). AWS credentials and region come from the environment (AWS_PROFILE etc.).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
VID="$1"; GPU="${2:-0}"; shift $(( $# >= 2 ? 2 : 1 ))
S="$(dirname "${BASH_SOURCE[0]}")"
VD="$CHOIR_ROOT/output/$VID"; RUN="$PROJECT_STORAGE_ROOT/runs/fprecon/$VID"; T="${REFINE_TAG:+_$REFINE_TAG}"
O="$PROJECT_STORAGE_ROOT/runs/refine$T/$VID"
HTML="$(dirname "$PROJECT_ROOT")/outputs/refine_viser$T/$VID.html"
mkdir -p "$O" "$(dirname "$HTML")"
export CUDA_VISIBLE_DEVICES="$GPU" TORCH_HOME HF_HOME
"$CONDA_ENVS_PATH/fpose/bin/python" -u "$S/agent.py" --video-dir "$VD" --run-dir "$RUN" --out "$O" "$@"
"$CONDA_ENVS_PATH/sam3d/bin/python" "$S/../fprecon/view_fprecon.py" --video-dir "$VD" --run-dir "$O/export" \
  --export-html "$HTML" > "$O/viewer.log" 2>&1 && tail -1 "$O/viewer.log"
