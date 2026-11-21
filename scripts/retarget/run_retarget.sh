#!/usr/bin/env bash
# Run Do-as-I-Do retargeting (MANO demo -> Sharpa hand, MuJoCo Warp) for one prepared input dir.
#
#   bash scripts/retarget/run_retarget.sh RAW_DIR TASK [HAND=right] [extra launch.py args...]
#   e.g. bash scripts/retarget/run_retarget.sh $PROJECT_STORAGE_ROOT/runs/retarget_in/IMG_5710_real IMG_5710
#
# RAW_DIR comes from prepare_daid_input.py (or the CHOIR adapter); TASK is the hand-mesh subdir name.
# launch.py only forwards --output-root-dir to stage 1 (stages 2-5 use retargeting/outputs/), so we keep the
# default root and symlink this task's three output dirs to $PROJECT_STORAGE_ROOT/runs/retarget instead.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
RAW="$(realpath "$1")"; TASK="$2"; HAND="${3:-right}"; shift $(( $# >= 3 ? 3 : 2 ))
OUT_REPO="$DAID_ROOT/retargeting/outputs"
OUT_DATA="$PROJECT_STORAGE_ROOT/runs/retarget"
for rel in "mano/$HAND/$TASK" "sharpa/$HAND/$TASK" "assets/objects/$TASK"; do
  mkdir -p "$OUT_DATA/$rel" "$(dirname "$OUT_REPO/$rel")"
  if [ -e "$OUT_REPO/$rel" ] && [ ! -L "$OUT_REPO/$rel" ]; then
    echo "$OUT_REPO/$rel exists and is not a symlink; refusing to overwrite" >&2; exit 1
  fi
  ln -sfn "$OUT_DATA/$rel" "$OUT_REPO/$rel"
done
cd "$DAID_ROOT/retargeting"
exec "$CONDA_ENVS_PATH/retargeting/bin/python" launch.py --task "$TASK" --raw-dir "$RAW" --robot-type sharpa \
  --hand-type "$HAND" --no-show-viewer --no-wait-on-finish "$@"
