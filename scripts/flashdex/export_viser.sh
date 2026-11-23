#!/bin/bash
# export_viser.py in the flashdex env, with FDR's viser (third_party/viewer_site: the Isaac env pins websockets 12).
#   bash scripts/flashdex/export_viser.sh --dataset DATA --archive RUN/evals/step<N> [--out OUT] [--clips VID ...]
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FDR="${FDR:-/data/sky/projects/CounterfactualHOI/build/FlashDexRetarget}"
source "$FDR/scripts/env.sh"
export FDR MUJOCO_GL=disabled
PYTHONPATH="$FDR/third_party/viewer_site${PYTHONPATH:+:$PYTHONPATH}" exec python "$HERE/export_viser.py" "$@"
