#!/bin/bash
# view_embodiment.py in the flashdex env with FDR's viser (the one scripts/flashdex/export_viser.sh uses; its client
# embeds a recording into one HTML file). CPU only.
#   bash scripts/embodiment/view_embodiment.sh --arm vega_1p [--clips VID ...] [--ik-dir DIR] [--out OUT]
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../env.sh"
source "$FDR/scripts/env.sh"
export FDR PROJECT_STORAGE_ROOT MUJOCO_GL=disabled CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONPATH="$FDR/third_party/viewer_site${PYTHONPATH:+:$PYTHONPATH}" exec python "$HERE/view_embodiment.py" "$@"
