#!/bin/bash
# One refine round -> the two FDR datasets: TrackEnv (the 10 clips) and GenEnv (+ warped copies), retargeted to Sharpa.
#
#   bash scripts/flashdex/prepare_dataset.sh REFINE [OBJECTS_FROM]
#   e.g. bash scripts/flashdex/prepare_dataset.sh refine_r7 data_r5
#
# -> $PROJECT_STORAGE_ROOT/runs/flashdex/data_<rN>/motion_bimanual.pt and data_<rN>_gen/motion_bimanual.pt. OBJECTS_FROM: a dataset
# whose objects/ (convex parts, USD, SDF) are copied first, so the slow coacd step is skipped: the object meshes are
# the fprecon run's, the same in every refine round.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REFINE="${1:?REFINE}"; FROM="${2:-}"
ST=/data/sky/projects/CounterfactualHOI
FDR="$ST/build/FlashDexRetarget"
D="$ST/runs/flashdex/data_${REFINE#refine_}"
PY="$ST/conda_envs/flashdex/bin/python"
mkdir -p "$D"
if [ -n "$FROM" ] && [ ! -e "$D/objects" ]; then cp -a "$ST/runs/flashdex/$FROM/objects" "$D/objects"; fi
cd "$FDR"
bash scripts/retarget/import_human_demo.sh source=cfhoi raw="$ST" dataset="$D" sides=right workers=10 source.refine="$REFINE"
"$PY" -I "$HERE/add_table.py" "$D"
bash scripts/retarget/kinematic_retargeting.sh dataset="$D" robot=sharpa desk.mode=keep
"$PY" -I "$HERE/make_gen_clips.py" "$D" "${D}_gen" --variants 2 --seed 0
bash scripts/retarget/kinematic_retargeting.sh dataset="${D}_gen" robot=sharpa desk.mode=keep
"$PY" -I "$HERE/pad_bimanual.py" "$D"                        # train on motion_bimanual.pt (the bimanual robot)
"$PY" -I "$HERE/pad_bimanual.py" "${D}_gen"
echo "[prepare] done: $D/motion_bimanual.pt ${D}_gen/motion_bimanual.pt"
