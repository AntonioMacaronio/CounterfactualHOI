#!/usr/bin/env bash
# After a refinement round (every run_refine.sh of REFINE_TAG=TAG has finished): the automatic-fit baseline
# (runs/refine_TAG_auto: each clip's v001, the search tools run over the whole clip before the agent), the 3-vote
# reviews (review.py), the summaries (summarize.py), the blind A/B comparisons of the round against the initial
# reconstruction, the automatic fit and PREV_TAG (compare.py --tag finalN; the site reads runs/compare/*_finalN),
# and the refine_viser_TAG index page (its viewers come from run_refine.sh).
#
#   bash scripts/refine/post_round.sh TAG [PREV_TAG] [VID ...]      e.g. bash scripts/refine/post_round.sh r8 r7
#
# Default clips: every runs/refine_TAG/VID with a report.md. Reviews are split over NGPU GPUs (default: all visible).
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd "$PROJECT_ROOT"
TAG="$1"; PREV="${2:-}"; shift $(( $# >= 2 ? 2 : 1 ))
R="$PROJECT_STORAGE_ROOT/runs"; L="$R/logs"; PY="$CONDA_ENVS_PATH/fpose/bin/python"
NGPU="${NGPU:-$(nvidia-smi -L | wc -l)}"; CT="final${TAG#r}"
mkdir -p "$L"
V=("$@")
if [ ${#V[@]} -eq 0 ]; then
  for d in "$R/refine_$TAG"/*/; do [ -f "$d/report.md" ] && V+=("$(basename "$d")"); done
fi
[ ${#V[@]} -gt 0 ] || { echo "no finished runs in $R/refine_$TAG" >&2; exit 1; }
for v in "${V[@]}"; do
  mkdir -p "$R/refine_${TAG}_auto/$v"
  cp "$R/refine_$TAG/$v/versions/v001.npz" "$R/refine_${TAG}_auto/$v/final.npz"
  cp "$R/refine_$TAG/$v/phases.json" "$R/refine_${TAG}_auto/$v/"
done
pids=()
for ((g = 0; g < NGPU && g < ${#V[@]}; g++)); do           # clips round-robin over the GPUs
  grp=(); for ((i = g; i < ${#V[@]}; i += NGPU)); do grp+=("${V[$i]}"); done
  REFINE_TAG="$TAG" CUDA_VISIBLE_DEVICES=$g "$PY" -u scripts/refine/review.py --samples 3 "${grp[@]}" \
    > "$L/review_${TAG}_g$g.log" 2>&1 &
  pids+=($!)
done
wait "${pids[@]}"; echo "[post] reviews done"
REFINE_TAG="$TAG" CUDA_VISIBLE_DEVICES=0 "$PY" -u scripts/refine/summarize.py "${V[@]}" > "$L/summarize_$TAG.log" 2>&1
echo "[post] summaries done"
pids=(); g=0
for a in initial "refine_${TAG}_auto" ${PREV:+"refine_$PREV"}; do
  CUDA_VISIBLE_DEVICES=$((g % NGPU)) "$PY" -u scripts/refine/compare.py --a "$a" --b "refine_$TAG" --tag "$CT" "${V[@]}" \
    > "$L/compare_${a}_vs_refine_${TAG}_$CT.log" 2>&1 &
  pids+=($!); g=$((g + 1))
done
REFINE_TAG="$TAG" "$CONDA_ENVS_PATH/sam3d/bin/python" scripts/choir/build_choir_html.py --method refine --index-only \
  > "$L/build_refine_viser_$TAG.log" 2>&1
wait "${pids[@]}"; echo "[post] comparisons done"
grep -h "^overall" "$L"/compare_*_vs_refine_"${TAG}_$CT".log
