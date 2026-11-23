#!/bin/bash
# FlashDexRetarget training on our clips, Sharpa Wave (scripts/flashdex/train_cfhoi.py: FDR's train.py + every
# clip's eval rollout archived + metrics.jsonl).
#
#   bash scripts/flashdex/train.sh MOTION RUN_NAME [hydra overrides ...]
#   e.g. bash scripts/flashdex/train.sh $PROJECT_STORAGE_ROOT/runs/flashdex/data_r5/motion_bimanual.pt cfhoi_r5
# (motion_bimanual.pt: scripts/flashdex/pad_bimanual.py)
#
# Logs and checkpoints: $FDR/logs/flashsac/<RUN_NAME>-<MMDD-HHMMSS>/ (FDR's layout; on /data).
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FDR="${FDR:-/data/sky/projects/CounterfactualHOI/build/FlashDexRetarget}"
MOTION="$(realpath "${1:?MOTION}")"; NAME="${2:?RUN_NAME}"; shift 2
cd "$FDR"
source scripts/env.sh
unset HEADLESS FDR_ROBOT_USD          # FDR_ROBOT_USD in env.sh is the XHAND USD
export FDR
exec python "$HERE/train_cfhoi.py" envs/robot@robot=sharpa_bimanual commands.motion_file="$MOTION" \
    logger.wandb.run_name="$NAME" "logger.wandb.tags=[cfhoi,sharpa]" logger.wandb.mode=offline \
    per_hand_actor_critic=false "$@"          # right hand only: one actor-critic (the per-hand split needs both)
