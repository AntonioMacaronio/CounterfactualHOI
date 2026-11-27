#!/usr/bin/env bash
# FlashDexRetarget training on MuJoCo Warp (scripts/mjwarp/train.py), the twin of scripts/flashdex/train.sh.
#
#   bash scripts/mjwarp/train.sh MOTION RUN_NAME [--trials DATASET/trials.json] [hydra overrides ...]
#   e.g. bash scripts/mjwarp/train.sh $D/motion_bimanual.pt cfhoi_r10_mjw --trials $D/trials.json num_envs=4096
#
# Same launch defaults as the Isaac launcher: Sharpa bimanual robot, right hand only (per_hand_actor_critic=false),
# wandb offline, the two small smoothness rewards (CFHOI_ACTION_RATE, CFHOI_OBJ_SMOOTH; see scripts/flashdex/train.sh).
# Logs and checkpoints: $FDR/logs/flashsac/<RUN_NAME>-<MMDD-HHMMSS>/. Needs scripts/mjwarp/setup.sh.
set -e
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOTION="$(realpath "${1:?MOTION}")"; NAME="${2:?RUN_NAME}"; shift 2
TRIALS=()
if [ "${1:-}" = "--trials" ]; then TRIALS=(--trials "$(realpath "$2")"); shift 2; fi
export MUJOCO_GL=disabled WANDB_MODE=offline WANDB_DIR="$PROJECT_STORAGE_ROOT/runs/wandb"
export WARP_CACHE_PATH="$SHARED_CACHE_ROOT/warp" TORCHINDUCTOR_CACHE_DIR="$PROJECT_STORAGE_ROOT/cache/torchinductor"
AR="${CFHOI_ACTION_RATE:--0.02}"; OS="${CFHOI_OBJ_SMOOTH:-0.1}"
SMOOTH=(); [ "$AR" != "0" ] && SMOOTH+=("++rewards.reward_weights.action_rate=$AR")
[ "$OS" != "0" ] && SMOOTH+=("++rewards.reward_weights.tracking_obj_lin_vel_error_exp=$OS" "++rewards.reward_scales.tracking_obj_lin_vel_error_exp=10.0" \
                             "++rewards.reward_weights.tracking_obj_ang_vel_error_exp=$OS" "++rewards.reward_scales.tracking_obj_ang_vel_error_exp=1.0")
exec "$PROJECT_STORAGE_ROOT/venvs/fdr-mjwarp/bin/python" "$HERE/train.py" "${TRIALS[@]}" envs/robot@robot=sharpa_bimanual \
    commands.motion_file="$MOTION" logger.wandb.run_name="$NAME" "logger.wandb.tags=[cfhoi,sharpa,mjwarp]" \
    logger.wandb.mode=offline per_hand_actor_critic=false "${SMOOTH[@]}" "$@"
