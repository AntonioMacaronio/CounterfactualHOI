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
# Smoothness (the user, 2026-11-24: "add a very small smoothness reward ... the hand is quite jittery"): FDR defines
# an action_rate term (the summed squared change of the policy's action between control steps) but gives it no
# weight, and the finger actions are not low-pass filtered. eval/action_rate_l2 of cfhoi_r7_flat grew from 0.04 at
# 10M steps to 2.2-2.8 from 210M on, about 0.3 per action dimension per step. At that jitter -0.02 costs about 2%
# of the per-step reward, and the cost falls quadratically as the motion smooths. CFHOI_ACTION_RATE=0 turns it
# off; a later rewards.reward_weights.action_rate=... override on the command line wins. Launches that select
# their own reward config (policy/train_policy.sh: envs/rewards@rewards=cfhoi_policy) keep theirs unless
# CFHOI_ACTION_RATE is set.
case " $* " in
  *"envs/rewards@rewards="*) AR="${CFHOI_ACTION_RATE:-0}" ;;
  *) AR="${CFHOI_ACTION_RATE:--0.02}" ;;
esac
SMOOTH=(); [ "$AR" != "0" ] && SMOOTH+=("++rewards.reward_weights.action_rate=$AR")
exec python "$HERE/train_cfhoi.py" envs/robot@robot=sharpa_bimanual commands.motion_file="$MOTION" \
    logger.wandb.run_name="$NAME" "logger.wandb.tags=[cfhoi,sharpa]" logger.wandb.mode=offline \
    per_hand_actor_critic=false "${SMOOTH[@]}" "$@"          # right hand only: one actor-critic (the per-hand split needs both)
