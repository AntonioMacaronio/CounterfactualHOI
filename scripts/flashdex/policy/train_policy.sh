#!/bin/bash
# The §1.4 policy (scripts/flashdex/policy/cfhoi_terms.py): FDR's env and FlashSAC learner with the §1.4 actor
# observations and the AdaDex hybrid reward, on a dataset with GenEnv clips (scripts/flashdex/make_gen_clips.py).
#
#   bash scripts/flashdex/policy/train_policy.sh MOTION RUN_NAME [hydra overrides ...]
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
MOTION="${1:?MOTION}"; NAME="${2:?RUN_NAME}"; shift 2
exec bash "$HERE/../train.sh" "$MOTION" "$NAME" envs/obs@obs=cfhoi_policy envs/rewards@rewards=cfhoi_policy \
    runner.agent.traj_encoder.enable=false share_actor_critic_obs=false \
    "commands.object.support_disks.radii=[0.08,0.15,0.25,0.5]" "$@"     # asymmetric: §1.4 actor, privileged critic
