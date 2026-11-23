#!/usr/bin/env bash
# Link our import source (cfhoi) into the FlashDexRetarget clone (after scripts/flashdex/setup.sh).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FDR="$PROJECT_STORAGE_ROOT/build/FlashDexRetarget"
ln -sfn "$HERE/cfhoi_source.py" "$FDR/src/retarget/human_demo/sources/cfhoi.py"
ln -sfn "$HERE/cfhoi_source.yaml" "$FDR/config/retarget/source/cfhoi.yaml"
# the §1.4 policy's observation and reward configs (scripts/flashdex/policy/)
ln -sfn "$HERE/policy/config/obs_cfhoi_policy.yaml" "$FDR/config/envs/obs/cfhoi_policy.yaml"
ln -sfn "$HERE/policy/config/rewards_cfhoi_policy.yaml" "$FDR/config/envs/rewards/cfhoi_policy.yaml"
# env.sh for FlashDexRetarget's launchers: our path-based env instead of a named one
cat > "$FDR/scripts/env.sh" <<EOT
# FlashDexRetarget environment (written by CounterfactualHOI scripts/flashdex/setup_cfhoi.sh)
export PATH="$CONDA_ENVS_PATH/flashdex/bin:\$PATH"
export MUJOCO_GL=\${MUJOCO_GL:-disabled}
export OMNI_KIT_ACCEPT_EULA=1
export FDR_ROBOT_USD=\${FDR_ROBOT_USD:-$FDR/assets/robot/xhand/urdf/converted_spec/xhand_bimanual_spec.usd}
export TMPDIR=$PROJECT_STORAGE_ROOT/tmp XDG_CACHE_HOME=$SHARED_CACHE_ROOT PIP_CACHE_DIR=$SHARED_CACHE_ROOT/pip
export WANDB_MODE=\${WANDB_MODE:-offline} WANDB_DIR=$PROJECT_STORAGE_ROOT/runs/wandb
export TORCHINDUCTOR_CACHE_DIR=$PROJECT_STORAGE_ROOT/cache/torchinductor CUDA_CACHE_PATH=$PROJECT_STORAGE_ROOT/cache/nv
export TRITON_CACHE_DIR=$PROJECT_STORAGE_ROOT/cache/triton
PROJECT_DIR=$FDR
EOT
# MANO models without chumpy objects (smplx would need chumpy, which does not import on current numpy)
"$CONDA_ENVS_PATH/flashdex/bin/python" -I "$HERE/mano_nochumpy.py" "$CHOIR_ROOT/stage1_preprocess/Dyn_HaMR_new/_DATA/data/mano" "$FDR/assets/mano"
echo "linked cfhoi source; wrote $FDR/scripts/env.sh and chumpy-free MANO models"
