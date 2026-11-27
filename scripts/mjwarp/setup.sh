#!/usr/bin/env bash
# The MuJoCo Warp environment for FlashDexRetarget's tracking env (scripts/mjwarp/): a uv venv at
# $PROJECT_STORAGE_ROOT/venvs/fdr-mjwarp with vanilla mjlab 1.3.0 (the version FDR vendors, trimmed, for its Isaac
# backend), MuJoCo Warp, and FDR's own packages without Isaac Sim.
#
#   bash scripts/mjwarp/setup.sh          (after scripts/flashdex/setup.sh: FDR's checkout and its rsl_rl_flashsac)
#
# Versions: mujoco-warp 3.8.1 / mujoco 3.8.0 / warp 1.12.1, the releases closest to mjlab 1.3.0 (April 2026; warp
# 1.19 drops wp.context, which mjlab 1.3.0 reads). torch 2.7.0 cu128 as in FDR's Isaac env (FlashSAC's torch.compile
# was tuned on it). mjlab 1.3.0 pins rsl-rl-lib 5.0.1 and FDR needs 5.5.1, so mjlab goes in with --no-deps and its
# other requirements are listed here. Also links our simulator config into FDR (config/envs/simulator/mjwarp.yaml)
# and builds the wrist-hardware hand (scripts/mjwarp/hand_variant.py) when the embodiment models exist.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
V="$PROJECT_STORAGE_ROOT/venvs/fdr-mjwarp"
PY="$V/bin/python"
PIP=(uv pip install -q --python "$PY")
[ -x "$PY" ] || uv venv -q --python 3.11 "$V"
"${PIP[@]}" torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
"${PIP[@]}" --no-deps mjlab==1.3.0
"${PIP[@]}" "mujoco-warp==${MJWARP_VERSION:-3.8.1}" "mujoco==${MUJOCO_VERSION:-3.8.0}" "warp-lang==1.12.1" \
  prettytable tqdm "tyro>=1.0.1" "trimesh==4.5.1" "numpy<2.3" "scipy==1.15.3" mediapy \
  hydra-core==1.3.2 omegaconf==2.3.0 gymnasium==1.2.0 tensordict==0.13.0 pyvers==0.2.2 wandb smplx==0.1.28 \
  coacd==1.0.10 rtree pyyaml imageio-ffmpeg torchrunx viser onnxscript tensorboard gitpython
# mjviser without its dependencies: they pull mujoco-core 4.x, whose libmujoco clashes with mujoco 3.8
"${PIP[@]}" --no-deps mjviser==0.0.11
"${PIP[@]}" --no-deps rsl-rl-lib==5.5.1
"${PIP[@]}" --no-deps -e "$FDR/third_party/rsl_rl_flashsac"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ln -sfn "$HERE/config/simulator_mjwarp.yaml" "$FDR/config/envs/simulator/mjwarp.yaml"
if [ -d "$PROJECT_STORAGE_ROOT/build/embodiment/vega_1p_sharpa_r/wrist_stub" ]; then
  "$PY" "$HERE/hand_variant.py" --arm vega_1p
fi
"$PY" - <<'PY'
import importlib.metadata as md
import mujoco, mujoco_warp, warp, torch
print(f"[mjwarp] torch {torch.__version__} cuda={torch.cuda.is_available()}; mujoco {mujoco.__version__}; "
      f"mujoco_warp {md.version('mujoco-warp')}; warp {warp.__version__}; mjlab {md.version('mjlab')}")
PY
