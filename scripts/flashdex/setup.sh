#!/usr/bin/env bash
# FlashDexRetarget (github.com/Holiday-Robot/FlashDexRetarget, MIT) for retargeting our reconstructions to the
# Sharpa Wave hand: its scripts/setup.sh, but with the conda env at $CONDA_ENVS_PATH/flashdex (not a named env in
# ~/miniconda3, which lives on the small root filesystem) and caches under /data.
#
#   bash scripts/flashdex/setup.sh        (clone in $PROJECT_STORAGE_ROOT/build/FlashDexRetarget)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
CONDA="$HOME/miniconda3/bin/conda"                      # only "create -p": the base env is never modified
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$SHARED_CACHE_ROOT/conda-pkgs}"
FDR="$PROJECT_STORAGE_ROOT/build/FlashDexRetarget"
E="$CONDA_ENVS_PATH/flashdex"
FDR_PIN=e84da37bca8ccb9543d9a1e4a202515ced6aad0d      # the commit our runs used (2026-11-23)
[ -d "$FDR/.git" ] || git clone -q https://github.com/Holiday-Robot/FlashDexRetarget.git "$FDR"
cd "$FDR"
git checkout -q "$FDR_PIN"
git submodule update --init third_party/rsl_rl_flashsac
[ -x "$E/bin/python" ] || "$CONDA" create -y -q -p "$E" python=3.11 pip
PIP="$E/bin/python -m pip"
$PIP install -q --upgrade pip
$PIP install -q "setuptools<81" wheel
$PIP install -q --no-build-isolation flatdict==4.0.1
FREEZE="$TMPDIR/fdr_freeze.txt"
grep -v -E '^(rsl-rl-lib|tensordict)==' requirements/isaac_cu128.txt > "$FREEZE"
$PIP install -q --no-deps --prefer-binary -r "$FREEZE" \
  --extra-index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.nvidia.com
$PIP install -q --no-deps rsl-rl-lib==5.5.1 tensordict==0.13.0 pyvers==0.2.2 coacd==1.0.10 cuacd==0.1.0 smplx==0.1.28
$PIP install -q --no-deps -e third_party/rsl_rl_flashsac
$PIP install -q --no-deps -e .
mkdir -p assets/mano                                   # MANO models for its human-demo import
for s in RIGHT LEFT; do
  [ -e "assets/mano/MANO_$s.pkl" ] || ln -s "$CHOIR_ROOT/stage1_preprocess/Dyn_HaMR_new/_DATA/data/mano/MANO_$s.pkl" "assets/mano/MANO_$s.pkl"
done
export MUJOCO_GL=disabled OMNI_KIT_ACCEPT_EULA=1
[ -f assets/robot/sharpa/urdf/converted/sharpa_bimanual.usd ] || \
  "$E/bin/python" src/simulator/isaacsim/convert_assets.py --robot --robot-name sharpa --robot-variant bimanual
"$E/bin/python" - <<'PY'
import importlib.metadata as md, torch
print(f"[setup] torch {torch.__version__} cuda={torch.cuda.is_available()}")
for p in ("isaacsim", "isaaclab", "rsl-rl-lib", "rsl-rl-flashsac", "tensordict", "FlashDexRetarget"):
    print(f"[setup] {p} {md.version(p)}")
PY
echo "[setup] done"
