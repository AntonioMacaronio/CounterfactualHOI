#!/usr/bin/env bash
# Everything scripts/embodiment needs: a small CPU-only venv and the vendors' robot models at pinned commits.
#
#   bash scripts/embodiment/setup.sh
#
# Nothing here uses a GPU. All of it lands on /data ($PROJECT_STORAGE_ROOT):
#   venvs/arm                      mujoco (the version FDR's env has), qpsolvers + daqp (IK), pyroki + jax (collision-
#                                  aware IK), cadquery (mount CAD), trimesh, torch-cpu (to read FDR's motion.pt),
#                                  pymupdf (drawings -> PNG)
#   build/tianji_marvin            Tianji Marvin M6-CCS URDF + meshes, kinematics config, manual   (Apache-2.0)
#   build/dexmate_urdf             Dexmate Vega models                                              (Apache-2.0)
#   build/sharpa_upstream          Sharpa Wave flange / wrist meshes and the with_flange URDF       (Apache-2.0)
# The hand itself comes from FlashDexRetarget's checkout: run scripts/flashdex/setup.sh first.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
V="$PROJECT_STORAGE_ROOT/venvs/arm"
B="$PROJECT_STORAGE_ROOT/build"
echo "[setup] storage: $PROJECT_STORAGE_ROOT"

fetch() {  # url dest sha256
  [ -f "$2" ] || { mkdir -p "$(dirname "$2")"; curl -sSL --fail --max-time 600 -o "$2" "$1"; }
  echo "$3  $2" | sha256sum -c --quiet - || { echo "[setup] checksum mismatch: $2" >&2; exit 1; }
}
unrar_to() {  # archive dir   (RAR: libarchive's bsdtar, 7z or unrar, whichever exists)
  mkdir -p "$2"
  if command -v bsdtar >/dev/null; then bsdtar -xf "$1" -C "$2"
  elif [ -x "$HOME/miniconda3/bin/bsdtar" ]; then "$HOME/miniconda3/bin/bsdtar" -xf "$1" -C "$2"
  elif command -v 7z >/dev/null; then 7z x -y -o"$2" "$1" >/dev/null
  elif command -v unrar >/dev/null; then unrar x -o+ "$1" "$2/" >/dev/null
  else echo "[setup] need bsdtar, 7z or unrar to unpack $1" >&2; exit 1; fi
}

# ---- venv
if [ ! -x "$V/bin/python" ]; then uv venv -q --python 3.11 "$V"; fi
uv pip install -q --python "$V/bin/python" "mujoco==3.10.0" "numpy>=2" scipy trimesh pyyaml pillow matplotlib \
  "qpsolvers>=4.13" daqp "cadquery>=2.8" pymupdf pymeshlab
uv pip install -q --python "$V/bin/python" torch --index-url https://download.pytorch.org/whl/cpu
# PyRoki (collision-aware IK, ik_pyroki.py) and its solver jaxls: git only, pinned. JAX installs CPU-only.
uv pip install -q --python "$V/bin/python" \
  "jaxls @ git+https://github.com/brentyi/jaxls.git@50a58be88c5ef74532f09e3f55268b4f02c490e3" \
  "pyroki @ git+https://github.com/chungmin99/pyroki.git@388e43e1fc0d0ee382968d3dd72970fd62a0450c"

# ---- Tianji Marvin M6-S-CCS-696 V4.0
TJ=https://raw.githubusercontent.com/cynthia-you/TJ_FX_ROBOT_CONTRL_SDK
TJ_DOCS=2ae22e4bec5ea1ce8bebbfcd4c15713bc56414b6      # branch 资料文档
TJ_SDK=44ff42090d085cf77414a8ed468ecc54cc0600c6       # branch master
T="$B/tianji_marvin"
fetch "$TJ/$TJ_DOCS/URDF/Marvin%20M6-CCS-696-urdf_V4.0.rar" "$T/urdf_rar/Marvin_M6-CCS-696-urdf_V4.0.rar" \
  8471d1d8c77c56d41101bdbc51ba38323c42a835e870658b363d9f6cb02a027c
fetch "$TJ/$TJ_SDK/CommonConfig/ccs_m6_40.MvKDCfg" "$T/sdk_master/ccs_m6_40.MvKDCfg" \
  ce20b1a80974c3ed58cdac4973152625096c8ad93a779e5fd4108c26ce19bd52
fetch "$TJ/$TJ_SDK/Marvin%E7%B3%BB%E5%88%97%E6%9C%BA%E5%99%A8%E4%BA%BA%E4%BD%BF%E7%94%A8%E8%AF%B4%E6%98%8E%E4%B9%A6_1121200066_V2.2.pdf" \
  "$T/sdk_master/marvin_manual_V2.2.pdf" acbea475a3ad047372084d0e533d853e56be88272d2de31d7c496907603f679d
[ -f "$T/sdk_master/LICENSE" ] || curl -sSL -o "$T/sdk_master/LICENSE" "$TJ/$TJ_SDK/LICENSE"
if [ ! -d "$T/urdf_m6/R" ]; then
  unrar_to "$T/urdf_rar/Marvin_M6-CCS-696-urdf_V4.0.rar" "$T/urdf_m6"
  for s in L R; do unrar_to "$T/urdf_m6/Marvin M6-CCS-696-urdf_V4.0/Marvin M6-S-$s-CCS-696-V4.0 urdf (FUSION).rar" "$T/urdf_m6/$s"; done
fi

# ---- Dexmate Vega
DX=ffcc62500fea1e2e7600411b4ec9f3b420bb780a
D="$B/dexmate_urdf"
fetch "https://codeload.github.com/dexmate-ai/dexmate-urdf/tar.gz/$DX" "$D/dexmate-urdf-$DX.tar.gz" \
  9ef8178a9a87487fd79b763c54e323ed29914d3d7a07965407af7b9ebb0a367e
[ -d "$D/dexmate-urdf-$DX/robots" ] || tar -xzf "$D/dexmate-urdf-$DX.tar.gz" -C "$D" --exclude='*/src/dexmate_urdf/robots/*'

# ---- Sharpa Wave flange and wrist (the hand's own meshes come with FlashDexRetarget)
SH=https://raw.githubusercontent.com/sharpa-robotics/sharpa-urdf-usd-xml/d2db0b8ba4aeb737e6274d4c4386e210eedfa1da/wave_01/right_sharpa_wave
S="$B/sharpa_upstream/right_sharpa_wave"
fetch "$SH/meshes/flange_B.STL" "$S/meshes/flange_B.STL" c5051bf5e1ee87cdecc061326d4413b9e6232fe07777e5b3437f4a459a9280bc
fetch "$SH/meshes/wrist_B.STL" "$S/meshes/wrist_B.STL" ce978dda8bbfb8ad59cb2f5decf647882d1be67fd758329d40c7c4cefacbf771
fetch "$SH/meshes/right_hand_C_MC_visual_.STL" "$S/meshes/right_hand_C_MC_visual_.STL" \
  9234e513974246de38de76ce5f2a48078ebb014e890083219e2bd93caef55b60
fetch "$SH/right_sharpa_wave_with_flange.urdf" "$S/right_sharpa_wave_with_flange.urdf" \
  261c1a5284e04bc9690e0f5ac39d7a7cace0a16d335d663a1d3c7fcf4e4daf31

[ -f "$FDR/assets/robot/sharpa/right.xml" ] || { echo "[setup] FlashDexRetarget's Sharpa model is missing: run scripts/flashdex/setup.sh" >&2; exit 1; }

# ---- build: mount CAD, then the models (the model picks the mount's mesh up)
cd "$PROJECT_ROOT/scripts/embodiment"
export MUJOCO_GL=disabled
for arm in marvin_m6ccs vega_1p vega_1p_no_ft; do
  "$V/bin/python" d405_mount.py --arm "$arm"
  "$V/bin/python" build_model.py --arm "$arm"
done
echo "[setup] done: $PROJECT_STORAGE_ROOT/build/embodiment"
