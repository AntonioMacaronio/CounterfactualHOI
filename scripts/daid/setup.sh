#!/usr/bin/env bash
# Set up Do-as-I-Do reconstruction (third_party/do-as-i-do/reconstruction) on an H200 node, keeping
# weights, envs and caches on /data. Replaces the upstream setup/01_create_envs.sh + 02_fetch_weights.sh,
# which write weights into the repo and pip-install into whatever env is active.
#
#   bash scripts/daid/setup.sh weights            # SAM3D, SAM3, HaWoR, BootsTAPIR, MANO links
#   bash scripts/daid/setup.sh env sam3|tapnet|sam3d|hawor|retargeting
#   bash scripts/daid/setup.sh check              # import smoke test for each env
#
# Envs: conda-forge python + ffmpeg + CUDA 12.8 compiler, then pip at the exact versions of the
# upstream working envs (reconstruction/env/*.yml), with the source builds compiled for one GPU architecture:
# TORCH_CUDA_ARCH_LIST, default 9.0 (H100/H200); set it first on other GPUs (e.g. 8.0 A100, 8.9 L40S/RTX 4090).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

R="$DAID_ROOT/reconstruction"
W="$DAID_WEIGHTS"
CONDA="$HOME/miniconda3/bin/conda"
HF="$PROJECT_STORAGE_ROOT/venvs/hf-dl/bin/hf"
PINS="python3 $PROJECT_ROOT/scripts/daid/yml_pins.py"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"     # 9.0 = H100/H200
export MAX_JOBS="${MAX_JOBS:-32}"
export FORCE_CUDA=1

link() {  # link SRC DST: symlink unless DST is a real (tracked) file
  local src="$1" dst="$2"
  if [ -e "$dst" ] && [ ! -L "$dst" ]; then echo "  keep tracked $dst"; return; fi
  ln -sfn "$src" "$dst"
}

weights() {
  mkdir -p "$W/weights/tapnet"
  if [ ! -L "$R/weights" ]; then
    [ -e "$R/weights" ] && { echo "$R/weights exists and is not a symlink; move it first" >&2; exit 1; }
    ln -s "$W/weights" "$R/weights"
  fi

  echo "[weights] SAM3D (gated facebook/sam-3d-objects, ~13 GB)"
  "$HF" download facebook/sam-3d-objects --include "checkpoints/*" --local-dir "$W/sam3d-download"
  for name in sam-3d-objects Fast-SAM3D; do
    dst="$R/modules/$name/checkpoints/hf"; mkdir -p "$dst"
    for f in "$W/sam3d-download/checkpoints/"*; do link "$f" "$dst/$(basename "$f")"; done
  done

  echo "[weights] SAM3 (gated facebook/sam3) into the HF cache used at runtime"
  "$HF" download facebook/sam3 config.json sam3.pt

  echo "[weights] HaWoR (+ WiLoR detector, DROID, Metric3D) and MANO"
  "$HF" download ThunderVVV/HaWoR --local-dir "$W/hawor"
  H="$R/modules/HaWoR"
  link "$W/hawor" "$H/weights"          # weights/{external,hawor}/... matches HaWoR's layout
  link "$W/hawor/external/metric_depth_vit_large_800k.pth" "$H/thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth"
  mkdir -p "$H/_DATA/data/mano" "$H/_DATA/data_left/mano_left"
  [ -f "$MANO_DIR/MANO_RIGHT.pkl" ] || { echo "MANO_RIGHT.pkl not found in $MANO_DIR" >&2; exit 1; }
  link "$MANO_DIR/MANO_RIGHT.pkl" "$H/_DATA/data/mano/MANO_RIGHT.pkl"
  link "$MANO_DIR/MANO_LEFT.pkl" "$H/_DATA/data_left/mano_left/MANO_LEFT.pkl"

  echo "[weights] BootsTAPIR"
  wget -q -nc https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt -P "$W/weights/tapnet/"
  du -shL "$W"/* 2>/dev/null
}

mkenv() {  # mkenv NAME PYTHON [extra conda pkgs...]
  local name="$1" py="$2"; shift 2
  E="$CONDA_ENVS_PATH/$name"
  if [ ! -x "$E/bin/python" ]; then
    "$CONDA" create -y -q -p "$E" -c conda-forge --override-channels "python=$py" "ffmpeg=7.1" "$@"   # run_pipeline.sh uses -vsync (removed in ffmpeg 8)
  fi
  PIP="$E/bin/python -m pip"
  $PIP install -q --upgrade pip
}

cuda_build_env() {  # point torch extension builds at the env's conda-forge CUDA 12.8 toolkit
  export CUDA_HOME="$E" PATH="$E/bin:$PATH"
  export CPATH="$E/targets/x86_64-linux/include${CPATH:+:$CPATH}"
  export LIBRARY_PATH="$E/targets/x86_64-linux/lib:$E/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
}

CUDA_PKGS=(cuda-version=12.8 cuda-nvcc cuda-cudart-dev cuda-libraries-dev cuda-cccl)

env_sam3() {
  mkenv sam3 3.12
  $PIP install -q torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
  $PIP install -q -e "$R/modules/sam3[dev,notebooks,train]"
}

env_tapnet() {
  mkenv tapnet 3.10
  $PIP install -q torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
  $PIP install -q -e "$R/modules/tapnet[torch]"
  $PIP install -q einops tqdm mediapy
}

env_sam3d() {
  mkenv sam3d 3.11 "${CUDA_PKGS[@]}"
  cuda_build_env
  local reqs="$TMPDIR/sam3d_reqs.txt"
  $PINS "$R/env/sam3d.yml" --exclude pytorch3d nvdiffrast kaolin diff-gaussian-rasterization \
    sam3d-objects moge utils3d conda-env-export > "$reqs"
  $PIP install -q torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
  $PIP install -q -r "$reqs"
  $PIP install -q kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html
  $PIP install -q "git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b"
  # SAM3D forces ATTN_BACKEND=flash_attn on A100/H100/H200 (inference_pipeline.py), but DAID's env (built on
  # other GPUs) lacks it. Prebuilt wheel at SAM3D's pinned version (requirements.p3d.txt).
  $PIP install -q "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
  echo "[sam3d] building pytorch3d 0.7.9, nvdiffrast, diff-gaussian-rasterization for sm_$TORCH_CUDA_ARCH_LIST"
  $PIP install -q --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9"
  $PIP install -q --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git@v0.4.0"
  local ms="$PROJECT_STORAGE_ROOT/build/mip-splatting"
  [ -d "$ms" ] || git clone -q --recursive https://github.com/autonomousvision/mip-splatting.git "$ms"
  $PIP install -q --no-build-isolation "$ms/submodules/diff-gaussian-rasterization"
  $PIP install -q --no-deps -e "$R/modules/sam-3d-objects"
  $PIP install -q "geocalib @ git+https://github.com/cvg/GeoCalib.git"
  $PIP uninstall -y -q notebook || true     # un-shadow the repo's notebook/ package (upstream env/README)
  "$E/bin/python" "$R/modules/sam-3d-objects/patching/hydra"
}

env_hawor() {
  mkenv hawor 3.10 "${CUDA_PKGS[@]}"
  cuda_build_env
  local reqs="$TMPDIR/hawor_reqs.txt"
  $PINS "$R/env/hawor.yml" --exclude droid-backends lietorch pytorch3d torch-scatter chumpy > "$reqs"
  $PIP install -q torch==2.9.0+cu128 torchvision==0.24.0+cu128 --index-url https://download.pytorch.org/whl/cu128
  $PIP install -q "setuptools<81"            # pytorch-lightning 2.2.4 needs pkg_resources
  $PIP install -q -r "$reqs"
  $PIP install -q torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.9.0+cu128.html
  $PIP install -q --no-build-isolation "chumpy@git+https://github.com/mattloper/chumpy"
  echo "[hawor] building pytorch3d 0.7.8 and DROID-SLAM (lietorch + droid_backends) for sm_$TORCH_CUDA_ARCH_LIST"
  $PIP install -q --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8"   # sic: this release is tagged with a capital V
  # lietorch's dispatch.h uses an API removed in torch 2.x (upstream env/README one-line fix).
  local disp="$R/modules/HaWoR/thirdparty/DROID-SLAM/thirdparty/lietorch/lietorch/include/dispatch.h"
  sed -i 's/at::ScalarType _st = ::detail::scalar_type(the_type);/at::ScalarType _st = the_type.scalarType();/' "$disp"
  # DROID-SLAM's setup.py declares two packages (lietorch, droid_backends), which pip's metadata step
  # rejects ("Multiple .egg-info directories"), so use the legacy install, as upstream does.
  (cd "$R/modules/HaWoR/thirdparty/DROID-SLAM" && \
   "$E/bin/python" setup.py -q build --build-base "$PROJECT_STORAGE_ROOT/build/droid-slam" install)
  # torch>=2.6 defaults torch.load(weights_only=True), which rejects HaWoR's (trusted) checkpoints.
  "$CONDA" env config vars set -p "$E" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 >/dev/null
}

env_retargeting() {
  # Do-as-I-Do retargeting (MuJoCo Warp + mink -> Sharpa hand). Pins from retargeting/env/retargeting.yml;
  # torch 2.12.0 is PyPI's default CUDA 13 build (the node's 580 driver supports CUDA 13.0).
  mkenv retargeting 3.12
  $PIP install -q torch==2.12.0 numpy==2.4.6 coacd==1.0.11 mink==1.1.0 viser==1.0.30 \
    warp-lang==1.10.1 mujoco==3.4.0
  $PIP install -q -e "$DAID_ROOT/retargeting"
}

check() {
  for name in sam3 tapnet sam3d hawor retargeting; do
    E="$CONDA_ENVS_PATH/$name"
    [ -x "$E/bin/python" ] || { echo "$name: not built"; continue; }
    case $name in
      sam3)   mods="torch sam3.model_builder" ;;
      tapnet) mods="torch tapnet.torch.tapir_model mediapy" ;;
      sam3d)  mods="torch pytorch3d.renderer kaolin nvdiffrast.torch diff_gaussian_rasterization gsplat spconv.pytorch moge.model geocalib sam3d_objects viser" ;;
      hawor)  mods="torch pytorch3d droid_backends lietorch torch_scatter chumpy smplx pytorch_lightning" ;;
      retargeting) mods="torch mujoco mujoco_warp warp mink coacd viser retargeting" ;;
    esac
    # SAM3D's entry points set LIDRA_SKIP_INIT before importing (sam3d_objects.init is Meta-internal).
    (cd "$R" && LIDRA_SKIP_INIT=true "$E/bin/python" - $mods <<'EOF'
import importlib, sys, torch
bad = []
for m in sys.argv[1:]:
    try: importlib.import_module(m)
    except Exception as e: bad.append(f"{m}: {type(e).__name__}: {e}"[:200])
ok = torch.cuda.is_available() and float((torch.ones(1, device="cuda") * 2).item()) == 2.0
cap = "sm_%d%d" % torch.cuda.get_device_capability() if torch.cuda.is_available() else "-"
print(f"torch {torch.__version__} cuda_ok={ok} ({cap}) ->", "OK" if not bad else "FAIL\n  " + "\n  ".join(bad))
EOF
    ) 2>&1 | tail -4 | sed "s/^/$name: /"
  done
}

case "${1:-}" in
  weights) weights ;;
  env) "env_${2:?env name: sam3|tapnet|sam3d|hawor|retargeting}" ;;
  check) check ;;
  *) sed -n '2,12p' "$0"; exit 1 ;;
esac
