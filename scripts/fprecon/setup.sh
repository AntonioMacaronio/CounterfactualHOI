#!/usr/bin/env bash
# Environments for the FoundationPose + Depth Pro 2 reconstruction pipeline (scripts/fprecon/), all on /data.
#
#   bash scripts/fprecon/setup.sh fpose      # FoundationPose @ build/FoundationPose, conda env conda_envs/fpose
#   bash scripts/fprecon/setup.sh depthpro2  # Depth Pro 2 @ build/ml-depth-pro-2, uv venv venvs/depthpro2
#   bash scripts/fprecon/setup.sh check
#
# FoundationPose follows its readme's conda route (python 3.11, numpy 2), with conda's CUDA 12.6 toolkit and the
# system gcc 11 as for the CHOIR envs; torch 2.6 cu126 matches the choir env. Weights: the official Google Drive
# folder (refiner 2023-10-28-18-33-37, scorer 2024-01-11-20-02-45). Depth Pro 2 is a uv project (torch>=2.7);
# its H+ weights (3.5 GB) are fetched into the HF cache on first use.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
CONDA="$HOME/miniconda3/bin/conda"
FP="$PROJECT_STORAGE_ROOT/build/FoundationPose"
DP="$PROJECT_STORAGE_ROOT/build/ml-depth-pro-2"
FP_PIN=a1b694b
DP_PIN=aa389cb

fpose() {
  [ -d "$FP/.git" ] || git clone -q https://github.com/NVlabs/FoundationPose.git "$FP"
  git -C "$FP" checkout -q "$FP_PIN"
  local E="$CONDA_ENVS_PATH/fpose"
  [ -x "$E/bin/python" ] || "$CONDA" create -y -q -p "$E" -c conda-forge --override-channels python=3.11 pip \
    cmake ninja eigen boost-cpp pybind11 libusb cuda-version=12.6 cuda-nvcc cuda-cudart-dev cuda-libraries-dev cuda-cccl
  local PIP="$E/bin/python -m pip"
  export CUDA_HOME="$E" PATH="$E/bin:$PATH" CONDA_PREFIX="$E"
  export CPATH="$E/targets/x86_64-linux/include${CPATH:+:$CPATH}"
  export LIBRARY_PATH="$E/targets/x86_64-linux/lib:$E/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
  export TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=32
  $PIP install -q --upgrade pip setuptools wheel
  $PIP install -q torch==2.6.0+cu126 torchvision==0.21.0+cu126 --index-url https://download.pytorch.org/whl/cu126
  $PIP install -q -r "$FP/requirements.txt"
  # rtree: trimesh proximity queries (refinement agent's SDF); anthropic[bedrock]: the refinement agent (scripts/refine).
  $PIP install -q rtree "anthropic[bedrock]"
  "$E/bin/python" -c "import pytorch3d" 2>/dev/null ||
    $PIP install -q --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8"
  "$E/bin/python" -c "import nvdiffrast.torch" 2>/dev/null ||
    $PIP install -q --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git@v0.4.0"
  (cd "$FP" && bash build_all_conda.sh > "$TMPDIR/fpose_mycpp_build.log" 2>&1)
  local W="$FP/weights"
  [ -f "$W/2023-10-28-18-33-37/model_best.pth" ] && [ -f "$W/2024-01-11-20-02-45/model_best.pth" ] ||
    uvx --from gdown gdown -q --folder "https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i" -O "$W"
}

depthpro2() {
  [ -d "$DP/.git" ] || git clone -q https://github.com/apple-aiml-research/ml-depth-pro-2.git "$DP"
  git -C "$DP" checkout -q "$DP_PIN"
  (cd "$DP" && UV_PROJECT_ENVIRONMENT="$PROJECT_STORAGE_ROOT/venvs/depthpro2" uv sync -q)
}

check() {
  (cd "$FP" && CUDA_VISIBLE_DEVICES=0 "$CONDA_ENVS_PATH/fpose/bin/python" -c "
import torch, nvdiffrast.torch, pytorch3d, warp
from estimater import FoundationPose
print('fpose: torch', torch.__version__, 'cuda', torch.cuda.is_available(), '-> OK')")
  CUDA_VISIBLE_DEVICES=0 "$PROJECT_STORAGE_ROOT/venvs/depthpro2/bin/python" -c "
import torch, depth_pro_2
print('depthpro2: torch', torch.__version__, 'cuda', torch.cuda.is_available(), '-> OK')"
}

"$@"
