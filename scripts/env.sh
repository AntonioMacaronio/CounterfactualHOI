# Source this before running anything in this repo:  source scripts/env.sh
# Keeps all large/growing data (weights, venvs, caches, outputs) on /data, not the root fs.

export PROJECT_NAME="CounterfactualHOI"
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_STORAGE_ROOT="/data/sky/projects/$PROJECT_NAME"
export SHARED_CACHE_ROOT="/data/sky/cache"

export TMPDIR="$PROJECT_STORAGE_ROOT/tmp"
export XDG_CACHE_HOME="$SHARED_CACHE_ROOT"
export PIP_CACHE_DIR="$SHARED_CACHE_ROOT/pip"
export UV_CACHE_DIR="$SHARED_CACHE_ROOT/uv"
export UV_PYTHON_INSTALL_DIR="$SHARED_CACHE_ROOT/uv-python"
# ~/.config/uv/uv.toml points at a FAR CodeArtifact index this project can't auth to; use PyPI.
export UV_NO_CONFIG=1
export UV_KEYRING_PROVIDER=disabled
export HF_HOME="$SHARED_CACHE_ROOT/huggingface"
if [ -f "$HOME/.cache/huggingface/token" ]; then export HF_TOKEN_PATH="$HOME/.cache/huggingface/token"; fi
export TORCH_HOME="$SHARED_CACHE_ROOT/torch"
export TORCH_EXTENSIONS_DIR="$PROJECT_STORAGE_ROOT/cache/torch_extensions"
export TRITON_CACHE_DIR="$PROJECT_STORAGE_ROOT/cache/triton"
export WANDB_DIR="$PROJECT_STORAGE_ROOT/runs/wandb"
export WANDB_CACHE_DIR="$SHARED_CACHE_ROOT/wandb"
export CONDA_PKGS_DIRS="$SHARED_CACHE_ROOT/conda-pkgs"
export FLASHINFER_WORKSPACE_BASE="$PROJECT_STORAGE_ROOT/cache"

# MiniMax H3 (self-hosted Ref2VA via SGLang)
export H3_MODEL_DIR="$PROJECT_STORAGE_ROOT/models/MiniMax-H3"
export H3_REVISION="42ed227ee7df40d41602854ae760620d6eb651fe"
export H3_SGLANG_COMMIT="2511743bd784e69e5a81ca3d926a000711dae4ab"
export H3_SGLANG_SRC="$PROJECT_STORAGE_ROOT/build/sglang"
export H3_SGLANG_VENV="$PROJECT_STORAGE_ROOT/venvs/sglang-h3"
export H3_PORT="${H3_PORT:-30011}"
export H3_URL="${H3_URL:-http://127.0.0.1:$H3_PORT}"

# Small tool env (ffmpeg/ffprobe + the videogen client)
export TOOLS_ENV="$PROJECT_STORAGE_ROOT/venvs/tools"
export VIDEOGEN_OUT="$PROJECT_STORAGE_ROOT/runs/videogen"

mkdir -p "$TMPDIR" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$WANDB_DIR" \
         "$PROJECT_STORAGE_ROOT/models" "$PROJECT_STORAGE_ROOT/venvs" \
         "$PROJECT_STORAGE_ROOT/build" "$VIDEOGEN_OUT"

if [ -d "$TOOLS_ENV/bin" ]; then export PATH="$TOOLS_ENV/bin:$PATH"; fi
