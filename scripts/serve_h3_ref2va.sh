#!/usr/bin/env bash
# Serve MiniMax H3-Base Ref2VA (video-to-video via reference video) with SGLang.
#
#   bash scripts/serve_h3_ref2va.sh            # foreground, all visible GPUs
#   CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 bash scripts/serve_h3_ref2va.sh
#
# Readiness: curl $H3_URL/health  -> {"status":"ok"} (503 while warming up).
# One-time setup (weights + venv) is described in videogen/README.md.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

NUM_GPUS="${NUM_GPUS:-4}"
# Ulysses-only is the H200 latency default (~94 GB/GPU at 4 GPUs). On 80 GB cards, or to
# leave headroom, use TP 2 + Ulysses 2 (~63 GB/GPU): TP_SIZE=2 ULYSSES=2.
TP_SIZE="${TP_SIZE:-1}"
ULYSSES="${ULYSSES:-$((NUM_GPUS / TP_SIZE))}"

[ -f "$H3_MODEL_DIR/Ref2VA/model_index.json" ] || { echo "Missing Ref2VA weights at $H3_MODEL_DIR" >&2; exit 1; }
command -v ffprobe >/dev/null || { echo "ffprobe not on PATH (expected in $TOOLS_ENV/bin)" >&2; exit 1; }

source "$H3_SGLANG_VENV/bin/activate"
# SGLang JIT-compiles fused kernels at warmup. No system CUDA toolkit here, so use the nvcc
# that ships in the venv's nvidia wheels. The wheel has lib/libcudart.so.13 but the JIT link
# step wants $CUDA_HOME/lib64/libcudart.so, so expose it through a symlink shim on /data.
CU13="$H3_SGLANG_VENV/lib/python3.12/site-packages/nvidia/cu13"
export CUDA_HOME="$PROJECT_STORAGE_ROOT/build/cuda13-shim"
if [ ! -e "$CUDA_HOME/lib64/libcudart.so" ]; then
  mkdir -p "$CUDA_HOME/lib64"
  for d in bin include nvvm; do ln -sfn "$CU13/$d" "$CUDA_HOME/$d"; done
  for f in "$CU13"/lib/*; do ln -sfn "$f" "$CUDA_HOME/lib64/$(basename "$f")"; done
  ln -sfn "$CU13/lib/libcudart.so.13" "$CUDA_HOME/lib64/libcudart.so"
fi
export PATH="$CUDA_HOME/bin:$PATH"
# Adds os.memfd_create (missing from uv's CPython build); see scripts/pyshim/sitecustomize.py.
export PYTHONPATH="$PROJECT_ROOT/scripts/pyshim${PYTHONPATH:+:$PYTHONPATH}"
# Keep the JIT and other SGLang caches off the root filesystem.
export SGLANG_CACHE_DIR="$PROJECT_STORAGE_ROOT/cache/sglang"
export SGLANG_JIT_CACHE_DIR="$SGLANG_CACHE_DIR/jit"
mkdir -p "$VIDEOGEN_OUT/server_outputs" "$VIDEOGEN_OUT/server_inputs"

# Note: --use-fsdp-inference is avoided on purpose (sglang#34227: silently corrupted output).
exec env HF_HUB_OFFLINE=1 sglang serve \
  --model-path "$H3_MODEL_DIR" \
  --model-variant ref2va \
  --num-gpus "$NUM_GPUS" \
  --tp-size "$TP_SIZE" \
  --ulysses-degree "$ULYSSES" \
  --encoder-parallel auto \
  --performance-mode speed \
  --warmup-resolutions 1344x768 \
  --output-path "$VIDEOGEN_OUT/server_outputs" \
  --input-save-path "$VIDEOGEN_OUT/server_inputs" \
  --host 127.0.0.1 \
  --port "$H3_PORT"
