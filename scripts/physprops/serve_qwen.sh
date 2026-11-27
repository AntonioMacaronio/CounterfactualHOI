#!/usr/bin/env bash
# Serve Qwen3.8-27B (FP8, image + text) with vLLM, OpenAI-compatible, for scripts/physprops/estimate.py.
#
#   bash scripts/physprops/serve_qwen.sh [GPU]          foreground; ready when `curl $QWEN_URL/v1/models` answers
#   QWEN_MEM=0.27 QWEN_PORT=30021 bash scripts/physprops/serve_qwen.sh 0
#
# One-time setup (both on /data):
#   $PROJECT_STORAGE_ROOT/venvs/hf-dl/bin/hf download Qwen/Qwen3.8-27B-FP8                   (~29 GB, HF cache)
#   uv venv --python 3.12 $PROJECT_STORAGE_ROOT/venvs/vllm && uv pip install --python $PROJECT_STORAGE_ROOT/venvs/vllm/bin/python vllm
#
# Made to share a GPU with other jobs: QWEN_MEM is the fraction of the GPU's memory vLLM may take (weights ~29 GB,
# the rest KV cache), a short context, eager mode (no CUDA-graph memory). Stop it when the estimates are done.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
GPU="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU" VLLM_CACHE_ROOT="$PROJECT_STORAGE_ROOT/cache/vllm" HF_HUB_OFFLINE=1
exec "$PROJECT_STORAGE_ROOT/venvs/vllm/bin/vllm" serve "${QWEN_MODEL:-Qwen/Qwen3.8-27B-FP8}" \
  --served-model-name qwen3.8-27b --port "${QWEN_PORT:-30021}" --host 127.0.0.1 \
  --gpu-memory-utilization "${QWEN_MEM:-0.27}" --max-model-len "${QWEN_CTX:-16384}" --max-num-seqs 4 \
  --enforce-eager --limit-mm-per-prompt '{"image": 8}' --reasoning-parser qwen3
