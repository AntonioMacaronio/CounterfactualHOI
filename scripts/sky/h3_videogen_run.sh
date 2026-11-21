#!/usr/bin/env bash
# Run step of scripts/sky/h3_videogen.yaml: serve MiniMax H3 Ref2VA on this pod's GPUs, generate CLIPS with
# videogen.generate, and mirror the outputs to $CFHOI_S3/videogen/$RUN_NAME/ while it runs.
#
# Resumable: a recovered job first pulls this job's clips from S3, and generate.py skips any seed<k>.mp4 that
# already exists, so only missing clips are generated. Exits non-zero unless every planned clip exists, which
# lets the managed job's max_restarts_on_errors retry the missing ones on a fresh pod.
set -euo pipefail
: "${RUN_NAME:?}" "${CLIPS:?}" "${CFHOI_S3:?}" "${CONFIG:?}"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/env.sh
export NUM_GPUS="${NUM_GPUS:-4}"
export VIDEOGEN_GEN_DIR="$PROJECT_STORAGE_ROOT/runs/videogen/$RUN_NAME"
OUT_S3="$CFHOI_S3/videogen/$RUN_NAME"
VIDEO_DIR="$PROJECT_STORAGE_ROOT/datasets/realvideos"
TAG="${CLIPS// /+}"
LOG_DIR="$VIDEOGEN_GEN_DIR/logs/$TAG"
MANIFEST="$VIDEOGEN_GEN_DIR/manifests/$TAG.jsonl"
mkdir -p "$LOG_DIR" "$(dirname "$MANIFEST")"
read -r -a CLIP_ARR <<< "$CLIPS"
read -r -a MODE_ARR <<< "${MODES:-type_a type_b}"
read -r -a SEED_ARR <<< "${SEEDS:-0}"
GEN_ARGS=(--config "$CONFIG" --video-dir "$VIDEO_DIR" --out "$VIDEOGEN_GEN_DIR" --manifest "$MANIFEST"
          --clips "${CLIP_ARR[@]}" --modes "${MODE_ARR[@]}" --seeds "${SEED_ARR[@]}" --num-gpus "$NUM_GPUS")

# Only this job's files: its clips, their normalized sources, its manifest and logs.
SCOPE=(--exclude '*')
for c in "${CLIP_ARR[@]}"; do SCOPE+=(--include "$c/*" --include "sources/$c.mp4"); done
SCOPE+=(--include "manifests/$TAG.jsonl" --include "logs/$TAG/*")
# Skip in-progress files: the client downloads to seed<k>.mp4.part, prep.normalize writes <clip>.part.mp4.
sync_up() { aws s3 sync "$VIDEOGEN_GEN_DIR/" "$OUT_S3/" --only-show-errors "${SCOPE[@]}" --exclude '*.part' --exclude '*.part.mp4'; }

aws s3 sync "$OUT_S3/" "$VIDEOGEN_GEN_DIR/" --only-show-errors "${SCOPE[@]}"
# Keep earlier attempts' logs: this attempt writes its own files.
ATTEMPT="$(date -u +%Y%m%dT%H%M%SZ)"
{
  echo "attempt $ATTEMPT on $(hostname), job ${SKYPILOT_TASK_ID:-?}"
  echo "clips: $CLIPS; modes: ${MODE_ARR[*]}; seeds: ${SEED_ARR[*]}; config: $CONFIG"
  echo "code: $CODE_BUNDLE sha256 $CODE_SHA256"
  echo "env: $ENV_TAR sha256 $ENV_SHA256; H3 $H3_REVISION; SGLang $H3_SGLANG_COMMIT"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
} | tee "$LOG_DIR/attempt_$ATTEMPT.txt"

python - "${GEN_ARGS[@]}" > "$LOG_DIR/planned.txt" <<'PY'
import json, sys
from videogen.generate import parse_args, plan_jobs
args = parse_args(sys.argv[1:])
for j in plan_jobs(json.loads(args.config.read_text()), args):
    vid = j["variant"]["id"] if j["variant"] else "identity"
    print(args.out / j["clip_id"] / j["mode"] / vid / f"seed{j['seed']}.mp4")
PY
n_planned=$(wc -l < "$LOG_DIR/planned.txt")
n_done() { local n=0 f; while read -r f; do [[ -s "$f" ]] && n=$((n + 1)); done < "$LOG_DIR/planned.txt"; echo "$n"; }
echo "planned $n_planned clip(s), $(n_done) already done"

if (( $(n_done) < n_planned )); then
  bash scripts/serve_h3_ref2va.sh > "$LOG_DIR/server_$ATTEMPT.log" 2>&1 &
  SERVER_PID=$!
  ( while sleep 120; do sync_up || true; done ) &
  SYNC_PID=$!
  trap 'kill "$SYNC_PID" "$SERVER_PID" 2>/dev/null || true' EXIT

  t0=$SECONDS
  until curl -sf "$H3_URL/health" 2>/dev/null | grep -q '"ok"'; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "H3 server exited during startup"; tail -n 80 "$LOG_DIR/server_$ATTEMPT.log"; sync_up; exit 1
    fi
    if (( SECONDS - t0 > 3600 )); then
      echo "H3 server not healthy after 60 min"; tail -n 80 "$LOG_DIR/server_$ATTEMPT.log"; sync_up; exit 1
    fi
    sleep 15
  done
  echo "H3 server healthy after $((SECONDS - t0)) s"

  set +e
  python -m videogen.generate "${GEN_ARGS[@]}" 2>&1 | tee "$LOG_DIR/generate_$ATTEMPT.log"
  gen_rc=${PIPESTATUS[0]}
  set -e
  echo "generate exited with $gen_rc"
  kill "$SYNC_PID" 2>/dev/null || true
  wait "$SYNC_PID" 2>/dev/null || true
fi

sync_up
done_now=$(n_done)
echo "$done_now/$n_planned clip(s) present; outputs: $OUT_S3/"
(( done_now == n_planned )) || { echo "INCOMPLETE: $((n_planned - done_now)) clip(s) missing"; exit 1; }
echo "ALL CLIPS DONE"
