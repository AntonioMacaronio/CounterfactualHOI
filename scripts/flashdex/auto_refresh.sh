#!/bin/bash
# Keep the runs' learning curves current and re-export tab 5 by itself (runs until killed; one line per action).
#
#   nohup bash scripts/flashdex/auto_refresh.sh >> $PROJECT_STORAGE_ROOT/runs/logs/flashdex_auto_refresh.log 2>&1 &
#
# Every 5 min: eval_curves.py on every run with evals (scored against the motion file it trained on). The page's run is read from CONF each time (one line:
# "<run dir name> <dataset dir name> <refine round>"), so the page can be switched without restarting. It is
# re-exported (export_viser.py + build_flashdex_html.py) when that run's latest eval is >= 50M env steps past the
# last export, or its mean AdaDex progress is >= 10 points above it, or the run in CONF changed. Each export prints
# "[auto] exported ...".
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ST=/data/sky/projects/CounterfactualHOI
B=$ST/build/FlashDexRetarget/logs/flashsac
PY=$ST/conda_envs/flashdex/bin/python
OUT=/home/sky/sky_workdir/visualrecognition/outputs/flashdex_viser
CONF=$ST/runs/flashdex/auto_refresh.conf
STATE=$ST/runs/flashdex/auto_refresh.state           # "<run> <step> <mean progress>" of the last export
NOTE="Still training: numbers update with each export. AdaDex progress counts until the object first leaves 5 cm or 20° of the reconstruction. The reconstructed object orientation is off by 12-22° on average, and several reconstructed rest poses are not stable on the table in simulation (they settle 15-23°), so progress stays well below the position-only number."
while true; do
  for r in "$B"/*/; do
    [ -d "$r/evals" ] && timeout 1200 "$PY" "$HERE/eval_curves.py" "$r" > /dev/null 2>&1
  done
  if [ -f "$CONF" ]; then
    read -r run data refine < "$CONF"
    read -r step prog < <("$PY" -c "
import json, sys
c = json.load(open('$B/$run/curves.json')); s = max(c, key=int)
print(s, c[s]['summary']['track']['progress'])" 2>/dev/null || echo "0 0")
    last_run=""; last_step=0; last_prog=0
    [ -f "$STATE" ] && read -r last_run last_step last_prog < "$STATE"
    go=$("$PY" -c "print(int('$run' != '$last_run' or $step - $last_step >= 50_000_000 or $prog - $last_prog >= 0.10))")
    if [ "$step" != 0 ] && [ "$go" = 1 ]; then
      arch=$(ls -d "$B/$run/evals/step$step")
      if bash "$HERE/export_viser.sh" --dataset "$ST/runs/flashdex/$data" --archive "$arch" --refine "$refine" --out "$OUT" > /dev/null 2>&1 \
         && python3 "$HERE/build_flashdex_html.py" --out "$OUT" --note "$NOTE" > /dev/null 2>&1; then
        echo "$run $step $prog" > "$STATE"
        echo "[auto] exported $run step $step mean progress $prog at $(date '+%H:%M')"
      else
        echo "[auto] export FAILED for $run step $step at $(date '+%H:%M')"
      fi
    fi
  fi
  sleep 300
done
