#!/usr/bin/env bash
# Run a command in its own process group and kill that group if the root filesystem gets low.
#
#   bash scripts/choir/guarded.sh LOGFILE CMD [ARGS...]
#
# Root ("/") on this pod is shared with other pods; below ~350 GB free kubelet may evict the pod.
# MIN_ROOT_FREE_GB (default 352) is checked every CHECK_SECS (default 30). Only the command's own
# process group is signalled; nothing else is touched.
set -uo pipefail
LOG="$1"; shift
MIN_GB="${MIN_ROOT_FREE_GB:-352}"
CHECK_SECS="${CHECK_SECS:-30}"
free_gb() { df -BG --output=avail / | tail -1 | tr -dc '0-9'; }

if [ "$(free_gb)" -lt "$MIN_GB" ]; then
  echo "[guard] root has $(free_gb) GB free (< $MIN_GB); not starting: $*" | tee -a "$LOG"; exit 75
fi
setsid "$@" >> "$LOG" 2>&1 < /dev/null &
PID=$!
echo "[guard] started pgid $PID: $* (root free $(free_gb) GB, floor $MIN_GB GB)" | tee -a "$LOG"
while kill -0 "$PID" 2>/dev/null; do
  if [ "$(free_gb)" -lt "$MIN_GB" ]; then
    echo "[guard] root free $(free_gb) GB < $MIN_GB GB: stopping pgid $PID" | tee -a "$LOG"
    kill -TERM -- "-$PID" 2>/dev/null; sleep 10; kill -KILL -- "-$PID" 2>/dev/null
    wait "$PID" 2>/dev/null; exit 75
  fi
  sleep "$CHECK_SECS"
done
wait "$PID"; rc=$?
echo "[guard] pgid $PID exited rc=$rc (root free $(free_gb) GB)" | tee -a "$LOG"
exit $rc
