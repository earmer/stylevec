#!/usr/bin/env bash
# Start an AutoDL instance, push the code and a spec's datasets, and start the spec's queue in tmux.
#
#   launch_queue.sh <instance-id> <spec.yaml> [run_queue args, e.g. --prune --only staged]
#
# 1. Stage 0a on the Mac (re-plans, builds nothing; fails if a dataset is missing). SKIP_PREFLIGHT=1 skips.
# 2. Starts the instance if it is not running, retrying every 15 s until a GPU is free
#    (START_RETRY_SECONDS changes the interval).
# 3. sync_to_remote.sh with the spec's datasets, then the queue's dry run (Stage 0b), then the queue
#    itself in tmux session `queue`, logging to /root/queue_<spec>.log on the box.
#
# The queue powers the box off when it ends. If this script fails after starting the box itself, it
# stops the box again so nothing bills idle; KEEP_UP=1 leaves it running for debugging.
set -euo pipefail

cd "$(dirname "$0")/../../.."
source expandtraining/2606/scripts/autodl_lib.sh

usage="usage: launch_queue.sh <instance-id> <spec.yaml> [run_queue args]"
[[ $# -ge 2 ]] || { echo "$usage" >&2; exit 2; }
INSTANCE="$1"
SPEC="$2"
shift 2
[[ -f "$SPEC" ]] || { echo "no spec: $SPEC" >&2; exit 2; }
NAME="$(sed -n 's/^name: *//p' "$SPEC" | head -1)"
REMOTE_ROOT="${ROOT:-/root/stylevec}"

if [[ "${SKIP_PREFLIGHT:-0}" != 1 ]]; then
  echo "== Stage 0a"
  uv run python expandtraining/2606/scripts/preflight.py local "$SPEC"
fi
datasets=()
while IFS= read -r d; do datasets+=("$d"); done < <(
  python3 -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))["datasets"]))' \
    "expandtraining/2606/outputs/ablations/$NAME/plan.json")

started_here=0
launched=0
cleanup() {
  if (( started_here && !launched )) && [[ "${KEEP_UP:-0}" != 1 ]]; then
    echo "== launch failed; stopping $INSTANCE" >&2
    autodl_cli stop "$INSTANCE" >/dev/null || echo "WARNING: stop failed; $INSTANCE may still be billing" >&2
  fi
}
trap cleanup EXIT

if [[ "$(instance_status "$INSTANCE")" != running ]]; then
  echo "== starting $INSTANCE"
  start_with_retry "$INSTANCE"
  started_here=1
fi
resolve_ssh "$INSTANCE"
wait_for_ssh
echo "== timed shutdowns"
disarm_shutdown_timers

# Waiting for stock can take long; if the code changed meanwhile, Stage 0b on the box would refuse it.
if [[ "${SKIP_PREFLIGHT:-0}" != 1 ]] && ! uv run python - "$NAME" <<'PY'
import json, sys
sys.path.insert(0, "expandtraining/2606/src")
from ablation import generated_dir
from preflight import code_fingerprint, preflight_path
sys.exit(json.loads(preflight_path(generated_dir(sys.argv[1])).read_text())["code"] != code_fingerprint())
PY
then
  echo "== code changed since Stage 0a; re-running it"
  uv run python expandtraining/2606/scripts/preflight.py local "$SPEC"
fi

echo "== sync to $HOST:$SSH_PORT (datasets: ${datasets[*]:-none})"
SSH_PORT="$SSH_PORT" bash expandtraining/2606/scripts/sync_to_remote.sh "$HOST:$REMOTE_ROOT" "${datasets[@]}"

ssh_cmd=(ssh -o BatchMode=yes -o ServerAliveInterval=30 -p "$SSH_PORT" "$HOST")
log="/root/queue_${NAME}.log"
queue_args="$(printf '%q ' "$SPEC" "$@")"
"${ssh_cmd[@]}" "cd '$REMOTE_ROOT' && tmux has-session -t queue 2>/dev/null" \
  && { echo "a tmux session 'queue' is already running on the box; not starting another" >&2; exit 1; }
echo "== Stage 0b and plan"
"${ssh_cmd[@]}" "cd '$REMOTE_ROOT' && bash expandtraining/2606/scripts/run_queue.sh $queue_args --dry-run"
"${ssh_cmd[@]}" "cd '$REMOTE_ROOT' && tmux new -d -s queue \"bash expandtraining/2606/scripts/run_queue.sh $queue_args 2>&1 | tee -a $log\""
launched=1
sleep 5
"${ssh_cmd[@]}" "tmux ls; tail -3 $log"
echo "== queue running on $INSTANCE; log $log. It powers the box off when it ends."
echo "   pull later: bash expandtraining/2606/scripts/sync_from_remote.sh --instance $INSTANCE $NAME"
