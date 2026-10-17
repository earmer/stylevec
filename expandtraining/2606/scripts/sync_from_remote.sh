#!/usr/bin/env bash
# Pull run results back from a remote box.
#
#   sync_from_remote.sh user@host:/root/stylevec <run-or-spec> ...
#   sync_from_remote.sh --instance pro-xxxx <run-or-spec> ...
#
# Each name is a run name or an ablation spec name (pulls every <spec>__* run and the queue status).
# Pulled: validation/, config.yaml, data_manifest.json, curriculum_schedule.json, queue logs.
# Never pulled: training_state.pt (optimizer state). WITH_ADAPTER=1 also pulls checkpoints/latest/adapter;
# WITH_TB=1 also pulls /root/tf-logs/<run>.
#
# --instance resolves host and port with the autodl CLI. A stopped instance is started for the pull
# (billing while it runs) and stopped again afterwards; a running one is left running.
# Env: SSH_PORT (with user@host form), ROOT (with --instance; default /root/stylevec).
set -euo pipefail

cd "$(dirname "$0")/../../.."

usage="usage: sync_from_remote.sh (user@host:/root/stylevec | --instance <id>) <run-or-spec> ..."
[[ $# -ge 2 ]] || { echo "$usage" >&2; exit 2; }

autodl_cli() {
  # The CLI reads AUTODL_TOKEN; it is set for this one process only, from .env, and never printed.
  local token
  token="$(sed -n 's/^AUTODL_DEV_TOK=//p' .env | head -1 | sed -e 's/^"//' -e 's/"$//')"
  [[ -n "$token" ]] || { echo "AUTODL_DEV_TOK missing from .env" >&2; return 1; }
  PATH="$HOME/Library/pnpm/bin:$PATH" AUTODL_TOKEN="$token" autodl "$@"
}

instance_field() {
  python3 -c 'import json,sys; d=json.load(sys.stdin)["data"]; print(eval(sys.argv[1], {}, {"d": d}))' "$1"
}

INSTANCE=""
started_here=0
if [[ "$1" == "--instance" ]]; then
  INSTANCE="$2"
  shift 2
  [[ $# -ge 1 ]] || { echo "$usage" >&2; exit 2; }
  status="$(autodl_cli info "$INSTANCE" --json | instance_field 'd["instance"]["status"]')"
  if [[ "$status" != "running" ]]; then
    echo "== starting $INSTANCE (was $status; billing until it is stopped again)"
    autodl_cli start "$INSTANCE" --wait >/dev/null
    started_here=1
  fi
  info="$(autodl_cli info "$INSTANCE" --json)"
  HOST="$(instance_field 'd["snapshot"]["ssh"]["user"] + "@" + d["snapshot"]["ssh"]["host"]' <<<"$info")"
  SSH_PORT="$(instance_field 'd["snapshot"]["ssh"]["port"]' <<<"$info")"
  REMOTE_ROOT="${ROOT:-/root/stylevec}"
else
  REMOTE="$1"
  shift
  [[ "$REMOTE" == *:* ]] || { echo "$usage" >&2; exit 2; }
  HOST="${REMOTE%%:*}"
  REMOTE_ROOT="${REMOTE#*:}"
fi
REMOTE_ROOT="${REMOTE_ROOT%/}"

stop_if_started() {
  if (( started_here )); then
    echo "== stopping $INSTANCE"
    autodl_cli stop "$INSTANCE" >/dev/null || echo "WARNING: stop failed; $INSTANCE may still be billing" >&2
  fi
}
trap stop_if_started EXIT

ssh_cmd=(ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6)
[[ -n "${SSH_PORT:-}" ]] && ssh_cmd+=(-p "$SSH_PORT")
# A freshly started box may take a few seconds to accept ssh.
for _ in 1 2 3 4 5 6; do "${ssh_cmd[@]}" -o ConnectTimeout=10 "$HOST" true 2>/dev/null && break; sleep 10; done

filters=(--exclude 'training_state.pt' --exclude '.rsync-partial/')
[[ "${WITH_ADAPTER:-0}" == 1 ]] || filters+=(--exclude 'checkpoints/')
for name in "$@"; do
  filters+=(--include "/${name}/***" --include "/${name}__*/***")
done
filters+=(--exclude '*')

runs_dir="expandtraining/2606/outputs/runs"
mkdir -p "$runs_dir"
echo "== runs: $*"
rsync -a --human-readable --info=progress2 --prune-empty-dirs -e "${ssh_cmd[*]}" \
  "${filters[@]}" "$HOST:$REMOTE_ROOT/$runs_dir/" "$runs_dir/"

# Queue status for spec names (absent for plain run names).
mkdir -p expandtraining/2606/outputs/queue
for name in "$@"; do
  rsync -a -e "${ssh_cmd[*]}" "$HOST:$REMOTE_ROOT/expandtraining/2606/outputs/queue/$name.json" \
    expandtraining/2606/outputs/queue/ 2>/dev/null || true
done

if [[ "${WITH_TB:-0}" == 1 ]]; then
  echo "== tensorboard"
  tb_filters=()
  for name in "$@"; do tb_filters+=(--include "/${name}/***" --include "/${name}__*/***"); done
  mkdir -p expandtraining/2606/tf-logs
  rsync -a -e "${ssh_cmd[*]}" "${tb_filters[@]}" --exclude '*' "$HOST:/root/tf-logs/" expandtraining/2606/tf-logs/
fi

pulled=0
for name in "$@"; do
  for dir in "$runs_dir/$name" "$runs_dir/$name"__*; do
    [[ -d "$dir" ]] || continue
    if [[ -s "$dir/validation/per_rule.csv" ]]; then
      pulled=$((pulled + 1))
    else
      echo "no validation yet: $dir"
    fi
  done
done
echo "$pulled validated run(s) pulled into $runs_dir"
