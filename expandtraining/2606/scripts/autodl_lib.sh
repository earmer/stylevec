# Sourced by the Mac-side scripts that drive an AutoDL instance. Run from the repo root.

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

instance_status() {
  autodl_cli info "$1" --json | instance_field 'd["instance"]["status"]'
}

# Start an instance, retrying every START_RETRY_SECONDS (default 15) for as long as it takes: a Pro box
# with no free GPU on its host refuses to start until one frees up. Errors that waiting cannot fix (bad
# token, unknown instance) stop it. Billing begins when a start succeeds.
start_with_retry() {
  local id="$1" interval="${START_RETRY_SECONDS:-15}" tries=0 out
  while true; do
    tries=$((tries + 1))
    if out="$(autodl_cli start "$id" --wait 2>&1)"; then
      echo "== $id started (try $tries; billing until it is stopped)"
      return 0
    fi
    if grep -qiE 'token|unauthori[sz]ed|401|403|not found|不存在|无权|登录' <<<"$out"; then
      echo "start $id failed, not retrying: $(tail -1 <<<"$out")" >&2
      return 1
    fi
    [[ "$(instance_status "$id" 2>/dev/null)" == running ]] && { echo "== $id is running (try $tries)"; return 0; }
    echo "$(date +%T) start $id: $(tail -1 <<<"$out" | cut -c1-160); retrying in ${interval}s (try $tries)"
    sleep "$interval"
  done
}

# Sets HOST and SSH_PORT for a running instance.
resolve_ssh() {
  local info
  info="$(autodl_cli info "$1" --json)"
  HOST="$(instance_field 'd["snapshot"]["ssh"]["user"] + "@" + d["snapshot"]["ssh"]["host"]' <<<"$info")"
  SSH_PORT="$(instance_field 'd["snapshot"]["ssh"]["port"]' <<<"$info")"
}

wait_for_ssh() {
  local _
  for _ in $(seq 1 18); do
    ssh -o BatchMode=yes -o ConnectTimeout=10 -p "$SSH_PORT" "$HOST" true 2>/dev/null && return 0
    sleep 10
  done
  echo "ssh to $HOST:$SSH_PORT did not come up" >&2
  return 1
}
