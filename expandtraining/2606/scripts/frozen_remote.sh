#!/usr/bin/env bash
# On the box: fetch the frozen LMs (once, onto autodl-fs) and run frozen_llm_eval.py, logging to stdout.
#
#   frozen_remote.sh <tag> [frozen_llm_eval args...]
#
# Ends with "FROZEN DONE" or "FROZEN FAILED". The Mac pulls the results and stops the box; if nobody has
# within GRACE_SECONDS (default 1200), the box powers itself off.
set -uo pipefail

cd "$(dirname "$0")/../../.."
TAG="${1:?usage: frozen_remote.sh <tag> [args]}"
shift
PY=/root/stylevec/.venv/bin/python
MODELS=artifacts/base-models

fetch() {  # fetch <repo> <dir>
  [[ -f "$MODELS/$2/config.json" ]] && return 0
  echo "== downloading $1"
  HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 "$PY" - "$1" "$MODELS/$2" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2], allow_patterns=["*.json", "*.safetensors", "*.txt", "merges.txt"])
PY
}

status=0
fetch Qwen/Qwen3-4B-Instruct-2507 qwen-3-4b-instruct && fetch Qwen/Qwen3-1.7B qwen-3-1.7b || status=1
if (( status == 0 )); then
  "$PY" expandtraining/2606/scripts/frozen_llm_eval.py --tag "$TAG" "$@" || status=1
fi
(( status == 0 )) && echo "FROZEN DONE" || echo "FROZEN FAILED"
sleep "${GRACE_SECONDS:-1200}"
bash /usr/bin/shutdown
