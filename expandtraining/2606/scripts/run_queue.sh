#!/usr/bin/env bash
# Instance entry point for scripts/run_queue.py; arguments pass through. Run it inside tmux.
set -euo pipefail

cd "$(dirname "$0")/../../.."
export PATH="/root/miniconda3/bin:$PATH"

UV_RUN=(uv run)
if [[ -f .venv/.remote-setup-lock && "$(cat .venv/.remote-setup-lock)" == "$(sha256sum uv.lock 2>/dev/null | cut -d' ' -f1)" ]]; then
  UV_RUN+=(--no-sync)
fi

exec "${UV_RUN[@]}" python expandtraining/2606/scripts/run_queue.py "$@"
