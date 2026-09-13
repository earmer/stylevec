#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export OMP_NUM_THREADS=8

RUN_NAME="${1:?usage: remote_train.sh <run_name> [--resume checkpoint_dir]}"
shift

if command -v uv >/dev/null 2>&1; then
  LAUNCH=(uv run accelerate launch)
else
  LAUNCH=(.venv/bin/python -m accelerate.commands.launch)
fi

"${LAUNCH[@]}" \
  --num_processes 2 \
  --mixed_precision bf16 \
  expandtraining/2606/scripts/train.py \
  --config expandtraining/2606/configs/remote_5090x2.yaml \
  --run-name "$RUN_NAME" \
  "$@"
