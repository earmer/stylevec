#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

RUN_NAME="${1:?usage: remote_train.sh <run_name> [--resume checkpoint_dir]}"
shift

uv run accelerate launch \
  --num_processes 2 \
  --mixed_precision bf16 \
  expandtraining/2606/scripts/train.py \
  --config expandtraining/2606/configs/remote_5090x2.yaml \
  --run-name "$RUN_NAME" \
  "$@"

