#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

RUN_NAME="${1:?usage: validate.sh <run_name> [checkpoint_dir]}"
CHECKPOINT="${2:-expandtraining/2606/outputs/runs/${RUN_NAME}/checkpoints/latest}"

uv run python expandtraining/2606/scripts/validate.py \
  --config expandtraining/2606/configs/remote_5090x2.yaml \
  --run-name "$RUN_NAME" \
  --resume "$CHECKPOINT"

