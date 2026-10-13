#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

# CONFIG overrides the config, e.g. an ablation arm from scripts/ablation.py plan.
CONFIG="${CONFIG:-expandtraining/2606/configs/remote_5090x2.yaml}"
RUN_NAME="${1:?usage: validate.sh <run_name> [checkpoint_dir]}"
CHECKPOINT="${2:-expandtraining/2606/outputs/runs/${RUN_NAME}/checkpoints/latest}"

uv run python expandtraining/2606/scripts/validate.py \
  --config "$CONFIG" \
  --run-name "$RUN_NAME" \
  --resume "$CHECKPOINT"

