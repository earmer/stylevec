#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

# CONFIG overrides the config, e.g. an ablation arm from scripts/ablation.py plan.
# DATASET names a prebuilt dataset (scripts/build_dataset.py); unset builds triplets from raw.
CONFIG="${CONFIG:-expandtraining/2606/configs/remote_5090x2.yaml}"
RUN_NAME="${1:?usage: validate.sh <run_name> [checkpoint_dir]}"
CHECKPOINT="${2:-expandtraining/2606/outputs/runs/${RUN_NAME}/checkpoints/latest}"

# remote_setup.sh installed .venv from the mirror; a plain `uv run` would re-sync from uv.lock's URLs.
UV_RUN=(uv run)
if [[ -f .venv/.remote-setup-lock && "$(cat .venv/.remote-setup-lock)" == "$(sha256sum uv.lock 2>/dev/null | cut -d' ' -f1)" ]]; then
  UV_RUN+=(--no-sync)
fi

"${UV_RUN[@]}" python expandtraining/2606/scripts/validate.py \
  --config "$CONFIG" \
  --run-name "$RUN_NAME" \
  --resume "$CHECKPOINT" \
  ${DATASET:+--dataset "$DATASET"}

