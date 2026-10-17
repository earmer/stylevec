#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export OMP_NUM_THREADS=8

# CONFIG overrides the config, e.g. an ablation arm from scripts/ablation.py plan.
# DATASET names a prebuilt dataset (scripts/build_dataset.py); unset builds triplets from raw.
CONFIG="${CONFIG:-expandtraining/2606/configs/remote_5090x2.yaml}"
RUN_NAME="${1:?usage: remote_train.sh <run_name> [--resume checkpoint_dir]}"
shift

# remote_setup.sh installed .venv from the mirror; a plain `uv run` would re-sync from uv.lock's URLs.
UV_RUN=(uv run)
if [[ -f .venv/.remote-setup-lock && "$(cat .venv/.remote-setup-lock)" == "$(sha256sum uv.lock 2>/dev/null | cut -d' ' -f1)" ]]; then
  UV_RUN+=(--no-sync)
fi

if command -v uv >/dev/null 2>&1; then
  LAUNCH=("${UV_RUN[@]}" accelerate launch)
else
  LAUNCH=(.venv/bin/python -m accelerate.commands.launch)
fi

"${LAUNCH[@]}" \
  --num_processes 2 \
  --mixed_precision bf16 \
  expandtraining/2606/scripts/train.py \
  --config "$CONFIG" \
  --run-name "$RUN_NAME" \
  ${DATASET:+--dataset "$DATASET"} \
  "$@"
