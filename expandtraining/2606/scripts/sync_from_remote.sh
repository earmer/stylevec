#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

REMOTE="${1:?usage: sync_from_remote.sh user@host:/workspace/stylevec <run_name>}"
RUN_NAME="${2:?usage: sync_from_remote.sh user@host:/workspace/stylevec <run_name>}"

mkdir -p "expandtraining/2606/outputs/runs/${RUN_NAME}"

rsync -a --human-readable --info=progress2 \
  "${REMOTE%/}/expandtraining/2606/outputs/runs/${RUN_NAME}/" \
  "expandtraining/2606/outputs/runs/${RUN_NAME}/"

if [[ ! -s "expandtraining/2606/outputs/runs/${RUN_NAME}/checkpoints/latest/training_state.pt" ]]; then
  echo "synced checkpoint is missing training_state.pt" >&2
  exit 1
fi

