#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

uv run python expandtraining/2606/scripts/train.py \
  --config expandtraining/2606/configs/dryrun.yaml \
  --run-name dryrun_local

