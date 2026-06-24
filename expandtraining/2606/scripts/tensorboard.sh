#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

LOGDIR="${1:-expandtraining/2606/tf-logs}"
uv run tensorboard --logdir "$LOGDIR"

