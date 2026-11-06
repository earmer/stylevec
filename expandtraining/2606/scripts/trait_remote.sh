#!/usr/bin/env bash
# On the box: run the Stage 4b fine-tune and the 红楼梦 trait test side by side, logging to stdout.
#
#   trait_remote.sh [--steps N]        (the args go to trait_scan_finetune.py)
#
# Each job has its own log under outputs/trait_remote/. Ends with "TRAIT REMOTE DONE" or "TRAIT REMOTE FAILED". The Mac pulls
# the results and stops the box; if nobody has within GRACE_SECONDS (default 1200), the box powers itself off.
set -uo pipefail

cd "$(dirname "$0")/../../.."
PY=/root/stylevec/.venv/bin/python
LOGS=expandtraining/2606/outputs/trait_remote
mkdir -p "$LOGS"

"$PY" -u expandtraining/2606/scripts/trait_scan_finetune.py "$@" >"$LOGS/stage4b.log" 2>&1 &
stage4b=$!
"$PY" -u expandtraining/2606/scripts/hongloumeng_trait_test.py --batch-size 64 >"$LOGS/hongloumeng.log" 2>&1 &
hongloumeng=$!

status=0
wait "$stage4b" || { echo "stage4b failed"; status=1; }
wait "$hongloumeng" || { echo "hongloumeng failed"; status=1; }
(( status == 0 )) && echo "TRAIT REMOTE DONE" || echo "TRAIT REMOTE FAILED"
sleep "${GRACE_SECONDS:-1200}"
bash /usr/bin/shutdown
