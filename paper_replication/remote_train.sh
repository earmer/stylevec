#!/bin/bash
set -e

SYNC_ONLY=false
FETCH_LOGS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --synconly|-s) SYNC_ONLY=true; shift ;;
        --fetchlogs|--fl) FETCH_LOGS=true; shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

LOCAL_DIR="/Users/earmercarey/stylevec"
REMOTE_DIR="/root/autodl-tmp/stylevec"
REMOTE_HOST="root@connect.westd.seetacloud.com"
SSH_PORT=34492
# ── Rsync exclude file ────────────────────────────────────────────────────────
EXCLUDE_FILE=$(mktemp)
cat > "$EXCLUDE_FILE" << 'EOF'
.git/
__pycache__/
*.pyc
.DS_Store
.venv/
checkpoints/
*.pt
*.pth
*.log
.cache/
bookraw/
ao3_zh_kudos_crawl/
ao3_random_subset/
datadelta/
tf-logs/
logs/
target/
EOF

# ── Sync ──────────────────────────────────────────────────────────────────────
rsync -avzP --delete --exclude-from="$EXCLUDE_FILE" \
    -e "ssh -p $SSH_PORT" \
    "$LOCAL_DIR/" \
    "$REMOTE_HOST:$REMOTE_DIR/"
rm "$EXCLUDE_FILE"

# ── Fetch logs from remote ──────────────────────────────────────────────────────
if $FETCH_LOGS; then
    echo "=== Fetching logs from $REMOTE_HOST:/root/tf-logs/ ==="
    LOCAL_LOG_DIR="$LOCAL_DIR/paper_replication/tf-logs"
    mkdir -p "$LOCAL_LOG_DIR"
    rsync -avzP \
        -e "ssh -p $SSH_PORT" \
        "$REMOTE_HOST:/root/tf-logs/" \
        "$LOCAL_LOG_DIR/"
    echo "=== Logs fetched to $LOCAL_LOG_DIR ==="
    echo "  tensorboard --logdir $LOCAL_LOG_DIR"
    exit 0
fi

# ── Remote setup & train ──────────────────────────────────────────────────────
if $SYNC_ONLY; then
    echo "=== Sync-only mode: skipping remote setup & training ==="
    exit 0
fi

ssh -p "$SSH_PORT" "$REMOTE_HOST" << 'ENDSSH'
  set -e
  cd /root/autodl-tmp/stylevec

  RESUME_RUN="20260505-005138"
  RESUME_PREFIX=$(date -d "${RESUME_RUN:0:8} ${RESUME_RUN:9:2}:${RESUME_RUN:11:2}:${RESUME_RUN:13:2}" +%s)
  RESUME_DIR=$(find paper_replication/checkpoints -maxdepth 1 -type d -name "${RESUME_PREFIX}-step-*" \
    | sort -t- -k3,3n \
    | tail -1)
  if [[ ! -d "$RESUME_DIR" || ! -f "$RESUME_DIR/training_state.pt" ]]; then
    echo "ERROR: Missing resume checkpoint for run $RESUME_RUN (prefix $RESUME_PREFIX)"
    exit 1
  fi
  echo "=== Resuming from $RESUME_DIR ==="
  .venv/bin/python - "$RESUME_DIR" "$RESUME_RUN" << 'PY'
import sys
from pathlib import Path

import torch

ckpt_dir = Path(sys.argv[1])
resume_run = sys.argv[2]
state = torch.load(ckpt_dir / "training_state.pt", map_location="cpu", weights_only=False)
print(f"checkpoint_global_step={state.get('global_step')}")

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    log_dir = Path("/root/tf-logs") / resume_run
    accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    max_step = None
    for tag in accumulator.Tags().get("scalars", []):
        for event in accumulator.Scalars(tag):
            max_step = event.step if max_step is None else max(max_step, event.step)
    print(f"tensorboard_max_step={max_step}")
except Exception as exc:
    print(f"tensorboard_max_step=unavailable ({exc})")
PY

  echo "=== Installing dependencies ==="
  uv sync

  echo "=== Stopping any existing training ==="
  pkill -9 -f "train.py" 2>/dev/null || true
  sleep 2   # wait for GPU memory release

  echo "=== Starting training ==="
  mkdir -p /root/tf-logs
  nohup uv run python paper_replication/train.py \
    --use-local-data \
    --resume-from "$RESUME_RUN" \
    --epochs 30 \
    --max-pairs-per-feature 32 \
    --val-every 250 \
    --save-every 1500 \
    --log-dir /root/tf-logs \
    > paper_replication/train.log 2>&1 &

  echo "=== Waiting for training to start (up to 30s) ==="
  for i in $(seq 1 30); do
    if grep -Eq "Resumed from|Epoch" paper_replication/train.log 2>/dev/null; then
      echo "Training started successfully."
      echo "Monitor: tail -f /root/autodl-tmp/stylevec/paper_replication/train.log"
      exit 0
    fi
    sleep 1
  done

  echo "ERROR: Training did not start within 30 seconds. Last log lines:"
  tail -20 paper_replication/train.log
  exit 1
ENDSSH
