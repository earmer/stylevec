#!/bin/bash
set -e

LOCAL_DIR="/Users/earmercarey/stylevec"
REMOTE_DIR="/root/autodl-tmp/stylevec"
REMOTE_HOST="root@connect.westd.seetacloud.com"
SSH_PORT=14914

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
target/
EOF

# ── Sync ──────────────────────────────────────────────────────────────────────
rsync -avzP --delete --exclude-from="$EXCLUDE_FILE" \
    -e "ssh -p $SSH_PORT" \
    "$LOCAL_DIR/" \
    "$REMOTE_HOST:$REMOTE_DIR/"
rm "$EXCLUDE_FILE"

# ── Remote setup & train ──────────────────────────────────────────────────────
ssh -p "$SSH_PORT" "$REMOTE_HOST" << 'ENDSSH'
  set -e
  cd /root/autodl-tmp/stylevec

  echo "=== Installing dependencies ==="
  uv sync

  echo "=== Stopping any existing training ==="
  pkill -9 -f "train.py" 2>/dev/null || true
  sleep 2   # wait for GPU memory release

  echo "=== Starting training ==="
  mkdir -p /root/tf-logs
  nohup uv run python paper_replication/train.py \
    --use-local-data \
    --batch 64 \
    --epochs 10 \
    --log-dir /root/tf-logs \
    > paper_replication/train.log 2>&1 &

  echo "=== Waiting for training to start (up to 30s) ==="
  for i in $(seq 1 30); do
    if grep -q "Epoch" paper_replication/train.log 2>/dev/null; then
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
